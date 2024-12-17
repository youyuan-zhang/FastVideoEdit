import glob
import os
import numpy as np
from pathlib import Path
import torch
import torch.nn as nn
import torchvision.transforms as T
import argparse
from PIL import Image
import yaml
from diffusers.utils.torch_utils import randn_tensor
from tqdm import tqdm
from transformers import logging
from diffusers import DDIMScheduler, StableDiffusionPipeline, LCMScheduler

from attn_ctrl import *
from util import save_video, seed_everything

from attention_refine import AttentionRefine, LocalBlend

# suppress partial model loading warning
logging.set_verbosity_error()

VAE_BATCH_SIZE = 10


def BCS(scheduler, x_s, x_t, timestep, e_s, e_t, x_0, noise, eta, to_next=True):
    if scheduler.num_inference_steps is None:
        raise ValueError(
            "Number of inference steps is 'None', you need to run 'set_timesteps' after creating the scheduler"
        )

    if scheduler.step_index is None:
        scheduler._init_step_index(timestep)

    prev_step_index = scheduler.step_index + 1
    if prev_step_index < len(scheduler.timesteps):
        prev_timestep = scheduler.timesteps[prev_step_index]
    else:
        prev_timestep = timestep

    alpha_prod_t = scheduler.alphas_cumprod[timestep]
    alpha_prod_t_prev = (
        scheduler.alphas_cumprod[prev_timestep] if prev_timestep >= 0 else scheduler.final_alpha_cumprod
    )
    beta_prod_t = 1 - alpha_prod_t
    beta_prod_t_prev = 1 - alpha_prod_t_prev
    variance = beta_prod_t_prev
    std_dev_t = eta * variance
    noise = std_dev_t ** (0.5) * noise

    e_c = (x_s - alpha_prod_t ** (0.5) * x_0) / (1 - alpha_prod_t) ** (0.5)

    pred_x0 = x_0 + ((x_t - x_s) - beta_prod_t ** (0.5) * (e_t - e_s)) / alpha_prod_t ** (0.5)
    eps = (e_t - e_s) + e_c
    dir_xt = (beta_prod_t_prev - std_dev_t) ** (0.5) * eps

    # Noise is not used for one-step sampling.
    if len(scheduler.timesteps) > 1:
        prev_xt = alpha_prod_t_prev ** (0.5) * pred_x0 + dir_xt + noise
        prev_xs = alpha_prod_t_prev ** (0.5) * x_0 + dir_xt + noise
    else:
        prev_xt = pred_x0
        prev_xs = x_0

    if to_next:
        scheduler._step_index += 1
    return prev_xs, prev_xt, pred_x0


class FVEPipeline(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.device = config["device"]

        sd_version = config["sd_version"]
        self.sd_version = sd_version
        if sd_version == '1.5':
            model_key = "../stable-diffusion-v1-5"
        elif sd_version == 'lcm':
            model_key = "SimianLuo/LCM_Dreamshaper_v7"
        else:
            raise ValueError(f'Stable-diffusion version {sd_version} not supported.')

        # Create SD models
        print('Loading SD model')

        pipe = StableDiffusionPipeline.from_pretrained(model_key, torch_dtype=torch.float16).to("cuda")
        # pipe.enable_xformers_memory_efficient_attention()

        self.vae = pipe.vae
        self.tokenizer = pipe.tokenizer
        self.text_encoder = pipe.text_encoder
        self.unet = pipe.unet

        self.scheduler = LCMScheduler.from_pretrained(model_key, subfolder="scheduler")
        self.scheduler.set_timesteps(config["n_timesteps"], device=self.device)
        print('SD model loaded')
        
        self.paths, self.frames, self.latents = self.get_data()
        self.target_embeds = self.get_text_embeds(config["prompt"], config["negative_prompt"])

    # Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion_img2img.StableDiffusionImg2ImgPipeline.get_timesteps
    def get_timesteps(self, num_inference_steps, strength, device):
        # get the original timestep using init_timestep
        init_timestep = min(int(num_inference_steps * strength), num_inference_steps)

        t_start = max(num_inference_steps - init_timestep, 0)
        timesteps = self.scheduler.timesteps[t_start * self.scheduler.order:]

        return timesteps, num_inference_steps - t_start

    @torch.no_grad()
    def get_text_embeds(self, prompt, negative_prompt, batch_size=1):
        # Tokenize text and get embeddings
        text_input = self.tokenizer(prompt, padding='max_length', max_length=self.tokenizer.model_max_length,
                                    truncation=True, return_tensors='pt')
        text_embeddings = self.text_encoder(text_input.input_ids.to(self.device))[0]

        # Do the same for unconditional embeddings
        uncond_input = self.tokenizer(negative_prompt, padding='max_length', max_length=self.tokenizer.model_max_length,
                                      return_tensors='pt')

        uncond_embeddings = self.text_encoder(uncond_input.input_ids.to(self.device))[0]

        # Cat for final embeddings
        text_embeddings = torch.cat([uncond_embeddings] * batch_size + [text_embeddings] * batch_size)
        return text_embeddings

    @torch.no_grad()
    def encode_imgs(self, imgs, batch_size=VAE_BATCH_SIZE, deterministic=False):
        imgs = 2 * imgs - 1
        latents = []
        for i in range(0, len(imgs), batch_size):
            posterior = self.vae.encode(imgs[i:i + batch_size]).latent_dist
            latent = posterior.mean if deterministic else posterior.sample()
            latents.append(latent * 0.18215)
        latents = torch.cat(latents)
        return latents

    @torch.no_grad()
    def decode_latents(self, latents, batch_size=VAE_BATCH_SIZE):
        latents = 1 / 0.18215 * latents
        imgs = []
        for i in range(0, len(latents), batch_size):
            imgs.append(self.vae.decode(latents[i:i + batch_size]).sample)
        imgs = torch.cat(imgs)
        imgs = (imgs / 2 + 0.5).clamp(0, 1)
        return imgs

    def get_data(self):
        # load frames
        paths = [os.path.join(config["data_path"], "%05d.jpg" % idx) for idx in
                 range(self.config["n_frames"])]
        if not os.path.exists(paths[0]):
            paths = [os.path.join(config["data_path"], "%05d.png" % idx) for idx in
                     range(self.config["n_frames"])]
        frames = [Image.open(paths[idx]).convert('RGB') for idx in range(self.config["n_frames"])]
        if frames[0].size[0] == frames[0].size[1]:
            frames = [frame.resize((512, 512), resample=Image.Resampling.LANCZOS) for frame in frames]
        frames = torch.stack([T.ToTensor()(frame) for frame in frames]).to(torch.float16).to(self.device)
        save_video(frames, f'{self.config["output_path"]}/input_fps10.mp4', fps=10)
        # encode to latents
        latents = self.encode_imgs(frames, deterministic=True).to(torch.float16).to(self.device)
        return paths, frames, latents

    @torch.no_grad()
    def denoise_step(self, source_latents, latents, mutual_latents, clean_latents, t, indices):
        latent_model_input = torch.cat([source_latents, latents, mutual_latents,
                                        source_latents, latents, mutual_latents,])

        register_time(self, t.item())

        # compute text embeddings
        text_embed_input = torch.cat([self.source_embeds[0].repeat(len(indices), 1, 1),
                                      self.target_embeds[0].repeat(len(indices), 1, 1),
                                      self.source_embeds[0].repeat(len(indices), 1, 1),
                                      self.source_embeds[1].repeat(len(indices), 1, 1),
                                      self.target_embeds[1].repeat(len(indices), 1, 1),
                                      self.source_embeds[1].repeat(len(indices), 1, 1),])

        # apply the denoising network
        noise_pred = self.unet(latent_model_input, t, encoder_hidden_states=text_embed_input)['sample']

        source_noise_pred_uncond, noise_pred_uncond, mutual_noise_pred_uncond,\
            source_noise_pred_text, noise_pred_text, mutual_noise_pred_text = noise_pred.chunk(6)
        source_noise_pred = source_noise_pred_text
        noise_pred = noise_pred_uncond + \
            self.config["guidance_scale"] * (noise_pred_text - noise_pred_uncond)
        mutual_noise_pred = mutual_noise_pred_uncond + \
            self.config["guidance_scale"] * (mutual_noise_pred_text - mutual_noise_pred_uncond)

        # compute the denoising step with the reference model
        noise = torch.randn(latents.shape, dtype=latents.dtype, device=latents.device)

        _, latents, pred_x0 = BCS(
            self.scheduler, source_latents,
            latents, t,
            source_noise_pred, noise_pred,
            clean_latents, noise=noise,
            eta=1, to_next=False,
        )
        source_latents, mutual_latents, pred_xm = BCS(
            self.scheduler, source_latents,
            mutual_latents, t,
            source_noise_pred, mutual_noise_pred,
            clean_latents, noise=noise,
            eta=1, to_next=False,
        )
        return source_latents, latents, mutual_latents, pred_x0

    @torch.autocast(dtype=torch.float16, device_type='cuda')
    def batched_denoise_step(self, source_latents, latents, mutual_latents, t, indices):
        batch_size = self.config["batch_size"]
        denoised_source_latents_list = []
        denoised_latents_list = []
        denoised_mutual_latents_list = []
        pred_x0_list = []
        pivotal_idx = torch.randint(batch_size, (len(latents) // batch_size,)) + torch.arange(0, len(latents),
                                                                                              batch_size)

        register_pivotal(self, True)
        self.denoise_step(source_latents[pivotal_idx], latents[pivotal_idx], mutual_latents[pivotal_idx],
                          self.latents[pivotal_idx], t,
                          indices[pivotal_idx])
        register_pivotal(self, False)
        for i, b in enumerate(range(0, len(latents), batch_size)):
            register_batch_idx(self, i)
            denoised_source_latents, denoised_latents, denoised_mutual_latents, pred_x0 = \
                self.denoise_step(source_latents[b:b + batch_size],
                                  latents[b:b + batch_size],
                                  mutual_latents[b:b + batch_size],
                                  self.latents[b:b + batch_size], t,
                                  indices[b:b + batch_size])
            denoised_source_latents_list.append(denoised_source_latents)
            denoised_latents_list.append(denoised_latents)
            denoised_mutual_latents_list.append(denoised_mutual_latents)
            pred_x0_list.append(pred_x0)
        denoised_source_latents = torch.cat(denoised_source_latents_list)
        denoised_latents = torch.cat(denoised_latents_list)
        denoised_mutual_latents = torch.cat(denoised_mutual_latents_list)
        pred_x0 = torch.cat(pred_x0_list)
        
        self.controller.cur_step += 1
        self.controller.between_steps()
        
        alpha_prod_t = self.scheduler.alphas_cumprod[t]
        denoised_mutual_latents, denoised_latents = self.controller.step_callback(
            self.controller.cur_step - 1, t,
            denoised_source_latents, denoised_latents, denoised_mutual_latents,
            alpha_prod_t)
            
        return denoised_source_latents, denoised_latents, denoised_mutual_latents, pred_x0

    def init_method(self, conv_injection_t, qk_injection_t, source_prompt, target_prompt, change, local):
        self.qk_injection_timesteps = self.scheduler.timesteps[:qk_injection_t] if qk_injection_t >= 0 else []
        self.conv_injection_timesteps = self.scheduler.timesteps[:conv_injection_t] if conv_injection_t >= 0 else []
        
        self.pnp_inversion_prompt = source_prompt
        self.source_embeds = self.get_text_embeds(self.pnp_inversion_prompt, config["negative_prompt"])
        source_prompt = self.pnp_inversion_prompt
        target_prompt = config["prompt"]
        if type(local) == float:
            local = ""
        local = local
        mutual = [""]
        if type(change) == float:
            change = ""
        change = change
        smask = [""]
        print(source_prompt)
        print(target_prompt)
        print(change)
        print(local)
        
        local_blend = LocalBlend(thresh_e=config["thresh_e"], thresh_m=config["thresh_m"], save_inter=True, mask_replace=config["mask_replace"])
        self.controller = AttentionRefine([source_prompt, target_prompt], [[local, mutual]], change, smask,
                                          self.tokenizer, self.text_encoder,
                                          15, 0, config["cross_replace"], config["self_replace"], local_blend=local_blend,
                                          p2p_scale=config["p2p_scale"])
        register_attention_control(self.unet, self.controller)
        set_tokenflow(self.unet)

    def edit_video(self, source_prompt, target_prompt, change, local):
        os.makedirs(f'{self.config["output_path"]}/img_ode', exist_ok=True)
        pnp_f_t = int(self.config["n_timesteps"] * self.config["pnp_f_t"])
        pnp_attn_t = int(self.config["n_timesteps"] * self.config["pnp_attn_t"])
        self.init_method(conv_injection_t=pnp_f_t, qk_injection_t=pnp_attn_t,
                         source_prompt=source_prompt, target_prompt=target_prompt, change=change, local=local)
        
        timesteps, num_inference_steps = self.get_timesteps(self.config["n_timesteps"], 1, self.device)
        latents = randn_tensor(self.latents.shape, device=self.device, dtype=torch.float16)
        edited_frames = self.sample_loop(latents, torch.arange(self.config["n_frames"]), timesteps)

        save_video(edited_frames, f'{self.config["output_path"]}/FVE_fps_10.mp4')
        print('Done!')

    def sample_loop(self, x, indices, timesteps):
        os.makedirs(f'{self.config["output_path"]}/img_ode', exist_ok=True)
        source_latents = x
        latents = x
        mutual_latents = x
        for i, t in enumerate(tqdm(self.scheduler.timesteps, desc="Sampling")):
            source_latents, latents, mutual_latents, pred_x0 = self.batched_denoise_step(source_latents, latents, mutual_latents, t, indices)
            self.scheduler._step_index += 1

        decoded_latents = self.decode_latents(pred_x0)
        for i in range(len(decoded_latents)):
            T.ToPILImage()(decoded_latents[i]).save(f'{self.config["output_path"]}/img_ode/%05d.png' % i)

        return decoded_latents


def run(config, source_prompt, target_prompt, change, local):
    seed_everything(config["seed"])
    editor = FVEPipeline(config)
    editor.edit_video(source_prompt, target_prompt, change, local)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', type=str, default='configs/config_teaser_4_2.yaml')
    opt = parser.parse_args()
    with open(opt.config_path, "r") as f:
        config = yaml.safe_load(f)
    
    video_name = config["data_path"].split("/")[-1]
    os.makedirs(config["output_path"], exist_ok=True)
    
    source_prompt = config["inv_prompt"]
    target_prompt = config["prompt"]
    change = config["change"]
    local = config["local"]
    
    run(config, source_prompt, target_prompt, change, local)
