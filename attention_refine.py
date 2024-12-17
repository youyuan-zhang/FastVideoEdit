import abc
import torch
from typing import Optional
from PIL import Image
import torch.nn.functional as nnf
import os
import numpy as np
import seq_aligner

LOW_RESOURCE = False
MAX_NUM_WORDS = 77
device = "cuda"
torch_dtype = torch.float16

class LocalBlend:

    def get_mask(self, x_t, maps, word_idx, thresh, i):
        maps = maps * word_idx.reshape(1, 1, 1, 1, 1, -1)
        maps = (maps[:, :, :, :, :, 1:self.len - 1]).mean(1, keepdim=True)
        maps = (maps).max(-1)[0]
        maps = maps.squeeze(1)
        maps = nnf.interpolate(maps, size=(x_t.shape[2:]))
        maps = maps / maps.max(2, keepdim=True)[0].max(3, keepdim=True)[0]
        mask = maps > thresh
        return mask

    def __call__(self, i, x_s, x_t, x_m, attention_store, alpha_prod, temperature=0.15, use_xm=False):
        maps = []
        for idx, map_down in enumerate(attention_store["down_cross"]):
            if idx % 4 == 2 or idx % 4 == 3:
                maps.append(map_down)
        for idx, map_up in enumerate(attention_store["up_cross"]):
            if idx % 6 < 3:
                maps.append(map_up)
        h, w = x_t.shape[2], x_t.shape[3]
        h, w = ((h + 1) // 2 + 1) // 2, ((w + 1) // 2 + 1) // 2
        maps = [
            item.reshape(2, -1, 8, 1, h, w,
                         MAX_NUM_WORDS) for item in maps]
        bs = maps[0].shape[1]
        num_b = len(maps) // 5
        maps_list = []
        for idx_b in range(num_b):
            maps_list.append(torch.cat(maps[idx_b * 5:idx_b *5 + 5], dim=2))
        maps = torch.cat(maps_list, dim=1)
        maps_t = maps[0, :]
        maps_s = maps[1, :]
        mask_c = self.get_mask(x_s, maps_s, self.alpha_c, self.thresh_e, i)
        mask_e = self.get_mask(x_t, maps_t, self.alpha_e, self.thresh_m, i)
        
        if i < self.mask_replace:
            return x_m, x_t

        if self.alpha_c.sum() == 0:
            x_t_out = x_t
        else:
            x_t_out = torch.where(mask_c, x_t, x_m)
        x_t_out = torch.where(torch.logical_not(mask_c), x_s, x_t_out)

        return x_m, x_t_out

    def __init__(self, thresh_e=0.3, thresh_m=0.3, mask_replace=100, save_inter=False):
        self.thresh_e = thresh_e
        self.thresh_m = thresh_m
        self.save_inter = save_inter
        self.mask_replace = mask_replace

    def set_map(self, ms, alpha, alpha_e, alpha_m, alpha_c, len):
        self.m = ms
        self.alpha = alpha
        self.alpha_e = alpha_e
        self.alpha_m = alpha_m
        self.alpha_c = alpha_c
        alpha_me = alpha_e.to(torch.bool) & alpha_m.to(torch.bool)
        self.alpha_me = alpha_me.to(torch.float)
        self.len = len

class AttentionRefine(abc.ABC):
    @property
    def num_uncond_att_layers(self):
        return self.num_att_layers if LOW_RESOURCE else 0

    def step_callback(self, i, t, x_s, x_t, x_m, alpha_prod):
        if (self.local_blend is not None) and (i > 0):
            use_xm = (self.cur_step + self.start_steps + 1 == self.num_steps)
            x_m, x_t = self.local_blend(i, x_s, x_t, x_m, self.attention_store, alpha_prod, use_xm=use_xm)
        return x_m, x_t

    def replace_self_attention(self, attn_base, att_replace):
        if att_replace.shape[2] <= 16 ** 2:
            return attn_base.unsqueeze(0).expand(att_replace.shape[0], *attn_base.shape)
        else:
            return att_replace

    def replace_cross_attention(self, attn_masa, att_replace):
        temp = 1 - self.alphas
        attn_masa_replace = attn_masa[:, :, :, self.mapper[0]]
        attn_replace = attn_masa_replace * self.alphas + \
                       att_replace * temp * self.p2p_scale
        return attn_replace

    def attn_batch(self, q, k, v, sim, attn, is_cross, place_in_unet, num_heads, **kwargs):
        b = q.shape[0] // num_heads

        sim = torch.einsum("h i d, h j d -> h i j", q, k) * kwargs.get("scale")
        attn = sim.softmax(-1)
        out = torch.einsum("h i j, h j d -> h i d", attn, v)
        return out

    def self_attn_forward(self, q, k, v, num_heads):
        qu, qc = q.chunk(2)
        ku, kc = k.chunk(2)
        vu, vc = v.chunk(2)
        batch_size = q.shape[0] // num_heads // 6
        if (self.self_replace_steps <= ((self.cur_step + self.start_steps + 1) * 1.0 / self.num_steps)):
            qu = torch.cat([qu[:batch_size * num_heads * 2], qu[batch_size * num_heads:batch_size * num_heads * 2]])
            qc = torch.cat([qc[:batch_size * num_heads * 2], qc[batch_size * num_heads:batch_size * num_heads * 2]])
            ku = torch.cat([ku[:batch_size * num_heads * 2], ku[:batch_size * num_heads]])
            kc = torch.cat([kc[:batch_size * num_heads * 2], kc[:batch_size * num_heads]])
            vu = torch.cat([vu[:batch_size * num_heads * 2], vu[:batch_size * num_heads]])
            vc = torch.cat([vc[:batch_size * num_heads * 2], vc[:batch_size * num_heads]])
            pass
        else:
            qu = torch.cat([qu[:batch_size * num_heads], qu[:batch_size * num_heads], qu[:batch_size * num_heads]])
            qc = torch.cat([qc[:batch_size * num_heads], qc[:batch_size * num_heads], qc[:batch_size * num_heads]])
            ku = torch.cat([ku[:batch_size * num_heads], ku[:batch_size * num_heads], ku[:batch_size * num_heads]])
            kc = torch.cat([kc[:batch_size * num_heads], kc[:batch_size * num_heads], kc[:batch_size * num_heads]])
            vu = torch.cat([vu[:batch_size * num_heads * 2], vu[:batch_size * num_heads]])
            vc = torch.cat([vc[:batch_size * num_heads * 2], vc[:batch_size * num_heads]])
            pass
        
        qu_s = qu[:batch_size * num_heads]
        qu_t = qu[batch_size * num_heads:batch_size * num_heads * 2]
        qu_m = qu[batch_size * num_heads * 2:batch_size * num_heads * 3]
        qc_s = qc[:batch_size * num_heads]
        qc_t = qc[batch_size * num_heads:batch_size * num_heads * 2]
        qc_m = qc[batch_size * num_heads * 2:batch_size * num_heads * 3]
        
        ku_s = ku[:batch_size * num_heads]
        ku_t = ku[batch_size * num_heads:batch_size * num_heads * 2]
        ku_m = ku[batch_size * num_heads * 2:batch_size * num_heads * 3]
        kc_s = kc[:batch_size * num_heads]
        kc_t = kc[batch_size * num_heads:batch_size * num_heads * 2]
        kc_m = kc[batch_size * num_heads * 2:batch_size * num_heads * 3]
        
        vu_s = vu[:batch_size * num_heads]
        vu_t = vu[batch_size * num_heads:batch_size * num_heads * 2]
        vu_m = vu[batch_size * num_heads * 2:batch_size * num_heads * 3]
        vc_s = vc[:batch_size * num_heads]
        vc_t = vc[batch_size * num_heads:batch_size * num_heads * 2]
        vc_m = vc[batch_size * num_heads * 2:batch_size * num_heads * 3]

        return [qu_s, qu_t, qu_m, qc_s, qc_t, qc_m], [ku_s, ku_t, ku_m, kc_s, kc_t, kc_m], [vu_s, vu_t, vu_m, vc_s, vc_t, vc_m]

    def forward(self, attn, is_cross: bool, place_in_unet: str, num_heads, is_pivotal):
        if is_cross:
            h = num_heads
            batch_size = attn.shape[0] // num_heads // 3
            attn = attn.reshape(3 * batch_size, h, *attn.shape[1:])
            attn_base, attn_repalce, attn_masa = attn[:batch_size], attn[batch_size: 2 * batch_size], attn[2 * batch_size:]
            attn_base_store = self.replace_cross_attention(attn_base, attn_repalce)
            if (self.cross_replace_steps >= ((self.cur_step + self.start_steps + 1) * 1.0 / self.num_steps)):
                attn[batch_size: 2 * batch_size] = attn_base_store
            attn_store = torch.cat([attn_base_store, attn_base])
            attn = attn.reshape(3 * batch_size * h, *attn.shape[2:])
            attn_store = attn_store.reshape(2 * batch_size * h, *attn_store.shape[2:])

            key = f"{place_in_unet}_{'cross' if is_cross else 'self'}"
            if attn_store.shape[1] <= 32 ** 2 and not is_pivotal:  # avoid memory overhead
                self.step_store[key].append(attn_store)

        return attn

    def __call__(self, attn, is_cross: bool, place_in_unet: str, num_heads, is_pivotal):
        if self.cur_att_layer >= self.num_uncond_att_layers:
            if LOW_RESOURCE:
                attn = self.forward(attn, is_cross, place_in_unet, num_heads, is_pivotal)
            else:
                h = attn.shape[0]
                attn[h // 2:] = self.forward(attn[h // 2:], is_cross, place_in_unet, num_heads, is_pivotal)
        self.cur_att_layer += 1
        if self.cur_att_layer == self.num_att_layers // 2 + self.num_uncond_att_layers:
            self.cur_att_layer = 0
        return attn

    @staticmethod
    def get_empty_store():
        return {"down_cross": [], "mid_cross": [], "up_cross": [],
                "down_self": [], "mid_self": [], "up_self": []}

    def between_steps(self):
        if len(self.attention_store) == 0:
            self.attention_store = self.step_store
        else:
            for key in self.attention_store:
                for i in range(len(self.attention_store[key])):
                    self.attention_store[key][i] += self.step_store[key][i]
        self.step_store = self.get_empty_store()

    def get_average_attention(self):
        average_attention = {key: [item / self.cur_step for item in self.attention_store[key]] for key in
                             self.attention_store}
        return average_attention

    def reset(self):
        self.cur_step = 0
        self.cur_att_layer = 0

        self.step_store = self.get_empty_store()
        self.attention_store = {}

    def __init__(self, prompts, prompt_specifiers, change, smask,
                 tokenizer, encoder,
                 num_steps: int, start_steps: int, cross_replace_steps: float,
                 self_replace_steps: float,
                 local_blend: Optional[LocalBlend] = None,
                 p2p_scale: float = 1):
        self.cur_step = 0
        self.num_att_layers = -1
        self.cur_att_layer = 0

        self.step_store = self.get_empty_store()
        self.attention_store = {}

        self.self_replace_steps = self_replace_steps
        self.cross_replace_steps = cross_replace_steps
        self.num_steps = num_steps
        self.start_steps = start_steps
        self.local_blend = local_blend

        self.mapper, alphas, ms, alpha_c, alpha_e, alpha_m = seq_aligner.get_refinement_mapper(prompts, prompt_specifiers,
                                                                                      change, smask,
                                                                                      tokenizer, encoder, device)
        self.mapper, alphas, ms = self.mapper.to(device), alphas.to(device).to(torch_dtype), ms.to(device).to(
            torch_dtype)
        self.alphas = alphas.reshape(alphas.shape[0], 1, 1, alphas.shape[1])
        self.ms = ms.reshape(ms.shape[0], 1, 1, ms.shape[1])
        ms = ms.to(device)
        alpha_e = alpha_e.to(device)
        alpha_m = alpha_m.to(device)
        alpha_c = alpha_c.to(device)
        t_len = len(tokenizer(prompts[1])["input_ids"])
        self.local_blend.set_map(ms, alphas, alpha_e, alpha_m, alpha_c, t_len)
        self.p2p_scale = p2p_scale