from typing import Type
import torch
import os

from util import isinstance_str, batch_cosine_sim

def register_pivotal(diffusion_model, is_pivotal):
    for _, module in diffusion_model.named_modules():
        # If for some reason this has a different name, create an issue and I'll fix it
        if isinstance_str(module, "BasicTransformerBlock"):
            setattr(module, "pivotal_pass", is_pivotal)
        if module.__class__.__name__ == "Attention":
            module.processor.is_pivotal = is_pivotal
            
def register_batch_idx(diffusion_model, batch_idx):
    for _, module in diffusion_model.named_modules():
        # If for some reason this has a different name, create an issue and I'll fix it
        if isinstance_str(module, "BasicTransformerBlock"):
            setattr(module, "batch_idx", batch_idx)


def register_time(model, t):
    conv_module = model.unet.up_blocks[1].resnets[1]
    setattr(conv_module, 't', t)
    down_res_dict = {0: [0, 1], 1: [0, 1], 2: [0, 1]}
    up_res_dict = {1: [0, 1, 2], 2: [0, 1, 2], 3: [0, 1, 2]}
    for res in up_res_dict:
        for block in up_res_dict[res]:
            module = model.unet.up_blocks[res].attentions[block].transformer_blocks[0].attn1
            setattr(module, 't', t)
            module = model.unet.up_blocks[res].attentions[block].transformer_blocks[0].attn2
            setattr(module, 't', t)
    for res in down_res_dict:
        for block in down_res_dict[res]:
            module = model.unet.down_blocks[res].attentions[block].transformer_blocks[0].attn1
            setattr(module, 't', t)
            module = model.unet.down_blocks[res].attentions[block].transformer_blocks[0].attn2
            setattr(module, 't', t)
    module = model.unet.mid_block.attentions[0].transformer_blocks[0].attn1
    setattr(module, 't', t)
    module = model.unet.mid_block.attentions[0].transformer_blocks[0].attn2
    setattr(module, 't', t)


def load_source_latents_t(t, latents_path):
    latents_t_path = os.path.join(latents_path, f'noisy_latents_{t}.pt')
    assert os.path.exists(latents_t_path), f'Missing latents at t {t} path {latents_t_path}'
    latents = torch.load(latents_t_path)
    return latents

def register_conv_injection(model, injection_schedule):
    def conv_forward(self):
        def forward(input_tensor, temb, scale=None):
            hidden_states = input_tensor

            hidden_states = self.norm1(hidden_states)
            hidden_states = self.nonlinearity(hidden_states)

            if self.upsample is not None:
                # upsample_nearest_nhwc fails with large batch sizes. see https://github.com/huggingface/diffusers/issues/984
                if hidden_states.shape[0] >= 64:
                    input_tensor = input_tensor.contiguous()
                    hidden_states = hidden_states.contiguous()
                input_tensor = self.upsample(input_tensor)
                hidden_states = self.upsample(hidden_states)
            elif self.downsample is not None:
                input_tensor = self.downsample(input_tensor)
                hidden_states = self.downsample(hidden_states)

            hidden_states = self.conv1(hidden_states)

            if temb is not None:
                temb = self.time_emb_proj(self.nonlinearity(temb))[:, :, None, None]

            if temb is not None and self.time_embedding_norm == "default":
                hidden_states = hidden_states + temb

            hidden_states = self.norm2(hidden_states)

            if temb is not None and self.time_embedding_norm == "scale_shift":
                scale, shift = torch.chunk(temb, 2, dim=1)
                hidden_states = hidden_states * (1 + scale) + shift

            hidden_states = self.nonlinearity(hidden_states)

            hidden_states = self.dropout(hidden_states)
            hidden_states = self.conv2(hidden_states)
            if self.injection_schedule is not None and (self.t in self.injection_schedule or self.t == 1000):
                source_batch_size = int(hidden_states.shape[0] // 6)
                # inject conditional
                hidden_states[4 * source_batch_size:5 * source_batch_size] = hidden_states[3 * source_batch_size:4 * source_batch_size]
                hidden_states[5 * source_batch_size:6 * source_batch_size] = hidden_states[3 * source_batch_size:4 * source_batch_size]

            if self.conv_shortcut is not None:
                input_tensor = self.conv_shortcut(input_tensor)

            output_tensor = (input_tensor + hidden_states) / self.output_scale_factor

            return output_tensor

        return forward

    conv_module = model.unet.up_blocks[1].resnets[1]
    conv_module.forward = conv_forward(conv_module)
    setattr(conv_module, 'injection_schedule', injection_schedule)

def register_extended_attention_pnp(model, injection_schedule):
    def sa_forward(self):
        to_out = self.to_out
        if type(to_out) is torch.nn.modules.container.ModuleList:
            to_out = self.to_out[0]
        else:
            to_out = self.to_out

        def forward(x, encoder_hidden_states=None, attention_mask=None):
            batch_size, sequence_length, dim = x.shape
            h = self.heads
            n_frames = batch_size // 6
            is_cross = encoder_hidden_states is not None
            encoder_hidden_states = encoder_hidden_states if is_cross else x
            q = self.to_q(x)
            k = self.to_k(encoder_hidden_states)
            v = self.to_v(encoder_hidden_states)

            if self.injection_schedule is not None and (self.t in self.injection_schedule or self.t == 1000):
                # inject unconditional
                q[n_frames:2 * n_frames] = q[:n_frames]
                k[n_frames:2 * n_frames] = k[:n_frames]
                # inject conditional
                q[4 * n_frames:5 * n_frames] = q[3 * n_frames:4 * n_frames]
                k[4 * n_frames:5 * n_frames] = k[3 * n_frames:4 * n_frames]
                q[5 * n_frames:6 * n_frames] = q[3 * n_frames:4 * n_frames]
                k[5 * n_frames:6 * n_frames] = k[3 * n_frames:4 * n_frames]
                pass

            q_source_uncond = q[:n_frames]
            q_source_text = q[3 * n_frames:4 * n_frames]
            q_target_uncond = q[n_frames:2 * n_frames]
            q_target_text = q[4 * n_frames:5 * n_frames]
            q_mutual_uncond = q[2 * n_frames:3 * n_frames]
            q_mutual_text = q[5 * n_frames:6 * n_frames]
            
            k_source_uncond = k[:n_frames]
            k_source_text = k[3 * n_frames:4 * n_frames]
            k_target_uncond = k[n_frames:2 * n_frames].reshape(1, n_frames * sequence_length, -1).repeat(n_frames, 1, 1)
            k_target_text = k[4 * n_frames:5 * n_frames].reshape(1, n_frames * sequence_length, -1).repeat(n_frames, 1, 1)
            k_mutual_uncond = k[2 * n_frames:3 * n_frames].reshape(1, n_frames * sequence_length, -1).repeat(n_frames, 1, 1)
            k_mutual_text = k[5 * n_frames:6 * n_frames].reshape(1, n_frames * sequence_length, -1).repeat(n_frames, 1, 1)
            # k_target_uncond = k[n_frames:2 * n_frames]
            # k_target_text = k[4 * n_frames:5 * n_frames]
            # k_mutual_uncond = k[2 * n_frames:3 * n_frames]
            # k_mutual_text = k[5 * n_frames:6 * n_frames]

            v_source_uncond = v[:n_frames]
            v_source_text = v[3 * n_frames:4 * n_frames]
            v_target_uncond = v[n_frames:2 * n_frames].reshape(1, n_frames * sequence_length, -1).repeat(n_frames, 1, 1)
            v_target_text = v[4 * n_frames:5 * n_frames].reshape(1, n_frames * sequence_length, -1).repeat(n_frames, 1, 1)
            v_mutual_uncond = v[2 * n_frames:3 * n_frames].reshape(1, n_frames * sequence_length, -1).repeat(n_frames, 1, 1)
            v_mutual_text = v[5 * n_frames:6 * n_frames].reshape(1, n_frames * sequence_length, -1).repeat(n_frames, 1, 1)
            # v_target_uncond = v[n_frames:2 * n_frames]
            # v_target_text = v[4 * n_frames:5 * n_frames]
            # v_mutual_uncond = v[2 * n_frames:3 * n_frames]
            # v_mutual_text = v[5 * n_frames:6 * n_frames]

            q_source_uncond = self.head_to_batch_dim(q_source_uncond)
            q_source_text = self.head_to_batch_dim(q_source_text)
            q_target_uncond = self.head_to_batch_dim(q_target_uncond)
            q_target_text = self.head_to_batch_dim(q_target_text)
            q_mutual_uncond = self.head_to_batch_dim(q_mutual_uncond)
            q_mutual_text = self.head_to_batch_dim(q_mutual_text)
            k_source_uncond = self.head_to_batch_dim(k_source_uncond)
            k_source_text = self.head_to_batch_dim(k_source_text)
            k_target_uncond = self.head_to_batch_dim(k_target_uncond)
            k_target_text = self.head_to_batch_dim(k_target_text)
            k_mutual_uncond = self.head_to_batch_dim(k_mutual_uncond)
            k_mutual_text = self.head_to_batch_dim(k_mutual_text)
            v_source_uncond = self.head_to_batch_dim(v_source_uncond)
            v_source_text = self.head_to_batch_dim(v_source_text)
            v_target_uncond = self.head_to_batch_dim(v_target_uncond)
            v_target_text = self.head_to_batch_dim(v_target_text)
            v_mutual_uncond = self.head_to_batch_dim(v_mutual_uncond)
            v_mutual_text = self.head_to_batch_dim(v_mutual_text)


            q_source_uncond = q_source_uncond.view(n_frames, h, sequence_length, dim // h)
            k_source_uncond = k_source_uncond.view(n_frames, h, sequence_length, dim // h)
            v_source_uncond = v_source_uncond.view(n_frames, h, sequence_length, dim // h)
            q_source_text = q_source_text.view(n_frames, h, sequence_length, dim // h)
            k_source_text = k_source_text.view(n_frames, h, sequence_length, dim // h)
            v_source_text = v_source_text.view(n_frames, h, sequence_length, dim // h)
            q_target_uncond = q_target_uncond.view(n_frames, h, sequence_length, dim // h)
            k_target_uncond = k_target_uncond.view(n_frames, h, sequence_length * n_frames, dim // h)
            v_target_uncond = v_target_uncond.view(n_frames, h, sequence_length * n_frames, dim // h)
            # k_target_uncond = k_target_uncond.view(n_frames, h, sequence_length, dim // h)
            # v_target_uncond = v_target_uncond.view(n_frames, h, sequence_length, dim // h)
            q_target_text = q_target_text.view(n_frames, h, sequence_length, dim // h)
            k_target_text = k_target_text.view(n_frames, h, sequence_length * n_frames, dim // h)
            v_target_text = v_target_text.view(n_frames, h, sequence_length * n_frames, dim // h)
            # k_target_text = k_target_text.view(n_frames, h, sequence_length, dim // h)
            # v_target_text = v_target_text.view(n_frames, h, sequence_length, dim // h)
            q_mutual_uncond = q_mutual_uncond.view(n_frames, h, sequence_length, dim // h)
            k_mutual_uncond = k_mutual_uncond.view(n_frames, h, sequence_length * n_frames, dim // h)
            v_mutual_uncond = v_mutual_uncond.view(n_frames, h, sequence_length * n_frames, dim // h)
            # k_mutual_uncond = k_mutual_uncond.view(n_frames, h, sequence_length, dim // h)
            # v_mutual_uncond = v_mutual_uncond.view(n_frames, h, sequence_length, dim // h)
            q_mutual_text = q_mutual_text.view(n_frames, h, sequence_length, dim // h)
            k_mutual_text = k_mutual_text.view(n_frames, h, sequence_length * n_frames, dim // h)
            v_mutual_text = v_mutual_text.view(n_frames, h, sequence_length * n_frames, dim // h)
            # k_mutual_text = k_mutual_text.view(n_frames, h, sequence_length, dim // h)
            # v_mutual_text = v_mutual_text.view(n_frames, h, sequence_length, dim // h)
            
            # if self.t > 300:
            #     q_mutual_uncond = q_target_uncond
            #     q_mutual_text = q_target_text
            #     k_mutual_uncond = k_source_uncond
            #     k_mutual_text = k_source_text
            #     v_mutual_uncond = v_source_uncond
            #     v_mutual_text = v_source_text
            # else:
            #     q_target_uncond = q_source_uncond
            #     q_target_text = q_source_text
            #     k_target_uncond = k_source_uncond
            #     k_target_text = k_source_text
            #     q_mutual_uncond = q_source_uncond
            #     q_mutual_text = q_source_text
            #     k_mutual_uncond = k_source_uncond
            #     k_mutual_text = k_source_text
            #     v_mutual_uncond = v_source_uncond
            #     v_mutual_text = v_source_text
                
            
            # if self.t > -1:
            #     k_target_uncond = k_source_uncond
            #     v_target_uncond = v_source_uncond
            #     k_target_text = k_source_text
            #     v_target_text = v_source_text
            #     k_mutual_uncond = k_source_uncond
            #     v_mutual_uncond = v_source_uncond
            #     k_mutual_text = k_source_text
            #     v_mutual_text = v_source_text

            out_source_uncond_all = []
            out_source_text_all = []
            out_target_uncond_all = []
            out_target_text_all = []
            out_mutual_uncond_all = []
            out_mutual_text_all = []
            
            single_batch = n_frames<=12
            b = n_frames if single_batch else 1

            for frame in range(0, n_frames, b):
                out_source_uncond = []
                out_source_text = []
                out_target_uncond = []
                out_target_text = []
                out_mutual_uncond = []
                out_mutual_text = []
                for j in range(h):
                    sim_source_uncond = torch.bmm(q_source_uncond[frame: frame + b, j], k_source_uncond[frame: frame + b, j].transpose(-1, -2)) * self.scale
                    sim_source_text = torch.bmm(q_source_text[frame: frame + b, j], k_source_text[frame: frame + b, j].transpose(-1, -2)) * self.scale
                    sim_target_uncond = torch.bmm(q_target_uncond[frame: frame + b, j], k_target_uncond[frame: frame + b, j].transpose(-1, -2)) * self.scale
                    sim_target_text = torch.bmm(q_target_text[frame: frame + b, j], k_target_text[frame: frame + b, j].transpose(-1, -2)) * self.scale
                    sim_mutual_uncond = torch.bmm(q_mutual_uncond[frame: frame + b, j], k_mutual_uncond[frame: frame + b, j].transpose(-1, -2)) * self.scale
                    sim_mutual_text = torch.bmm(q_mutual_text[frame: frame + b, j], k_mutual_text[frame: frame + b, j].transpose(-1, -2)) * self.scale

                    out_source_uncond.append(torch.bmm(sim_source_uncond.softmax(dim=-1), v_source_uncond[frame: frame + b, j]))
                    out_source_text.append(torch.bmm(sim_source_text.softmax(dim=-1), v_source_text[frame: frame + b, j]))
                    out_target_uncond.append(torch.bmm(sim_target_uncond.softmax(dim=-1), v_target_uncond[frame: frame + b, j]))
                    out_target_text.append(torch.bmm(sim_target_text.softmax(dim=-1), v_target_text[frame: frame + b, j]))
                    out_mutual_uncond.append(torch.bmm(sim_mutual_uncond.softmax(dim=-1), v_mutual_uncond[frame: frame + b, j]))
                    out_mutual_text.append(torch.bmm(sim_mutual_text.softmax(dim=-1), v_mutual_text[frame: frame + b, j]))

                out_source_uncond = torch.cat(out_source_uncond, dim=0)
                out_source_text = torch.cat(out_source_text, dim=0)
                out_target_uncond = torch.cat(out_target_uncond, dim=0)
                out_target_text = torch.cat(out_target_text, dim=0) 
                out_mutual_uncond = torch.cat(out_mutual_uncond, dim=0)
                out_mutual_text = torch.cat(out_mutual_text, dim=0) 
                if single_batch:
                    out_source_uncond = out_source_uncond.view(h, n_frames,sequence_length, dim // h).permute(1, 0, 2, 3).reshape(h * n_frames, sequence_length, -1)
                    out_source_text = out_source_text.view(h, n_frames,sequence_length, dim // h).permute(1, 0, 2, 3).reshape(h * n_frames, sequence_length, -1)
                    out_target_uncond = out_target_uncond.view(h, n_frames,sequence_length, dim // h).permute(1, 0, 2, 3).reshape(h * n_frames, sequence_length, -1)
                    out_target_text = out_target_text.view(h, n_frames,sequence_length, dim // h).permute(1, 0, 2, 3).reshape(h * n_frames, sequence_length, -1)
                    out_mutual_uncond = out_mutual_uncond.view(h, n_frames,sequence_length, dim // h).permute(1, 0, 2, 3).reshape(h * n_frames, sequence_length, -1)
                    out_mutual_text = out_mutual_text.view(h, n_frames,sequence_length, dim // h).permute(1, 0, 2, 3).reshape(h * n_frames, sequence_length, -1)
                out_source_uncond_all.append(out_source_uncond)
                out_source_text_all.append(out_source_text)
                out_target_uncond_all.append(out_target_uncond)
                out_target_text_all.append(out_target_text)
                out_mutual_uncond_all.append(out_mutual_uncond)
                out_mutual_text_all.append(out_mutual_text)
            
            out_source_uncond = torch.cat(out_source_uncond_all, dim=0)
            out_source_text = torch.cat(out_source_text_all, dim=0)
            out_target_uncond = torch.cat(out_target_uncond_all, dim=0)
            out_target_text = torch.cat(out_target_text_all, dim=0)
            out_mutual_uncond = torch.cat(out_mutual_uncond_all, dim=0)
            out_mutual_text = torch.cat(out_mutual_text_all, dim=0)
                
            out = torch.cat([out_source_uncond, out_target_uncond, out_mutual_uncond,
                             out_source_text, out_target_text, out_mutual_text,], dim=0)
            out = self.batch_to_head_dim(out)
            out = to_out(out)

            return out

        return forward

    for _, module in model.unet.named_modules():
        if isinstance_str(module, "BasicTransformerBlock"):
            module.attn1.forward = sa_forward(module.attn1)
            setattr(module.attn1, 'injection_schedule', [])

    res_dict = {1: [1, 2], 2: [0, 1, 2], 3: [0, 1, 2]}
    # we are injecting attention in blocks 4 - 11 of the decoder, so not in the first block of the lowest resolution
    for res in res_dict:
        for block in res_dict[res]:
            module = model.unet.up_blocks[res].attentions[block].transformer_blocks[0].attn1
            module.forward = sa_forward(module)
            setattr(module, 'injection_schedule', injection_schedule)

def register_attention_control(model, controller):
    class AttnProcessor():
        def __init__(self,place_in_unet):
            self.place_in_unet = place_in_unet
            self.is_pivotal = True

        def __call__(self,
            attn,
            hidden_states,
            encoder_hidden_states=None,
            attention_mask=None,
            temb=None,
            scale=1.0,):
            # The `Attention` class can call different attention processors / attention functions
    
            residual = hidden_states

            if attn.spatial_norm is not None:
                hidden_states = attn.spatial_norm(hidden_states, temb)

            input_ndim = hidden_states.ndim

            if input_ndim == 4:
                batch_size, channel, height, width = hidden_states.shape
                hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

            h = attn.heads
            is_cross = encoder_hidden_states is not None
            if encoder_hidden_states is None:
                encoder_hidden_states = hidden_states
            elif attn.norm_cross:
                encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

            batch_size, sequence_length, _ = (
                hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
            )
            attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)

            q = attn.to_q(hidden_states)
            k = attn.to_k(encoder_hidden_states)
            v = attn.to_v(encoder_hidden_states)
            q = attn.head_to_batch_dim(q)
            k = attn.head_to_batch_dim(k)
            v = attn.head_to_batch_dim(v)

            if not is_cross:
                q_list, k_list, v_list = controller.self_attn_forward(q, k, v, attn.heads)
                
                flag_list = [False, True, True, False, True, True]
                attention_probs_list = []
                hidden_states_list = []
                for q, k, v, flag in zip(q_list, k_list, v_list, flag_list):
                    if flag:
                        k = attn.batch_to_head_dim(k)
                        v = attn.batch_to_head_dim(v)
                        n_frames = k.shape[0]
                        k = k.reshape(1, n_frames * sequence_length, -1).repeat(n_frames, 1, 1)
                        v = v.reshape(1, n_frames * sequence_length, -1).repeat(n_frames, 1, 1)
                        k = attn.head_to_batch_dim(k)
                        v = attn.head_to_batch_dim(v)
                    attention_probs = attn.get_attention_scores(q, k, attention_mask)
                    hidden_states = torch.bmm(attention_probs, v)
                    hidden_states = attn.batch_to_head_dim(hidden_states)
                    hidden_states_list.append(hidden_states)
                hidden_states = torch.cat(hidden_states_list, dim=0)
            else:
                attention_probs = attn.get_attention_scores(q, k, attention_mask)
                attention_probs  = controller(attention_probs , is_cross, self.place_in_unet, attn.heads, self.is_pivotal)
                hidden_states = torch.bmm(attention_probs, v)
                hidden_states = attn.batch_to_head_dim(hidden_states)

            # linear proj   
            hidden_states = attn.to_out[0](hidden_states, scale=scale)
            # dropout
            hidden_states = attn.to_out[1](hidden_states)

            if input_ndim == 4:
                hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

            if attn.residual_connection:
                hidden_states = hidden_states + residual

            hidden_states = hidden_states / attn.rescale_output_factor

            return hidden_states


    def register_recr(net_, count, place_in_unet):
        for idx, m in enumerate(net_.modules()):
            # print(m.__class__.__name__)
            if m.__class__.__name__ == "Attention":
                count+=1
                m.processor = AttnProcessor( place_in_unet)
        return count

    cross_att_count = 0
    sub_nets = model.named_children()
    for net in sub_nets:
        if "down" in net[0]:
            cross_att_count += register_recr(net[1], 0, "down")
        elif "up" in net[0]:
            cross_att_count += register_recr(net[1], 0, "up")
        elif "mid" in net[0]:
            cross_att_count += register_recr(net[1], 0, "mid")
    controller.num_att_layers = cross_att_count

def make_tokenflow_attention_block(block_class: Type[torch.nn.Module]) -> Type[torch.nn.Module]:

    class TokenFlowBlock(block_class):

        def forward(
            self,
            hidden_states,
            attention_mask=None,
            encoder_hidden_states=None,
            encoder_attention_mask=None,
            timestep=None,
            cross_attention_kwargs=None,
            class_labels=None,
        ) -> torch.Tensor:
            
            batch_size, sequence_length, dim = hidden_states.shape
            n_frames = batch_size // 6
            mid_idx = n_frames // 2
            hidden_states = hidden_states.view(6, n_frames, sequence_length, dim)

            if self.use_ada_layer_norm:
                norm_hidden_states = self.norm1(hidden_states, timestep)
            elif self.use_ada_layer_norm_zero:
                norm_hidden_states, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.norm1(
                    hidden_states, timestep, class_labels, hidden_dtype=hidden_states.dtype
                )
            else:
                norm_hidden_states = self.norm1(hidden_states)

            norm_hidden_states = norm_hidden_states.view(6, n_frames, sequence_length, dim)
            if self.pivotal_pass:
                self.pivot_hidden_states = norm_hidden_states
            else:
                idx1 = []
                idx2 = [] 
                batch_idxs = [self.batch_idx]
                if self.batch_idx > 0:
                    batch_idxs.append(self.batch_idx - 1)
                
                sim = batch_cosine_sim(norm_hidden_states[3].reshape(-1, dim),
                                        self.pivot_hidden_states[3][batch_idxs].reshape(-1, dim))
                if len(batch_idxs) == 2:
                    sim1, sim2 = sim.chunk(2, dim=1)
                    # sim: n_frames * seq_len, len(batch_idxs) * seq_len
                    idx1.append(sim1.argmax(dim=-1))  # n_frames * seq_len
                    idx2.append(sim2.argmax(dim=-1))  # n_frames * seq_len
                else:
                    idx1.append(sim.argmax(dim=-1))
                idx1 = torch.stack(idx1 * 6, dim=0) # 3, n_frames * seq_len
                idx1 = idx1.squeeze(1)
                if len(batch_idxs) == 2:
                    idx2 = torch.stack(idx2 * 6, dim=0) # 3, n_frames * seq_len
                    idx2 = idx2.squeeze(1)

            # 1. Self-Attention
            cross_attention_kwargs = cross_attention_kwargs if cross_attention_kwargs is not None else {}
            if self.pivotal_pass:
                # norm_hidden_states.shape = 3, n_frames * seq_len, dim
                self.attn_output = self.attn1(
                        norm_hidden_states.view(batch_size, sequence_length, dim),
                        encoder_hidden_states=encoder_hidden_states if self.only_cross_attention else None,
                        **cross_attention_kwargs,
                    )
                # 3, n_frames * seq_len, dim - > 3 * n_frames, seq_len, dim
                self.kf_attn_output = self.attn_output 
            else:
                batch_kf_size, _, _ = self.kf_attn_output.shape
                self.attn_output = self.kf_attn_output.view(6, batch_kf_size // 6, sequence_length, dim)[:,
                                   batch_idxs]  # 3, n_frames, seq_len, dim --> 3, len(batch_idxs), seq_len, dim
            if self.use_ada_layer_norm_zero:
                self.attn_output = gate_msa.unsqueeze(1) * self.attn_output

            # gather values from attn_output, using idx as indices, and get a tensor of shape 3, n_frames, seq_len, dim
            if not self.pivotal_pass:
                if len(batch_idxs) == 2:
                    attn_1, attn_2 = self.attn_output[:, 0], self.attn_output[:, 1]
                    attn_output1 = attn_1.gather(dim=1, index=idx1.unsqueeze(-1).repeat(1, 1, dim))
                    attn_output2 = attn_2.gather(dim=1, index=idx2.unsqueeze(-1).repeat(1, 1, dim))

                    s = torch.arange(0, n_frames).to(idx1.device) + batch_idxs[0] * n_frames
                    # distance from the pivot
                    p1 = batch_idxs[0] * n_frames + n_frames // 2
                    p2 = batch_idxs[1] * n_frames + n_frames // 2
                    d1 = torch.abs(s - p1)
                    d2 = torch.abs(s - p2)
                    # weight
                    w1 = d2 / (d1 + d2)
                    w1 = torch.sigmoid(w1)
                    
                    w1 = w1.unsqueeze(0).unsqueeze(-1).unsqueeze(-1).repeat(6, 1, sequence_length, dim)
                    attn_output1 = attn_output1.view(6, n_frames, sequence_length, dim)
                    attn_output2 = attn_output2.view(6, n_frames, sequence_length, dim)
                    attn_output = w1 * attn_output1 + (1 - w1) * attn_output2
                else:
                    attn_output = self.attn_output[:,0].gather(dim=1, index=idx1.unsqueeze(-1).repeat(1, 1, dim))

                attn_output = attn_output.reshape(
                        batch_size, sequence_length, dim)  # 3 * n_frames, seq_len, dim
            else:
                attn_output = self.attn_output
            hidden_states = hidden_states.reshape(batch_size, sequence_length, dim)  # 3 * n_frames, seq_len, dim
            hidden_states = attn_output + hidden_states

            if self.attn2 is not None:
                norm_hidden_states = (
                    self.norm2(hidden_states, timestep) if self.use_ada_layer_norm else self.norm2(hidden_states)
                )

                # 2. Cross-Attention
                attn_output = self.attn2(
                    norm_hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                    attention_mask=encoder_attention_mask,
                    **cross_attention_kwargs,
                )
                hidden_states = attn_output + hidden_states

            # 3. Feed-forward
            norm_hidden_states = self.norm3(hidden_states)

            if self.use_ada_layer_norm_zero:
                norm_hidden_states = norm_hidden_states * (1 + scale_mlp[:, None]) + shift_mlp[:, None]


            ff_output = self.ff(norm_hidden_states)

            if self.use_ada_layer_norm_zero:
                ff_output = gate_mlp.unsqueeze(1) * ff_output

            hidden_states = ff_output + hidden_states

            return hidden_states

    return TokenFlowBlock


def set_tokenflow(
        model: torch.nn.Module):
    """
    Sets the tokenflow attention blocks in a model.
    """

    for _, module in model.named_modules():
        if isinstance_str(module, "BasicTransformerBlock"):
            make_tokenflow_block_fn = make_tokenflow_attention_block 
            module.__class__ = make_tokenflow_block_fn(module.__class__)

            # Something needed for older versions of diffusers
            if not hasattr(module, "use_ada_layer_norm_zero"):
                module.use_ada_layer_norm = False
                module.use_ada_layer_norm_zero = False

    return model
