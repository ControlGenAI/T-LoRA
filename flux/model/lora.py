import copy
import gc
import math
import tqdm.autonotebook as tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F

from diffusers.models.attention_processor import FluxAttnProcessor2_0


class LoRALinearLayer(nn.Module):
    def __init__(self, in_features, out_features, rank=4, do_training=True, sig_type=None, original_layer=None):
        super().__init__()

        if rank > min(in_features, out_features):
            raise ValueError(
                f"LoRA rank {rank} must be less or equal than {min(in_features, out_features)}"
            )

        self.rank = rank

        self.down = nn.Linear(in_features, rank, bias=False)
        self.up = nn.Linear(rank, out_features, bias=False)

        nn.init.normal_(self.down.weight, std=1 / rank)
        nn.init.zeros_(self.up.weight)

    def forward(self, hidden_states, mask=None):
        if mask is None:
            mask = torch.ones((1, self.rank))
        orig_dtype = hidden_states.dtype
        dtype = self.down.weight.dtype

        down_hidden_states = self.down(hidden_states.to(dtype)) * mask.to(hidden_states.device)
        up_hidden_states = self.up(down_hidden_states)

        return up_hidden_states.to(orig_dtype)


class FluxLoraAttnProcessor(nn.Module):
    """Attention processor used typically in processing the SD3-like self-attention projections."""

    def __init__(
        self,
        hidden_size,
        lora_linear_layer=LoRALinearLayer,
        cross_attention_dim=None,
        rank=4,
        do_training=True,
        sig_type='last',
        original_layer=None,
    ):
        super().__init__()
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError(
                "FluxAttnProcessor2_0 requires PyTorch 2.0, to use it, please upgrade PyTorch to 2.0."
            )
        self.do_training = do_training
        if original_layer:
            self.to_q_lora = lora_linear_layer(hidden_size, hidden_size, rank, self.do_training, sig_type=sig_type, original_layer=original_layer.to_q)
            self.to_k_lora = lora_linear_layer(hidden_size, hidden_size, rank, self.do_training, sig_type=sig_type, original_layer=original_layer.to_k)
            self.to_v_lora = lora_linear_layer(hidden_size, hidden_size, rank, self.do_training, sig_type=sig_type, original_layer=original_layer.to_v)
            if original_layer.to_out:
                self.to_out_lora = lora_linear_layer(hidden_size, hidden_size, rank, self.do_training, sig_type=sig_type, original_layer=original_layer.to_out[0])

            if original_layer.add_q_proj:
                self.to_q_proj_lora = lora_linear_layer(hidden_size, hidden_size, rank, self.do_training, sig_type=sig_type, original_layer=original_layer.add_q_proj)
                self.to_k_proj_lora = lora_linear_layer(hidden_size, hidden_size, rank, self.do_training, sig_type=sig_type, original_layer=original_layer.add_k_proj)
                self.to_v_proj_lora = lora_linear_layer(hidden_size, hidden_size, rank, self.do_training, sig_type=sig_type, original_layer=original_layer.add_v_proj)
        else:
            self.to_q_lora = lora_linear_layer(hidden_size, hidden_size, rank, self.do_training, sig_type=sig_type, original_layer=None)
            self.to_k_lora = lora_linear_layer(hidden_size, hidden_size, rank, self.do_training, sig_type=sig_type, original_layer=None)
            self.to_v_lora = lora_linear_layer(hidden_size, hidden_size, rank, self.do_training, sig_type=sig_type, original_layer=None)
            self.to_out_lora = lora_linear_layer(hidden_size, hidden_size, rank, self.do_training, sig_type=sig_type, original_layer=None)

            self.to_q_proj_lora = lora_linear_layer(hidden_size, hidden_size, rank, self.do_training, sig_type=sig_type, original_layer=None)
            self.to_k_proj_lora = lora_linear_layer(hidden_size, hidden_size, rank, self.do_training, sig_type=sig_type, original_layer=None)
            self.to_v_proj_lora = lora_linear_layer(hidden_size, hidden_size, rank, self.do_training, sig_type=sig_type, original_layer=None)

    def __call__(
        self,
        attn,
        hidden_states,
        encoder_hidden_states=None,
        attention_mask=None,
        image_rotary_emb=None,
        scale=1.0,
        sigma_mask=None,
    ) -> torch.FloatTensor:
        batch_size, _, _ = (
            hidden_states.shape
            if encoder_hidden_states is None
            else encoder_hidden_states.shape
        )

        query = attn.to_q(hidden_states) + scale * self.to_q_lora(hidden_states, sigma_mask)
        key = attn.to_k(hidden_states) + scale * self.to_k_lora(hidden_states, sigma_mask)
        value = attn.to_v(hidden_states) + scale * self.to_v_lora(hidden_states, sigma_mask)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads

        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        if attn.norm_q is not None:
            query = attn.norm_q(query)
        if attn.norm_k is not None:
            key = attn.norm_k(key)

        # the attention in FluxSingleTransformerBlock does not use `encoder_hidden_states`
        if encoder_hidden_states is not None:
            # `context` projections.
            encoder_hidden_states_query_proj = attn.add_q_proj(encoder_hidden_states) + scale * self.to_q_proj_lora(encoder_hidden_states, sigma_mask)
            encoder_hidden_states_key_proj = attn.add_k_proj(encoder_hidden_states) + scale * self.to_k_proj_lora(encoder_hidden_states, sigma_mask)
            encoder_hidden_states_value_proj = attn.add_v_proj(encoder_hidden_states) + scale * self.to_v_proj_lora(encoder_hidden_states, sigma_mask)

            encoder_hidden_states_query_proj = encoder_hidden_states_query_proj.view(
                batch_size, -1, attn.heads, head_dim
            ).transpose(1, 2)
            encoder_hidden_states_key_proj = encoder_hidden_states_key_proj.view(
                batch_size, -1, attn.heads, head_dim
            ).transpose(1, 2)
            encoder_hidden_states_value_proj = encoder_hidden_states_value_proj.view(
                batch_size, -1, attn.heads, head_dim
            ).transpose(1, 2)

            if attn.norm_added_q is not None:
                encoder_hidden_states_query_proj = attn.norm_added_q(
                    encoder_hidden_states_query_proj
                )
            if attn.norm_added_k is not None:
                encoder_hidden_states_key_proj = attn.norm_added_k(
                    encoder_hidden_states_key_proj
                )

            # attention
            query = torch.cat([encoder_hidden_states_query_proj, query], dim=2)
            key = torch.cat([encoder_hidden_states_key_proj, key], dim=2)
            value = torch.cat([encoder_hidden_states_value_proj, value], dim=2)

        if image_rotary_emb is not None:
            from diffusers.models.embeddings import apply_rotary_emb

            query = apply_rotary_emb(query, image_rotary_emb)
            key = apply_rotary_emb(key, image_rotary_emb)

        hidden_states = F.scaled_dot_product_attention(
            query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
        )

        hidden_states = hidden_states.transpose(1, 2).reshape(
            batch_size, -1, attn.heads * head_dim
        )
        hidden_states = hidden_states.to(query.dtype)

        if encoder_hidden_states is not None:
            encoder_hidden_states, hidden_states = (
                hidden_states[:, : encoder_hidden_states.shape[1]],
                hidden_states[:, encoder_hidden_states.shape[1] :],
            )

            # linear proj
            hidden_states = attn.to_out[0](hidden_states) + scale * self.to_out_lora(hidden_states, sigma_mask)
            # dropout
            hidden_states = attn.to_out[1](hidden_states)

            encoder_hidden_states = attn.to_add_out(encoder_hidden_states)

            return hidden_states, encoder_hidden_states
        else:
            return hidden_states


def default_set_attn_proc_func(
    name: str,
    hidden_size: int,
    cross_attention_dim,
    ori_attn_proc,
):
    return ori_attn_proc


def set_flux_transformer_attn_processor(
    transformer,
    set_attn_proc_func=default_set_attn_proc_func,
    set_attn_module_names=None,
):
    do_set_processor = lambda name, module_names: (
        any([name.startswith(module_name) for module_name in module_names])
        if module_names is not None
        else True
    )  # prefix match

    attn_procs = {}
    for name, attn_processor in tqdm.tqdm(transformer.attn_processors.items()):
        dim_head = transformer.config.attention_head_dim
        num_heads = transformer.config.num_attention_heads
        if name.endswith("attn.processor"):
            attn_procs[name] = (
                set_attn_proc_func(name, dim_head, num_heads, attn_processor)
                if do_set_processor(name, set_attn_module_names)
                else attn_processor
            )

    transformer.set_attn_processor(attn_procs)
