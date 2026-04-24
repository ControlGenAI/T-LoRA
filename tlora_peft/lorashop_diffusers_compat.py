"""
LoRAShop + modern diffusers FLUX compatibility (no edits to the LoRAShop tree).

Upstream LoRAShop's ``SingleTransformerBlock.forward`` matches an older diffusers
API (single concatenated ``hidden_states``).  Diffusers >= ~0.37 calls
``FluxSingleTransformerBlock`` with separate ``encoder_hidden_states`` /
``hidden_states`` keyword arguments and expects a ``(text, image)`` tuple back.

We monkey-patch ``flux_blocks.SingleTransformerBlock.forward`` once, in-process,
so ``inference_tlorashop.py`` can stay the only integration entrypoint touched.
"""

from __future__ import annotations

import functools
from typing import Any

import torch


def apply_lorashop_single_block_diffusers_forward_patch(flux_blocks: Any) -> None:
    """In-place patch on the imported ``flux_blocks`` module from LoRAShop."""
    cls = flux_blocks.SingleTransformerBlock
    if getattr(cls, "_tlora_peft_diffusers_single_block_patch", False):
        return

    _orig_forward = cls.forward

    @functools.wraps(_orig_forward)
    def forward(
        self,
        hidden_states,
        encoder_hidden_states=None,
        temb=None,
        image_rotary_emb=None,
        joint_attention_kwargs=None,
    ):
        text_seq_len = None
        if encoder_hidden_states is not None:
            text_seq_len = encoder_hidden_states.shape[1]
            hidden_states = torch.cat([encoder_hidden_states, hidden_states], dim=1)
        elif joint_attention_kwargs is not None and "prompt_length" in joint_attention_kwargs:
            text_seq_len = int(joint_attention_kwargs["prompt_length"])

        out = _orig_forward(self, hidden_states, temb, image_rotary_emb, joint_attention_kwargs)

        if text_seq_len is not None:
            return out[:, :text_seq_len], out[:, text_seq_len:]
        return out

    cls.forward = forward
    cls._tlora_peft_diffusers_single_block_patch = True
