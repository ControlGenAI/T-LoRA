#!/usr/bin/env python
"""
Multi-concept generation with T-LoRA adapters using LoRAShop.

Combines T-LoRA's timestep-dependent rank masking with LoRAShop's
block-blended denoising for multi-concept image generation on FLUX.1-dev.

Each --adapter_path should be a T-LoRA adapter directory produced by
the training script (containing adapter_model.safetensors,
adapter_config.json, and tlora_meta.json).

Usage:
    python inference_tlorashop.py \
        --adapter_paths /path/to/adapter_A /path/to/adapter_B \
        --prompt "a photo of concept_A and concept_B together" \
        --target_prompts "concept_A" "concept_B" \
        --seed 42 --output_path result.png

Environment:
    Set LORASHOP_DIR to the LoRAShop repo root if it is not located at
    ../LoRAShop relative to this script.
"""

import argparse
import os
import sys

import torch

LORASHOP_DIR = os.environ.get(
    "LORASHOP_DIR",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "LoRAShop"),
)
if os.path.isdir(LORASHOP_DIR):
    sys.path.insert(0, LORASHOP_DIR)
else:
    raise RuntimeError(
        f"LoRAShop directory not found at {LORASHOP_DIR}. "
        "Set LORASHOP_DIR env var to the LoRAShop repo root."
    )

import flux_blocks  # noqa: E402

from tlora_peft.lorashop_diffusers_compat import (  # noqa: E402
    apply_lorashop_single_block_diffusers_forward_patch,
)

apply_lorashop_single_block_diffusers_forward_patch(flux_blocks)

from flux_real_pipeline import RealGenerationPipeline  # noqa: E402

from tlora_peft import (  # noqa: E402
    load_tlora_weights,
    load_adapter,
    enable_timestep_hook,
    cast_tlora_adapter_dtype,
)


def parse_args():
    p = argparse.ArgumentParser(
        description="Multi-concept generation with T-LoRA + LoRAShop",
    )
    p.add_argument(
        "--adapter_paths", type=str, nargs="+", required=True,
        help="Paths to T-LoRA adapter directories (one per concept)",
    )
    p.add_argument("--prompt", type=str, required=True)
    p.add_argument(
        "--target_prompts", type=str, nargs="+", required=True,
        help="Per-concept substrings of the prompt (one per adapter, same order)",
    )
    p.add_argument(
        "--pretrained_model_name_or_path", type=str, default="black-forest-labs/FLUX.1-dev",
        help="HuggingFace model ID or local path for the base FLUX model",
    )
    p.add_argument("--hf_token", type=str, default=None)
    p.add_argument("--output_path", type=str, default="tlorashop_output.png")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num_inference_steps", type=int, default=50)
    p.add_argument("--guidance_scale", type=float, default=3.5)
    p.add_argument("--width", type=int, default=1024)
    p.add_argument("--height", type=int, default=1024)
    p.add_argument("--max_sequence_length", type=int, default=256)
    p.add_argument("--edit_start_step", type=int, default=0)
    return p.parse_args()


def main():
    args = parse_args()

    if len(args.adapter_paths) != len(args.target_prompts):
        raise ValueError(
            f"Got {len(args.adapter_paths)} adapter(s) but "
            f"{len(args.target_prompts)} target prompt(s) — counts must match."
        )

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    pipe = RealGenerationPipeline.from_pretrained(
        args.pretrained_model_name_or_path,
        torch_dtype=torch.bfloat16,
        token=args.hf_token,
    ).to("cuda")

    peft_model = load_tlora_weights(
        pipe.transformer,
        args.adapter_paths[0],
        adapter_name="default_0",
    )

    for i, adapter_path in enumerate(args.adapter_paths[1:], 1):
        load_adapter(peft_model, adapter_path, adapter_name=f"default_{i}")
    pipe.transformer = peft_model.base_model.model

    cast_tlora_adapter_dtype(pipe.transformer, torch.float32)
    hook = enable_timestep_hook(pipe.transformer, timestep_scale=1000.0)
    pipe.register_transformer_blocks()

    try:
        generator = torch.Generator(device="cuda").manual_seed(args.seed)

        result = pipe(
            prompt=args.prompt,
            target_prompt=args.target_prompts,
            guidance_scale=args.guidance_scale,
            num_inference_steps=args.num_inference_steps,
            max_sequence_length=args.max_sequence_length,
            height=args.height,
            width=args.width,
            generator=generator,
            edit_start_step=args.edit_start_step,
        )
    finally:
        hook.remove()

    os.makedirs(os.path.dirname(args.output_path) or ".", exist_ok=True)
    result.images[0].save(args.output_path)
    print(f"Saved to {args.output_path}")


if __name__ == "__main__":
    main()
