#!/usr/bin/env python
"""Generate an image with a T-LoRA adapter on FLUX.1-dev."""

import argparse

import torch
from diffusers import FluxPipeline

from tlora_peft import load_tlora_weights, enable_timestep_hook, cast_tlora_adapter_dtype


def parse_args():
    p = argparse.ArgumentParser(description="T-LoRA inference on FLUX.1-dev")
    p.add_argument("--prompt", type=str, required=True)
    p.add_argument("--adapter_path", type=str, required=True,
                   help="Directory saved by save_tlora_adapter (or a checkpoint-N folder)")
    p.add_argument("--pretrained_model_name_or_path", type=str,
                   default="black-forest-labs/FLUX.1-dev")
    p.add_argument("--output_path", type=str, default="output.png")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--num_inference_steps", type=int, default=28)
    p.add_argument("--guidance_scale", type=float, default=3.5)
    p.add_argument("--width", type=int, default=1024)
    p.add_argument("--height", type=int, default=1024)
    return p.parse_args()


def main():
    args = parse_args()

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    pipe = FluxPipeline.from_pretrained(
        args.pretrained_model_name_or_path, torch_dtype=torch.bfloat16,
    ).to("cuda")

    pipe.transformer = load_tlora_weights(pipe.transformer, args.adapter_path)

    cast_tlora_adapter_dtype(pipe.transformer, torch.float32)
    hook = enable_timestep_hook(pipe.transformer, timestep_scale=1000.0)

    try:
        generator = torch.Generator(device="cuda").manual_seed(args.seed)
        image = pipe(
            prompt=args.prompt,
            num_inference_steps=args.num_inference_steps,
            guidance_scale=args.guidance_scale,
            width=args.width,
            height=args.height,
            generator=generator,
        ).images[0]
    finally:
        hook.remove()

    image.save(args.output_path)
    print(f"Saved to {args.output_path}")


if __name__ == "__main__":
    main()
