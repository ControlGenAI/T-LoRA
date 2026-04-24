#!/usr/bin/env python
"""Generate an image with a T-LoRA adapter on Stable Diffusion XL."""

import argparse

import torch
from diffusers import AutoencoderKL, StableDiffusionXLPipeline

from tlora_peft import load_tlora_weights, enable_timestep_hook


def parse_args():
    p = argparse.ArgumentParser(description="T-LoRA inference on SD-XL")
    p.add_argument("--prompt", type=str, required=True)
    p.add_argument("--adapter_path", type=str, required=True,
                   help="Directory saved by save_tlora_adapter")
    p.add_argument("--pretrained_model_name_or_path", type=str,
                   default="stabilityai/stable-diffusion-xl-base-1.0")
    p.add_argument(
        "--pretrained_vae_model_name_or_path", type=str,
        default="madebyollin/sdxl-vae-fp16-fix",
        help="VAE checkpoint to use (must match the one used at training time).",
    )
    p.add_argument("--output_path", type=str, default="output.png")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--num_inference_steps", type=int, default=25)
    p.add_argument("--guidance_scale", type=float, default=5.0)
    p.add_argument("--width", type=int, default=1024)
    p.add_argument("--height", type=int, default=1024)
    return p.parse_args()


def main():
    args = parse_args()

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    vae = AutoencoderKL.from_pretrained(
        args.pretrained_vae_model_name_or_path,
    )

    pipe = StableDiffusionXLPipeline.from_pretrained(
        args.pretrained_model_name_or_path,
        vae=vae,
        torch_dtype=torch.float32,
    ).to("cuda")

    pipe.unet = load_tlora_weights(pipe.unet, args.adapter_path)
    hook = enable_timestep_hook(pipe.unet, timestep_scale=1.0)

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
