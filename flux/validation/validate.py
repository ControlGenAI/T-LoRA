import os
import torch
from contextlib import nullcontext


@torch.no_grad()
def log_validation(
    pipeline,
    args,
    accelerator,
    pipeline_args,
    epoch,
    torch_dtype,
    is_final_validation=False,
):

    pipeline = pipeline.to(accelerator.device, dtype=torch_dtype)
    pipeline.transformer.to(accelerator.device, dtype=torch_dtype)
    pipeline.set_progress_bar_config(disable=True)

    # run inference
    generator = (
        torch.Generator(device=accelerator.device).manual_seed(args.seed)
        if args.seed
        else None
    )
    # autocast_ctx = torch.autocast(accelerator.device.type) if not is_final_validation else nullcontext()
    autocast_ctx = nullcontext()
    samples_path = os.path.join(
        args.output_dir,
        f"checkpoint-{epoch}",
        "samples",
        "validation",
    )
    all_images, all_captions = [], []
    for prompt in args.validation_prompt.split('#'):
        prompt = prompt.format(f"{args.placeholder_token} {args.class_name}")
        with autocast_ctx:
            latents = [
                pipeline(prompt, generator=generator, output_type="latent").images[
                    0
                ]
                for _ in range(args.num_validation_images)
            ]
            for l in latents:
                l = pipeline._unpack_latents(
                    l[None],
                    pipeline.default_sample_size * pipeline.vae_scale_factor,
                    pipeline.default_sample_size * pipeline.vae_scale_factor,
                    pipeline.vae_scale_factor,
                )
                l = (
                    l / pipeline.vae.config.scaling_factor
                ) + pipeline.vae.config.shift_factor
                image = pipeline.vae.decode(l.to(torch_dtype), return_dict=False)[0]
                image = pipeline.image_processor.postprocess(image, output_type="pil")
                all_images += image
                all_captions.append(prompt)

        os.makedirs(os.path.join(samples_path), exist_ok=True)
        for idx, image in enumerate(all_images):
            prompt = all_captions[idx]
            image.save(os.path.join(samples_path, f"{idx}_{prompt}.png"))
