import logging
import math
import os
import warnings
from pathlib import Path
import yaml
import gc
import copy

import torch
import torch.utils.checkpoint
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import (
    DistributedDataParallelKwargs,
    ProjectConfiguration,
    set_seed,
)
from peft import LoraConfig
from peft.utils import get_peft_model_state_dict
from tqdm.auto import tqdm

import diffusers
from diffusers import FluxPipeline
from diffusers.loaders import AttnProcsLayers
from diffusers.optimization import get_scheduler
from diffusers.training_utils import (
    cast_training_params,
    compute_density_for_timestep_sampling,
    compute_loss_weighting_for_sd3,
    free_memory,
)

from data.dataset import DreamBoothDataset, collate_fn
from optimizer.optimizer import setup_optimizer
from model.base_model import setup_base_model
from model.encode import encode_prompt, compute_text_embeddings
from model.utils import unwrap_model, load_text_encoders, get_layer_by_name
from model.lora import (
    FluxLoraAttnProcessor,
    LoRALinearLayer,
    set_flux_transformer_attn_processor,
)
from model.pipeline import TLoRAFluxPipeline
from validation.validate import log_validation

diffusers.utils.logging.set_verbosity_error()
diffusers.utils.logging.disable_progress_bar()
warnings.filterwarnings("ignore")


logger = get_logger(__name__, log_level="ERROR")


def run_train(args):
    accelerator_project_config = ProjectConfiguration(
        project_dir=args.output_dir, logging_dir=Path(args.output_dir, args.logging_dir)
    )
    kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config,
        kwargs_handlers=[kwargs],
    )

    if torch.backends.mps.is_available():
        accelerator.native_amp = False

    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    if torch.backends.mps.is_available() and weight_dtype == torch.bfloat16:
        raise ValueError(
            "Mixed precision training with bfloat16 is not supported on MPS. Please use fp16 (recommended) or fp32 instead."
        )

    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(os.path.join(args.output_dir, args.logging_dir), exist_ok=True)

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    instance_prompt = f'a photo of {args.placeholder_token} {args.class_name}'

    # Dataset and DataLoaders creation:
    train_dataset = DreamBoothDataset(
        instance_prompt=instance_prompt,
        instance_data_root=args.instance_data_dir,
        resolution=args.resolution,
        repeats=args.repeats,
        one_image=args.one_image,
        mask_dir=args.mask_dir
    )
    args.instance_prompt = train_dataset.instance_prompt

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.train_batch_size,
        shuffle=True,
        collate_fn=lambda examples: collate_fn(examples),
        num_workers=args.dataloader_num_workers,
    )

    # Load the model
    tokenizer_one, tokenizer_two, text_encoder_cls_one, text_encoder_cls_two, text_encoder_one, text_encoder_two, vae, transformer, noise_scheduler, noise_scheduler_copy = setup_base_model(args, accelerator, weight_dtype)

    set_flux_transformer_attn_processor(
        transformer,
        set_attn_proc_func=lambda name, dh, nh, ap: FluxLoraAttnProcessor(
            hidden_size=transformer.inner_dim, rank=args.rank, 
            lora_linear_layer=LoRALinearLayer,
            do_training=True, 
        ),
    )
    lora_layers = AttnProcsLayers(transformer.attn_processors)

    if args.allow_tf32 and torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True

    # Make sure the trainable params are in float32.
    if args.mixed_precision == "fp16":
        models = [transformer]
        cast_training_params(models, dtype=torch.float32)

    transformer_lora_parameters = list(
        filter(lambda p: p.requires_grad, transformer.parameters())
    )

    # Optimization parameters
    transformer_parameters_with_lr = {
        "params": transformer_lora_parameters,
        "lr": args.learning_rate,
    }
    params_to_optimize = [transformer_parameters_with_lr]

    # Optimizer creation
    optimizer, params_to_optimize = setup_optimizer(args, logger, params_to_optimize)

    tokenizers = [tokenizer_one, tokenizer_two]
    text_encoders = [text_encoder_one, text_encoder_two]

    # If no type of tuning is done on the text_encoder and custom instance prompts are NOT
    # provided (i.e. the --instance_prompt is used for all images), we encode the instance prompt once to avoid
    # the redundant encoding.
    (
        instance_prompt_hidden_states,
        instance_pooled_prompt_embeds,
        instance_text_ids,
    ) = compute_text_embeddings(args.instance_prompt, text_encoders, tokenizers, accelerator, args)

    # Clear the memory here
    del text_encoder_one, text_encoder_two, tokenizer_one, tokenizer_two
    free_memory()

    # If custom instance prompts are NOT provided (i.e. the instance prompt is used for all images),
    # pack the statically computed variables appropriately here. This is so that we don't
    # have to pass them to the dataloader.

    prompt_embeds = instance_prompt_hidden_states
    pooled_prompt_embeds = instance_pooled_prompt_embeds
    text_ids = instance_text_ids

    vae_config_shift_factor = vae.config.shift_factor
    vae_config_scaling_factor = vae.config.scaling_factor
    vae_config_block_out_channels = vae.config.block_out_channels
    if args.cache_latents:
        latents_cache = []
        for batch in tqdm(train_dataloader, desc="Caching latents"):
            with torch.no_grad():
                batch["pixel_values"] = batch["pixel_values"].to(
                    accelerator.device, non_blocking=True, dtype=weight_dtype
                )
                latents_cache.append(vae.encode(batch["pixel_values"]).latent_dist)

        if args.validation_prompt is None:
            del vae
            free_memory()

    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(
        len(train_dataloader) / args.gradient_accumulation_steps
    )
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
        num_training_steps=args.max_train_steps * accelerator.num_processes,
        num_cycles=args.lr_num_cycles,
        power=args.lr_power,
    )

    transformer, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        transformer, optimizer, train_dataloader, lr_scheduler
    )

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(
        len(train_dataloader) / args.gradient_accumulation_steps
    )
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # Train!
    total_batch_size = (
        args.train_batch_size
        * accelerator.num_processes
        * args.gradient_accumulation_steps
    )

    with open(os.path.join(args.output_dir, args.logging_dir, "hparams.yml"), "w") as outfile:
        yaml.dump(vars(args), outfile)

    print("***** Running training *****")
    print(f"  Num examples = {len(train_dataset)}")
    print(f"  Num batches each epoch = {len(train_dataloader)}")
    print(f"  Num Epochs = {args.num_train_epochs}")
    print(f"  Instantaneous batch size per device = {args.train_batch_size}")
    print(
        f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}"
    )
    print(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    print(f"  Total optimization steps = {args.max_train_steps}")
    global_step = 0
    first_epoch = 0

    initial_global_step = 0

    progress_bar = tqdm(
        range(0, args.max_train_steps),
        initial=initial_global_step,
        desc="Steps",
        # Only show the progress bar once on each machine.
        disable=not accelerator.is_local_main_process,
    )

    def get_rank_by_timestep(timestep, max_timestep, max_rank, min_rank=1):
        r = (
            int((max_timestep - timestep) * (max_rank - min_rank) / max_timestep)
            + min_rank
        )
        return r

    def get_sigmas(timesteps, n_dim=4, dtype=torch.float32):
        sigmas = noise_scheduler_copy.sigmas.to(device=accelerator.device, dtype=dtype)
        schedule_timesteps = noise_scheduler_copy.timesteps.to(accelerator.device)
        timesteps = timesteps.to(accelerator.device)
        step_indices = [(schedule_timesteps == t).nonzero().item() for t in timesteps]

        sigma = sigmas[step_indices].flatten()
        while len(sigma.shape) < n_dim:
            sigma = sigma.unsqueeze(-1)
        return sigma

    for epoch in range(first_epoch, args.num_train_epochs):
        transformer.train()

        for step, batch in enumerate(train_dataloader):
            models_to_accumulate = [transformer]
            with accelerator.accumulate(models_to_accumulate):
                prompts = batch["prompts"]
                # Convert images to latent space
                if args.cache_latents:
                    model_input = latents_cache[step].sample()
                else:
                    pixel_values = batch["pixel_values"].to(dtype=vae.dtype)
                    model_input = vae.encode(pixel_values).latent_dist.sample()
                model_input = (
                    model_input - vae_config_shift_factor
                ) * vae_config_scaling_factor
                model_input = model_input.to(dtype=weight_dtype)

                vae_scale_factor = 2 ** (len(vae_config_block_out_channels))

                latent_image_ids = FluxPipeline._prepare_latent_image_ids(
                    model_input.shape[0],
                    model_input.shape[2] // 2,
                    model_input.shape[3] // 2,
                    accelerator.device,
                    weight_dtype,
                )
                # Sample noise that we'll add to the latents
                noise = torch.randn_like(model_input)
                bsz = model_input.shape[0]

                u = compute_density_for_timestep_sampling(
                    weighting_scheme=args.weighting_scheme,
                    batch_size=bsz,
                    logit_mean=args.logit_mean,
                    logit_std=args.logit_std,
                    mode_scale=args.mode_scale,
                )
                indices = (u * noise_scheduler_copy.config.num_train_timesteps).long()
                timesteps = noise_scheduler_copy.timesteps[indices].to(
                    device=model_input.device
                )
                # Add noise according to flow matching.
                # zt = (1 - texp) * x + texp * z1
                sigmas = get_sigmas(
                    timesteps, n_dim=model_input.ndim, dtype=model_input.dtype
                )
                noisy_model_input = (1.0 - sigmas) * model_input + sigmas * noise

                packed_noisy_model_input = FluxPipeline._pack_latents(
                    noisy_model_input,
                    batch_size=model_input.shape[0],
                    num_channels_latents=model_input.shape[1],
                    height=model_input.shape[2],
                    width=model_input.shape[3],
                )

                # handle guidance
                if transformer.config.guidance_embeds:
                    guidance = torch.tensor(
                        [args.guidance_scale], device=accelerator.device
                    )
                    guidance = guidance.expand(model_input.shape[0])
                else:
                    guidance = None

                if args.tlora:
                    r = get_rank_by_timestep(
                        timestep=timesteps[0],
                        max_timestep=noise_scheduler_copy.num_train_timesteps,
                        max_rank=args.rank,
                        min_rank=args.min_rank,
                    )
                    sigma_mask = torch.zeros((1, args.rank))
                    sigma_mask[:, :r] = 1.0
                    joint_attention_kwargs = {
                        "sigma_mask": sigma_mask.detach().to(
                            packed_noisy_model_input.device
                        )
                    }
                else:
                    joint_attention_kwargs = {}

                # Predict the noise residual
                model_pred = transformer(
                    hidden_states=packed_noisy_model_input,
                    timestep=timesteps / 1000,
                    guidance=guidance,
                    pooled_projections=pooled_prompt_embeds,
                    encoder_hidden_states=prompt_embeds,
                    txt_ids=text_ids,
                    img_ids=latent_image_ids,
                    return_dict=False,
                    joint_attention_kwargs=joint_attention_kwargs
                )[0]
                model_pred = FluxPipeline._unpack_latents(
                    model_pred,
                    height=model_input.shape[2] * vae_scale_factor,
                    width=model_input.shape[3] * vae_scale_factor,
                    vae_scale_factor=vae_scale_factor,
                )

                # these weighting schemes use a uniform timestep sampling
                # and instead post-weight the loss
                weighting = compute_loss_weighting_for_sd3(
                    weighting_scheme=args.weighting_scheme, sigmas=sigmas
                )

                # flow matching loss
                target = noise - model_input

                # Compute regular loss.
                if args.mask_dir:
                    mask = batch["mask"][0, 0]
                    loss = torch.mean(
                        (
                            weighting.float() * (model_pred.float() - target.float())[..., mask > 0] ** 2
                        ).reshape(target.shape[0], -1),
                        1,
                    )
                else:
                    loss = torch.mean(
                        (
                            weighting.float() * (model_pred.float() - target.float()) ** 2
                        ).reshape(target.shape[0], -1),
                        1,
                    )
                loss = loss.mean()

                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    params_to_clip = transformer.parameters()
                    accelerator.clip_grad_norm_(params_to_clip, args.max_grad_norm)

                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1

                if accelerator.is_main_process:
                    if global_step % args.checkpointing_steps == 0:
                        save_path = os.path.join(
                            args.output_dir, f"checkpoint-{global_step}"
                        )
                        os.makedirs(save_path, exist_ok=True)
                        torch.save(
                            lora_layers.state_dict(),
                            os.path.join(save_path, "lora.pt"),
                        )

                        text_encoder_one, text_encoder_two = load_text_encoders(
                            args, text_encoder_cls_one, text_encoder_cls_two
                        )
                        text_encoder_one.to(weight_dtype)
                        text_encoder_two.to(weight_dtype)

                        new_t = copy.deepcopy(transformer)

                        if args.tlora:
                            pipeline = TLoRAFluxPipeline.from_pretrained(
                                args.pretrained_model_name_or_path,
                                vae=vae,
                                text_encoder=unwrap_model(text_encoder_one, accelerator),
                                text_encoder_2=unwrap_model(text_encoder_two, accelerator),
                                transformer=unwrap_model(new_t.to(weight_dtype), accelerator),
                                revision=args.revision,
                                variant=args.variant,
                                torch_dtype=weight_dtype,
                                max_rank=args.rank,
                                min_rank = args.min_rank,
                            )
                        else:
                            pipeline = FluxPipeline.from_pretrained(
                                args.pretrained_model_name_or_path,
                                vae=vae,
                                text_encoder=unwrap_model(text_encoder_one, accelerator),
                                text_encoder_2=unwrap_model(text_encoder_two, accelerator),
                                transformer=unwrap_model(new_t.to(weight_dtype), accelerator),
                                revision=args.revision,
                                variant=args.variant,
                                torch_dtype=weight_dtype,
                            )

                        images = log_validation(
                            pipeline=pipeline,
                            args=args,
                            accelerator=accelerator,
                            pipeline_args={},
                            epoch=global_step,
                            torch_dtype=weight_dtype,
                        )
                        new_t.to("cpu")
                        del new_t

                        del text_encoder_one, text_encoder_two
                        free_memory()

                        del pipeline
                        gc.collect()
                        torch.cuda.empty_cache()

            logs = {"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
            progress_bar.set_postfix(**logs)

            if global_step >= args.max_train_steps:
                break

    accelerator.end_training()
