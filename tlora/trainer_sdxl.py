import os
import gc
from typing import Callable, Any, Optional

import math
import yaml
import random
import secrets
import logging
import itertools
from collections import defaultdict

from tqdm import tqdm
import wandb

import numpy as np

import torch
from torch.utils.data import DataLoader

import diffusers
from diffusers import (
    AutoencoderKL, EulerDiscreteScheduler, DDPMScheduler, UNet2DConditionModel, StableDiffusionXLPipeline,
)
from diffusers.loaders import AttnProcsLayers

from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration

import transformers
import transformers.utils.logging
from transformers import CLIPTokenizer, CLIPTextModel, CLIPTextModelWithProjection

from .utils.registry import ClassRegistry
from .utils.model import get_layer_by_name
from .model.lora import (
    lora_linear_layers, lora_prosessors, LOrthogonalLoRACrossAttnProcessor
)
from .model.pipeline_sdxl import StableDiffusionTLoRAPipeline
from .model.utils_sdxl import cast_training_params
from .data.dataset_sdxl import (
    ImageDataset, DreamBoothDataset, collate_fn, tokenize_prompt, encode_tokens, compute_time_ids,
)

logger = get_logger(__name__)
trainers = ClassRegistry()

BASE_PROMPT = "a photo of {0}"

torch.backends.cuda.enable_flash_sdp(True)


@trainers.add_to_registry("sdxl_lora")
class LoraTrainerSDXL:

    def __init__(self, config):
        self.config = config

    def setup_exp_name(self, exp_idx):
        exp_name = "{0:0>5d}-{1}-{2}".format(
            exp_idx + 1,
            secrets.token_hex(2),
            os.path.basename(os.path.normpath(self.config.train_data_dir)),
        )
        exp_name += f"_{self.config.trainer_type}{self.config.lora_rank}"
        
        return exp_name

    def setup_exp(self):
        os.makedirs(self.config.output_dir, exist_ok=True)
        # Get the last experiment idx
        exp_idx = 0
        for folder in os.listdir(self.config.output_dir):
            # noinspection PyBroadException
            try:
                curr_exp_idx = max(exp_idx, int(folder.split("-")[0].lstrip("0")))
                exp_idx = max(exp_idx, curr_exp_idx)
            except:
                pass

        self.config.exp_name = self.setup_exp_name(exp_idx)

        self.config.output_dir = os.path.abspath(
            os.path.join(self.config.output_dir, self.config.exp_name)
        )

        if os.path.exists(self.config.output_dir):
            raise ValueError(
                f"Experiment directory {self.config.output_dir} already exists. Race condition!"
            )
        os.makedirs(self.config.output_dir, exist_ok=True)

        self.config.logging_dir = os.path.join(self.config.output_dir, "logs")
        os.makedirs(self.config.logging_dir, exist_ok=True)

        with open(os.path.join(self.config.logging_dir, "hparams.yml"), "w") as outfile:
            yaml.dump(vars(self.config), outfile)

    def setup_accelerator(self):
        if self.config.wandb_api_key is not None:
            wandb.login(key=self.config.wandb_api_key)

        accelerator_project_config = ProjectConfiguration(
            project_dir=self.config.output_dir
        )
        self.accelerator = Accelerator(
            mixed_precision=self.config.mixed_precision,
            log_with="wandb",
            project_config=accelerator_project_config,
        )
        if self.config.wandb_api_key is not None:
            self.accelerator.init_trackers(
                project_name=self.config.project_name,
                config=self.config,
                init_kwargs={
                    "wandb": {
                        "name": self.config.exp_name,
                        "settings": wandb.Settings(
                            code_dir=os.path.dirname(self.config.argv[1])
                        ),
                    }
                },
            )

        logging.basicConfig(
            format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
            datefmt="%m/%d/%Y %H:%M:%S",
            level=logging.INFO,
        )
        logger.info(self.accelerator.state, main_process_only=False)
        if self.accelerator.is_local_main_process:
            transformers.utils.logging.set_verbosity_warning()
            diffusers.utils.logging.set_verbosity_info()
        else:
            transformers.utils.logging.set_verbosity_error()
            diffusers.utils.logging.set_verbosity_error()

        self.weight_dtype = torch.float32
        if self.accelerator.mixed_precision == "fp16":
            self.weight_dtype = torch.float16
        elif self.accelerator.mixed_precision == "bf16":
            self.weight_dtype = torch.bfloat16

    def setup_base_model(self):
        self.scheduler = DDPMScheduler.from_pretrained(
            self.config.pretrained_model_name_or_path,
            subfolder="scheduler",
            revision=self.config.revision,
        )
        self.unet = UNet2DConditionModel.from_pretrained(
            self.config.pretrained_model_name_or_path,
            subfolder="unet",
            revision=self.config.revision,
        )
        self.text_encoder = CLIPTextModel.from_pretrained(
            self.config.pretrained_model_name_or_path,
            subfolder="text_encoder",
            revision=self.config.revision,
        )
        self.text_encoder_2 = CLIPTextModelWithProjection.from_pretrained(
            self.config.pretrained_model_name_or_path,
            subfolder="text_encoder_2",
            revision=self.config.revision,
        )
        self.vae = AutoencoderKL.from_pretrained(
            "madebyollin/sdxl-vae-fp16-fix",
        )
        self.tokenizer = CLIPTokenizer.from_pretrained(
            self.config.pretrained_model_name_or_path,
            subfolder="tokenizer",
            revision=self.config.revision,
        )
        self.tokenizer_2 = CLIPTokenizer.from_pretrained(
            self.config.pretrained_model_name_or_path,
            subfolder="tokenizer_2",
            revision=self.config.revision,
        )

    def setup_model(self):
        self.vae.requires_grad_(False)
        self.unet.requires_grad_(False)
        self.text_encoder.requires_grad_(False)
        self.text_encoder_2.requires_grad_(False)

        lora_attn_processor = lora_prosessors[self.config.trainer_type]
        lora_linear_layer = lora_linear_layers[self.config.trainer_type]

        self.params_to_optimize = []
        lora_attn_procs = {}
        for name in self.unet.attn_processors.keys():
            cross_attention_dim = (
                None
                if name.endswith("attn1.processor")
                else self.unet.config.cross_attention_dim
            )
            if name.startswith("mid_block"):
                hidden_size = self.unet.config.block_out_channels[-1]
            elif name.startswith("up_blocks"):
                block_id = int(name[len("up_blocks.")])
                hidden_size = list(reversed(self.unet.config.block_out_channels))[
                    block_id
                ]
            elif name.startswith("down_blocks"):
                block_id = int(name[len("down_blocks.")])
                hidden_size = self.unet.config.block_out_channels[block_id]
            if cross_attention_dim:
                rank = min(cross_attention_dim, hidden_size, self.config.lora_rank)
            else:
                rank = min(hidden_size, self.config.lora_rank)
            kwargs = {
                "hidden_size": hidden_size,
                "cross_attention_dim": cross_attention_dim,
                "rank": rank,
                "lora_linear_layer": lora_linear_layer,
                "sig_type": self.config.sig_type,
                "do_training": True,
            }
            if isinstance(lora_attn_processor, LOrthogonalLoRACrossAttnProcessor):
                kwargs["original_layer"] = get_layer_by_name(self.unet, name.split(".processor")[0])

            lora_attn_procs[name] = lora_attn_processor(**kwargs)

        self.unet.set_attn_processor(lora_attn_procs)
        self.lora_layers = AttnProcsLayers(self.unet.attn_processors)
        self.accelerator.register_for_checkpointing(self.lora_layers)

        for name, param in self.lora_layers.named_parameters():
            if param.requires_grad == True:
                self.params_to_optimize.append(param)
        self.lora_layers.train()

    def setup_optimizer(self):
        self.optimizer = torch.optim.AdamW(
            self.params_to_optimize,
            lr=self.config.learning_rate,
            betas=(self.config.adam_beta1, self.config.adam_beta2),
            weight_decay=self.config.adam_weight_decay,
            eps=self.config.adam_epsilon,
        )

    def setup_lr_scheduler(self):
        pass

    def setup_dataset(self):
        if self.config.with_prior_preservation:
            self.train_dataset = DreamBoothDataset(
                instance_data_root=self.config.train_data_dir,
                instance_prompt=BASE_PROMPT.format(
                    f"{self.config.placeholder_token} {self.config.class_name}"
                ),
                class_data_root=(
                    self.config.class_data_dir
                    if self.config.with_prior_preservation
                    else None
                ),
                class_prompt=BASE_PROMPT.format(self.config.class_name),
                tokenizers=(self.tokenizer, self.tokenizer_2),
                size=self.config.resolution,
            )
            collator: Optional[Callable[[Any], dict[str, torch.Tensor]]] = (
                lambda examples: collate_fn(
                    examples, self.config.with_prior_preservation
                )
            )
        else:
            self.train_dataset = ImageDataset(
                train_data_dir=self.config.train_data_dir,
                resolution=self.config.resolution,
                one_image=self.config.one_image,
            )
            collator = None

        self.train_dataloader = DataLoader(
            self.train_dataset,
            batch_size=self.config.train_batch_size,
            shuffle=True,
            collate_fn=collator,
            num_workers=self.config.dataloader_num_workers,
            generator=self.generator,
        )

    # noinspection PyTypeChecker
    def move_to_device(self):
        self.lora_layers, self.optimizer, self.train_dataloader = self.accelerator.prepare(
            self.lora_layers, self.optimizer, self.train_dataloader
        )
        self.vae.to(self.accelerator.device, dtype=self.weight_dtype)
        self.unet.to(self.accelerator.device, dtype=self.weight_dtype)
        self.text_encoder.to(self.accelerator.device, dtype=self.weight_dtype)
        self.text_encoder_2.to(self.accelerator.device, dtype=self.weight_dtype)

        # # All trained parameters should be explicitly moved to float32 even for mixed precision training
        cast_training_params(
            (self.unet, self.text_encoder, self.text_encoder_2), dtype=torch.float32
        )

    def setup_seed(self):
        torch.manual_seed(self.config.seed)
        random.seed(self.config.seed)
        np.random.seed(self.config.seed)

        self.generator = torch.Generator()
        self.generator.manual_seed(self.config.seed)

    def setup(self):
        self.setup_exp()
        self.setup_accelerator()
        self.setup_seed()
        self.setup_base_model()
        self.setup_model()
        self.setup_optimizer()
        self.setup_lr_scheduler()
        self.setup_dataset()
        self.move_to_device()
        self.setup_pipeline()

    def train_step(self, batch):
        if self.config.with_prior_preservation:
            latents = self.vae.encode(
                batch["pixel_values"].to(self.weight_dtype)
            ).latent_dist.sample()
        else:
            latents = self.vae.encode(
                batch["image"].to(self.weight_dtype) * 2.0 - 1.0
            ).latent_dist.sample()
        latents = latents * self.vae.config.scaling_factor

        noise = torch.randn_like(latents)

        timesteps = torch.randint(
            0,
            self.scheduler.num_train_timesteps,
            (latents.shape[0],),
            device=latents.device,
        )

        if self.scheduler.config.prediction_type == "epsilon":
            target = noise
        elif self.scheduler.config.prediction_type == "v_prediction":
            target = self.scheduler.get_velocity(latents, noise, timesteps)
        else:
            raise ValueError(
                f"Unknown prediction type {self.scheduler.config.prediction_type}"
            )

        noisy_latents = self.scheduler.add_noise(latents, noise, timesteps)

        # Get encoder_hidden_states
        if not self.config.with_prior_preservation:
            input_ids_list = tokenize_prompt(
                (self.tokenizer, self.tokenizer_2),
                BASE_PROMPT.format(
                    f"{self.config.placeholder_token} {self.config.class_name}"
                ),
            )
            encoder_hidden_states, pooled_encoder_hidden_states = encode_tokens(
                (self.text_encoder, self.text_encoder_2), input_ids_list
            )
        else:
            encoder_hidden_states, pooled_encoder_hidden_states = encode_tokens(
                (self.text_encoder, self.text_encoder_2),
                (batch["input_ids"], batch["input_ids_2"]),
            )

        add_time_ids = compute_time_ids(
            original_size=batch["original_sizes"],
            crops_coords_top_left=batch["crop_top_lefts"],
            resolution=self.config.resolution,
        )
        unet_added_conditions = {
            "time_ids": add_time_ids,
            "text_embeds": pooled_encoder_hidden_states,
        }

        outputs = self.unet(
            noisy_latents,
            timesteps,
            encoder_hidden_states,
            added_cond_kwargs=unet_added_conditions,
        ).sample

        if self.config.with_prior_preservation:
            outputs, prior_outputs = torch.chunk(outputs, 2, dim=0)
            target, prior_target = torch.chunk(target, 2, dim=0)

            # Compute instance loss
            loss = torch.nn.functional.mse_loss(
                outputs.float(), target.float(), reduction="mean"
            )

            # Compute prior loss
            prior_loss = torch.nn.functional.mse_loss(
                prior_outputs.float(), prior_target.float(), reduction="mean"
            )

            # Add the prior loss to the instance loss.
            loss = loss + self.config.prior_loss_weight * prior_loss
        else:
            loss = torch.nn.functional.mse_loss(
                outputs.float(), target.float(), reduction="mean"
            )

        return loss

    def setup_pipeline(self):
        scheduler = EulerDiscreteScheduler.from_pretrained(
            self.config.pretrained_model_name_or_path, subfolder="scheduler"
        )
        self.pipeline = StableDiffusionXLPipeline.from_pretrained(
            self.config.pretrained_model_name_or_path,
            scheduler=scheduler,
            tokenizer=self.tokenizer,
            tokenizer_2=self.tokenizer_2,
            text_encoder=self.text_encoder,
            text_encoder_2=self.text_encoder_2,
            unet=self.accelerator.unwrap_model(self.unet, keep_fp32_wrapper=False),
            vae=self.vae,
            revision=self.config.revision,
            torch_dtype=(
                self.weight_dtype
                if self.accelerator.mixed_precision in ["fp16", "bf16"]
                else torch.float16
            ),
        )

        self.pipeline.safety_checker = None
        self.pipeline = self.pipeline.to(self.accelerator.device)
        self.pipeline.set_progress_bar_config(disable=True)

    @torch.no_grad()
    def validation(self, epoch):
        generator = torch.Generator(device=self.accelerator.device).manual_seed(42)
        prompts = self.config.validation_prompts.split('#')

        samples_path = os.path.join(
            self.config.output_dir,
            f"checkpoint-{epoch}",
            "samples",
            "ns0_gs0_validation",
            "version_0",
        )
        os.makedirs(samples_path, exist_ok=True)

        all_images, all_captions = [], []
        for prompt in prompts:
            with torch.autocast("cuda"):
                caption = prompt.format(
                    f"{self.config.placeholder_token} {self.config.class_name}"
                )
                kwargs = {
                    "num_inference_steps": 25,
                    "guidance_scale": 5.0,
                    "prompt": caption,
                    "num_images_per_prompt": self.config.num_val_imgs_per_prompt,
                }
                images = self.pipeline(generator=generator, **kwargs).images
                gc.collect()
                torch.cuda.empty_cache()

            all_images += images
            all_captions += [caption] * len(images)

            os.makedirs(os.path.join(samples_path, caption), exist_ok=True)
            for idx, image in enumerate(images):
                image.save(os.path.join(samples_path, caption, f"{idx}.png"))

        for tracker in self.accelerator.trackers:
            tracker.log(
                {
                    "validation": [
                        wandb.Image(image, caption=caption)
                        for image, caption in zip(all_images, all_captions)
                    ]
                }
            )
        torch.cuda.empty_cache()

    def save_model(self, epoch):
        save_path = os.path.join(self.config.output_dir, f"checkpoint-{epoch}")
        os.makedirs(save_path, exist_ok=True)
        self.unet.save_attn_procs(os.path.join(save_path))

    def train(self):
        for epoch in tqdm(range(self.config.num_train_epochs)):
            batch = next(iter(self.train_dataloader))
            with self.accelerator.autocast():
                loss = self.train_step(batch)

            self.accelerator.backward(loss)

            logs_dict = {
                "loss": loss,
            }
            self.optimizer.step()
            self.optimizer.zero_grad()

            del batch, loss
            gc.collect()
            torch.cuda.empty_cache()

            for tracker in self.accelerator.trackers:
                tracker.log(logs_dict)

            if self.accelerator.is_main_process:
                if epoch % self.config.checkpointing_steps == 0 and epoch != 0:
                    if self.config.wandb_api_key is not None and self.config.validation_prompts:
                        self.validation(epoch)
                    self.save_model(epoch)

            gc.collect()
            torch.cuda.empty_cache()
            gc.collect()
            torch.cuda.empty_cache()

        if self.accelerator.is_main_process:
            if self.config.wandb_api_key is not None and self.config.validation_prompts:
                self.validation(self.config.num_train_epochs)
            self.save_model(self.config.num_train_epochs)

        self.accelerator.end_training()


@trainers.add_to_registry("sdxl_tlora")
class TLoraTrainerSDXL(LoraTrainerSDXL):

    def __init__(self, config):
        super().__init__(config)

    def setup_exp_name(self, exp_idx):
        exp_name = "{0:0>5d}-{1}-{2}".format(
            exp_idx + 1,
            secrets.token_hex(2),
            os.path.basename(os.path.normpath(self.config.train_data_dir)),
        )
        exp_name += f"_t{self.config.trainer_type}{self.config.lora_rank}"
        return exp_name

    def get_mask_by_timestep(self, timestep, max_timestep, max_rank, min_rank=1, alpha=1):
        r = int(((max_timestep - timestep)/ max_timestep) ** alpha * (max_rank - min_rank)) + min_rank
        sigma_mask = torch.zeros((1, self.config.lora_rank))
        sigma_mask[:, :r] = 1.0
        return sigma_mask

    def train_step(self, batch):
        if self.config.with_prior_preservation:
            latents = self.vae.encode(
                batch["pixel_values"].to(self.weight_dtype)
            ).latent_dist.sample()
        else:
            latents = self.vae.encode(
                batch["image"].to(self.weight_dtype) * 2.0 - 1.0
            ).latent_dist.sample()
        latents = latents * self.vae.config.scaling_factor

        noise = torch.randn_like(latents)
        timesteps = torch.randint(
            0,
            self.scheduler.num_train_timesteps,
            (latents.shape[0],),
            device=latents.device,
        )

        sigma_mask = self.get_mask_by_timestep(
            timestep=timesteps[0],
            max_timestep=self.scheduler.num_train_timesteps,
            max_rank=self.config.lora_rank,
            min_rank=self.config.min_rank,
            alpha=self.config.alpha_rank_scale,
        )

        if self.scheduler.config.prediction_type == "epsilon":
            target = noise
        elif self.scheduler.config.prediction_type == "v_prediction":
            target = self.scheduler.get_velocity(latents, noise, timesteps)
        else:
            raise ValueError(
                f"Unknown prediction type {self.scheduler.config.prediction_type}"
            )

        noisy_latents = self.scheduler.add_noise(latents, noise, timesteps)

        # Get encoder_hidden_states
        if not self.config.with_prior_preservation:
            input_ids_list = tokenize_prompt(
                (self.tokenizer, self.tokenizer_2),
                BASE_PROMPT.format(
                    f"{self.config.placeholder_token} {self.config.class_name}"
                ),
            )
            encoder_hidden_states, pooled_encoder_hidden_states = encode_tokens(
                (self.text_encoder, self.text_encoder_2), input_ids_list
            )
        else:
            encoder_hidden_states, pooled_encoder_hidden_states = encode_tokens(
                (self.text_encoder, self.text_encoder_2),
                (batch["input_ids"], batch["input_ids_2"]),
            )
        add_time_ids = compute_time_ids(
            original_size=batch["original_sizes"],
            crops_coords_top_left=batch["crop_top_lefts"],
            resolution=self.config.resolution,
        )
        unet_added_conditions = {
            "time_ids": add_time_ids,
            "text_embeds": pooled_encoder_hidden_states,
        }
        outputs = self.unet(
            noisy_latents,
            timesteps,
            encoder_hidden_states,
            added_cond_kwargs=unet_added_conditions,
            cross_attention_kwargs={
                "sigma_mask": sigma_mask.detach().to(encoder_hidden_states.device)
            },
        ).sample

        if self.config.with_prior_preservation:
            outputs, prior_outputs = torch.chunk(outputs, 2, dim=0)
            target, prior_target = torch.chunk(target, 2, dim=0)

            # Compute instance loss
            loss = torch.nn.functional.mse_loss(
                outputs.float(), target.float(), reduction="mean"
            )

            # Compute prior loss
            prior_loss = torch.nn.functional.mse_loss(
                prior_outputs.float(), prior_target.float(), reduction="mean"
            )

            # Add the prior loss to the instance loss.
            loss = loss + self.config.prior_loss_weight * prior_loss
        else:
            loss = torch.nn.functional.mse_loss(
                outputs.float(), target.float(), reduction="mean"
            )

        return loss

    def setup_pipeline(self):
        scheduler = EulerDiscreteScheduler.from_pretrained(
            self.config.pretrained_model_name_or_path, subfolder="scheduler"
        )
        self.pipeline = StableDiffusionTLoRAPipeline.from_pretrained(
            self.config.pretrained_model_name_or_path,
            scheduler=scheduler,
            tokenizer=self.tokenizer,
            tokenizer_2=self.tokenizer_2,
            text_encoder=self.text_encoder,
            text_encoder_2=self.text_encoder_2,
            unet=self.accelerator.unwrap_model(self.unet, keep_fp32_wrapper=False),
            vae=self.vae,
            revision=self.config.revision,
            torch_dtype=(
                self.weight_dtype
                if self.accelerator.mixed_precision in ["fp16", "bf16"]
                else torch.float16
            ),
            max_rank=self.config.lora_rank,
            min_rank=self.config.min_rank,
            alpha=self.config.alpha_rank_scale,
        )

        self.pipeline.safety_checker = None
        self.pipeline = self.pipeline.to(self.accelerator.device)
        self.pipeline.set_progress_bar_config(disable=True)

    @torch.no_grad()
    def validation_loss(self, batch, acc_step, generator):
        latents = self.vae.encode(
            batch["image"].to(self.weight_dtype) * 2.0 - 1.0
        ).latent_dist.sample(generator=generator)
        latents = latents * self.vae.config.scaling_factor

        noise = self.eval_noise[acc_step * self.config.eval_batch_size: (acc_step + 1) * self.config.eval_batch_size]
        timesteps = self.rand_perm[acc_step * self.config.eval_batch_size: (acc_step + 1) * self.config.eval_batch_size]

        sigma_mask = self.get_mask_by_timestep(
            timestep=timesteps[0],
            max_timestep=self.scheduler.num_train_timesteps,
            max_rank=self.config.lora_rank,
            min_rank=self.config.min_rank,
            alpha=self.config.alpha_rank_scale,
        )

        if self.scheduler.config.prediction_type == "epsilon":
            target = noise
        elif self.scheduler.config.prediction_type == "v_prediction":
            target = self.scheduler.get_velocity(latents, noise, timesteps)
        else:
            raise ValueError(
                f"Unknown prediction type {self.scheduler.config.prediction_type}"
            )

        noisy_latents = self.scheduler.add_noise(latents.to(self.unet.device), noise.to(self.unet.device), timesteps.to(self.unet.device))

        # Get encoder_hidden_states
        input_ids_list = tokenize_prompt(
            (self.tokenizer, self.tokenizer_2),
            BASE_PROMPT.format(
                f"{self.config.placeholder_token} {self.config.class_name}"
            ),
        )
        encoder_hidden_states, pooled_encoder_hidden_states = encode_tokens(
            (self.text_encoder, self.text_encoder_2), input_ids_list
        )

        add_time_ids = compute_time_ids(
            original_size=batch["original_sizes"],
            crops_coords_top_left=batch["crop_top_lefts"],
            resolution=self.config.resolution,
        )
        unet_added_conditions = {
            "time_ids": add_time_ids,
            "text_embeds": pooled_encoder_hidden_states,
        }
        outputs = self.unet(
            noisy_latents,
            timesteps.to(self.unet.device),
            encoder_hidden_states,
            added_cond_kwargs=unet_added_conditions,
            cross_attention_kwargs={
                "sigma_mask": sigma_mask.detach().to(encoder_hidden_states.device)
            },
        ).sample

        loss = torch.nn.functional.mse_loss(
            outputs.float(), target.to(self.unet.device).float(), reduction="mean"
        )
        return loss.item()
