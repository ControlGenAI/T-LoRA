from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import inspect
import numpy as np
import torch
from transformers import (
    CLIPTextModel,
    CLIPTokenizer,
    T5EncoderModel,
    T5TokenizerFast,
)
from diffusers import FluxPipeline
from diffusers.pipelines.flux.pipeline_output import FluxPipelineOutput
from diffusers.pipelines.flux.pipeline_flux import calculate_shift
from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion import retrieve_timesteps
from diffusers import AutoencoderKL, FlowMatchEulerDiscreteScheduler, FluxTransformer2DModel
from diffusers.pipelines.flux.pipeline_flux_controlnet import FluxControlNetPipeline
from diffusers.pipelines.flux.pipeline_output import FluxPipelineOutput
from diffusers.models.controlnets.controlnet_flux import FluxControlNetModel, FluxMultiControlNetModel

XLA_AVAILABLE = False
USE_PEFT_BACKEND = False


class TLoRAFluxPipeline(FluxPipeline):
    def __init__(
        self,
        scheduler,
        vae,
        text_encoder,
        tokenizer,
        text_encoder_2,
        tokenizer_2,
        transformer,
        image_encoder=None,
        feature_extractor=None,
        max_rank=None,
        min_rank=1,
    ):
        super().__init__(
            scheduler=scheduler,
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            text_encoder_2=text_encoder_2,
            tokenizer_2=tokenizer_2,
            transformer=transformer,
            image_encoder=image_encoder,
            feature_extractor=feature_extractor,
        )
        self.max_rank = max_rank
        self.min_rank = min_rank

    def get_rank_by_timestep(self, timestep, max_timestep, max_rank, min_rank=1):
        r = (
            int((max_timestep - timestep) * (max_rank - min_rank) / max_timestep)
            + min_rank
        )
        return r

    @torch.no_grad()
    def __call__(
        self,
        prompt=None,
        prompt_2=None,
        negative_prompt=None,
        negative_prompt_2=None,
        true_cfg_scale=1.0,
        height=None,
        width=None,
        num_inference_steps=28,
        sigmas=None,
        guidance_scale=3.5,
        num_images_per_prompt=1,
        generator=None,
        latents=None,
        prompt_embeds=None,
        pooled_prompt_embeds=None,
        ip_adapter_image=None,
        ip_adapter_image_embeds=None,
        negative_ip_adapter_image=None,
        negative_ip_adapter_image_embeds=None,
        negative_prompt_embeds=None,
        negative_pooled_prompt_embeds=None,
        output_type="pil",
        return_dict=True,
        joint_attention_kwargs=None,
        callback_on_step_end=None,
        callback_on_step_end_tensor_inputs=["latents"],
        max_sequence_length=512,
    ):
        height = height or self.default_sample_size * self.vae_scale_factor
        width = width or self.default_sample_size * self.vae_scale_factor

        # 1. Check inputs. Raise error if not correct
        self.check_inputs(
            prompt,
            prompt_2,
            height,
            width,
            negative_prompt=negative_prompt,
            negative_prompt_2=negative_prompt_2,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
            callback_on_step_end_tensor_inputs=callback_on_step_end_tensor_inputs,
            max_sequence_length=max_sequence_length,
        )

        self._guidance_scale = guidance_scale
        self._joint_attention_kwargs = joint_attention_kwargs
        self._current_timestep = None
        self._interrupt = False

        # 2. Define call parameters
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        device = self._execution_device

        lora_scale = (
            self.joint_attention_kwargs.get("scale", None)
            if self.joint_attention_kwargs is not None
            else None
        )
        has_neg_prompt = negative_prompt is not None or (
            negative_prompt_embeds is not None
            and negative_pooled_prompt_embeds is not None
        )
        do_true_cfg = true_cfg_scale > 1 and has_neg_prompt
        (
            prompt_embeds,
            pooled_prompt_embeds,
            text_ids,
        ) = self.encode_prompt(
            prompt=prompt,
            prompt_2=prompt_2,
            prompt_embeds=prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            device=device,
            num_images_per_prompt=num_images_per_prompt,
            max_sequence_length=max_sequence_length,
            lora_scale=lora_scale,
        )
        if do_true_cfg:
            (
                negative_prompt_embeds,
                negative_pooled_prompt_embeds,
                _,
            ) = self.encode_prompt(
                prompt=negative_prompt,
                prompt_2=negative_prompt_2,
                prompt_embeds=negative_prompt_embeds,
                pooled_prompt_embeds=negative_pooled_prompt_embeds,
                device=device,
                num_images_per_prompt=num_images_per_prompt,
                max_sequence_length=max_sequence_length,
                lora_scale=lora_scale,
            )

        # 4. Prepare latent variables
        num_channels_latents = self.transformer.config.in_channels // 4
        latents, latent_image_ids = self.prepare_latents(
            batch_size * num_images_per_prompt,
            num_channels_latents,
            height,
            width,
            prompt_embeds.dtype,
            device,
            generator,
            latents,
        )

        # 5. Prepare timesteps
        sigmas = (
            np.linspace(1.0, 1 / num_inference_steps, num_inference_steps)
            if sigmas is None
            else sigmas
        )
        image_seq_len = latents.shape[1]
        mu = calculate_shift(
            image_seq_len,
            self.scheduler.config.get("base_image_seq_len", 256),
            self.scheduler.config.get("max_image_seq_len", 4096),
            self.scheduler.config.get("base_shift", 0.5),
            self.scheduler.config.get("max_shift", 1.15),
        )
        timesteps, num_inference_steps = retrieve_timesteps(
            self.scheduler,
            num_inference_steps,
            device,
            sigmas=sigmas,
            mu=mu,
        )
        num_warmup_steps = max(
            len(timesteps) - num_inference_steps * self.scheduler.order, 0
        )
        self._num_timesteps = len(timesteps)

        # handle guidance
        if self.transformer.config.guidance_embeds:
            guidance = torch.full(
                [1], guidance_scale, device=device, dtype=torch.float32
            )
            guidance = guidance.expand(latents.shape[0])
        else:
            guidance = None

        if (ip_adapter_image is not None or ip_adapter_image_embeds is not None) and (
            negative_ip_adapter_image is None
            and negative_ip_adapter_image_embeds is None
        ):
            negative_ip_adapter_image = np.zeros((width, height, 3), dtype=np.uint8)
            negative_ip_adapter_image = [
                negative_ip_adapter_image
            ] * self.transformer.encoder_hid_proj.num_ip_adapters

        elif (ip_adapter_image is None and ip_adapter_image_embeds is None) and (
            negative_ip_adapter_image is not None
            or negative_ip_adapter_image_embeds is not None
        ):
            ip_adapter_image = np.zeros((width, height, 3), dtype=np.uint8)
            ip_adapter_image = [
                ip_adapter_image
            ] * self.transformer.encoder_hid_proj.num_ip_adapters

        if self.joint_attention_kwargs is None:
            self._joint_attention_kwargs = {}

        image_embeds = None
        negative_image_embeds = None
        if ip_adapter_image is not None or ip_adapter_image_embeds is not None:
            image_embeds = self.prepare_ip_adapter_image_embeds(
                ip_adapter_image,
                ip_adapter_image_embeds,
                device,
                batch_size * num_images_per_prompt,
            )
        if (
            negative_ip_adapter_image is not None
            or negative_ip_adapter_image_embeds is not None
        ):
            negative_image_embeds = self.prepare_ip_adapter_image_embeds(
                negative_ip_adapter_image,
                negative_ip_adapter_image_embeds,
                device,
                batch_size * num_images_per_prompt,
            )

        # 6. Denoising loop
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                if self.interrupt:
                    continue

                self._current_timestep = t
                if image_embeds is not None:
                    self._joint_attention_kwargs["ip_adapter_image_embeds"] = (
                        image_embeds
                    )
                r = self.get_rank_by_timestep(
                    timestep=t,
                    max_timestep=self.scheduler.num_train_timesteps,
                    max_rank=self.max_rank,
                    min_rank=self.min_rank,
                )
                sigma_mask = torch.zeros((1, self.max_rank))
                sigma_mask[:, :r] = 1.0
                # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
                timestep = t.expand(latents.shape[0]).to(latents.dtype)

                self._joint_attention_kwargs["sigma_mask"] = sigma_mask.detach().to(
                    self.transformer.device
                )

                noise_pred = self.transformer(
                    hidden_states=latents,
                    timestep=timestep / 1000,
                    guidance=guidance,
                    pooled_projections=pooled_prompt_embeds,
                    encoder_hidden_states=prompt_embeds,
                    txt_ids=text_ids,
                    img_ids=latent_image_ids,
                    joint_attention_kwargs=self.joint_attention_kwargs,
                    return_dict=False,
                )[0]

                if do_true_cfg:
                    if negative_image_embeds is not None:
                        self._joint_attention_kwargs["ip_adapter_image_embeds"] = (
                            negative_image_embeds
                        )
                    neg_noise_pred = self.transformer(
                        hidden_states=latents,
                        timestep=timestep / 1000,
                        guidance=guidance,
                        pooled_projections=negative_pooled_prompt_embeds,
                        encoder_hidden_states=negative_prompt_embeds,
                        txt_ids=text_ids,
                        img_ids=latent_image_ids,
                        joint_attention_kwargs=self.joint_attention_kwargs,
                        return_dict=False,
                    )[0]
                    noise_pred = neg_noise_pred + true_cfg_scale * (
                        noise_pred - neg_noise_pred
                    )

                # compute the previous noisy sample x_t -> x_t-1
                latents_dtype = latents.dtype
                latents = self.scheduler.step(
                    noise_pred, t, latents, return_dict=False
                )[0]

                if latents.dtype != latents_dtype:
                    if torch.backends.mps.is_available():
                        # some platforms (eg. apple mps) misbehave due to a pytorch bug: https://github.com/pytorch/pytorch/pull/99272
                        latents = latents.to(latents_dtype)

                if callback_on_step_end is not None:
                    callback_kwargs = {}
                    for k in callback_on_step_end_tensor_inputs:
                        callback_kwargs[k] = locals()[k]
                    callback_outputs = callback_on_step_end(self, i, t, callback_kwargs)

                    latents = callback_outputs.pop("latents", latents)
                    prompt_embeds = callback_outputs.pop("prompt_embeds", prompt_embeds)

                # call the callback, if provided
                if i == len(timesteps) - 1 or (
                    (i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0
                ):
                    progress_bar.update()

                if XLA_AVAILABLE:
                    xm.mark_step()

        self._current_timestep = None

        if output_type == "latent":
            image = latents
        else:
            latents = self._unpack_latents(
                latents, height, width, self.vae_scale_factor
            )
            latents = (
                latents / self.vae.config.scaling_factor
            ) + self.vae.config.shift_factor
            image = self.vae.decode(latents, return_dict=False)[0]
            image = self.image_processor.postprocess(image, output_type=output_type)

        # Offload all models
        self.maybe_free_model_hooks()

        if not return_dict:
            return (image,)

        return FluxPipelineOutput(images=image)

class TLoRAFluxControlNetSwitchingPipeline(FluxControlNetPipeline):
    model_cpu_offload_seq = "text_encoder->text_encoder_2->transformer->vae"
    _optional_components = []
    _callback_tensor_inputs = ["latents", "prompt_embeds"]

    def __init__(
        self,
        scheduler: FlowMatchEulerDiscreteScheduler,
        vae: AutoencoderKL,
        text_encoder: CLIPTextModel,
        tokenizer: CLIPTokenizer,
        text_encoder_2: T5EncoderModel,
        tokenizer_2: T5TokenizerFast,
        transformer: FluxTransformer2DModel,
        controlnet: Union[
            FluxControlNetModel, List[FluxControlNetModel], Tuple[FluxControlNetModel], FluxMultiControlNetModel
        ],
        feature_extractor=None,
        image_encoder=None,
        max_rank=None,
        min_rank=1,
    ):
        super().__init__(
            scheduler=scheduler, vae=vae, text_encoder=text_encoder, tokenizer=tokenizer,
            text_encoder_2=text_encoder_2, tokenizer_2=tokenizer_2, transformer=transformer, controlnet=controlnet
        )
        self.max_rank = max_rank
        self.min_rank = min_rank

    def get_rank_by_timestep(self, timestep, max_timestep, max_rank, min_rank=1):
        r = (
            int((max_timestep - timestep) * (max_rank - min_rank) / max_timestep)
            + min_rank
        )
        return r

    @torch.no_grad()
    def __call__(
        self,
        prompt: Union[str, List[str]] = None,
        prompt_2: Optional[Union[str, List[str]]] = None,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: int = 28,
        timesteps: List[int] = None,
        guidance_scale: float = 7.0,
        control_image = None,
        control_mode: Optional[Union[int, List[int]]] = None,
        controlnet_conditioning_scale: Union[float, List[float]] = 1.0,
        num_images_per_prompt: Optional[int] = 1,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.FloatTensor] = None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        joint_attention_kwargs: Optional[Dict[str, Any]] = {},
        callback_on_step_end: Optional[Callable[[int, int, Dict], None]] = None,
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],
        max_sequence_length: int = 512,
        switch_t = 50,
    ):
        height = height or self.default_sample_size * self.vae_scale_factor
        width = width or self.default_sample_size * self.vae_scale_factor

        # 1. Check inputs. Raise error if not correct
        self.check_inputs(
            prompt,
            prompt_2,
            height,
            width,
            prompt_embeds=prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            callback_on_step_end_tensor_inputs=callback_on_step_end_tensor_inputs,
            max_sequence_length=max_sequence_length,
        )

        self._guidance_scale = guidance_scale
        self._joint_attention_kwargs = joint_attention_kwargs
        self._interrupt = False

        # 2. Define call parameters
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        device = self._execution_device
        dtype = self.transformer.dtype

        lora_scale = (
            self.joint_attention_kwargs.get("scale", None) if self.joint_attention_kwargs is not None else None
        )
        (
            prompt_embeds,
            pooled_prompt_embeds,
            text_ids,
        ) = self.encode_prompt(
            prompt=prompt,
            prompt_2=prompt_2,
            prompt_embeds=prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            device=device,
            num_images_per_prompt=num_images_per_prompt,
            max_sequence_length=max_sequence_length,
            lora_scale=lora_scale,
        )

        (
            prompt_embeds_neg,
            pooled_prompt_embeds_neg,
            text_ids_neg,
        ) = self.encode_prompt(
            prompt=negative_prompt,
            prompt_2=None,
            prompt_embeds=None,
            pooled_prompt_embeds=None,
            device=device,
            num_images_per_prompt=num_images_per_prompt,
            max_sequence_length=max_sequence_length,
            lora_scale=lora_scale,
        )

        # 3. Prepare control image
        num_channels_latents = self.transformer.config.in_channels // 4
        if isinstance(self.controlnet, FluxControlNetModel):
            control_image = self.prepare_image(
                image=control_image,
                width=width,
                height=height,
                batch_size=batch_size * num_images_per_prompt,
                num_images_per_prompt=num_images_per_prompt,
                device=device,
                dtype=dtype,
            )
            height, width = control_image.shape[-2:]

            # vae encode
            control_image = self.vae.encode(control_image).latent_dist.sample()
            control_image = (control_image - self.vae.config.shift_factor) * self.vae.config.scaling_factor

            # pack
            height_control_image, width_control_image = control_image.shape[2:]
            control_image = self._pack_latents(
                control_image,
                batch_size * num_images_per_prompt,
                num_channels_latents,
                height_control_image,
                width_control_image,
            )

            # set control mode
            if control_mode is not None:
                control_mode = torch.tensor(control_mode).to(device, dtype=torch.long)
                control_mode = control_mode.reshape([-1, 1])

        elif isinstance(self.controlnet, FluxMultiControlNetModel):
            control_images = []

            for control_image_ in control_image:
                control_image_ = self.prepare_image(
                    image=control_image_,
                    width=width,
                    height=height,
                    batch_size=batch_size * num_images_per_prompt,
                    num_images_per_prompt=num_images_per_prompt,
                    device=device,
                    dtype=dtype,
                )
                height, width = control_image_.shape[-2:]

                # vae encode
                control_image_ = self.vae.encode(control_image_).latent_dist.sample()
                control_image_ = (control_image_ - self.vae.config.shift_factor) * self.vae.config.scaling_factor

                # pack
                height_control_image, width_control_image = control_image_.shape[2:]
                control_image_ = self._pack_latents(
                    control_image_,
                    batch_size * num_images_per_prompt,
                    num_channels_latents,
                    height_control_image,
                    width_control_image,
                )

                control_images.append(control_image_)

            control_image = control_images

            # set control mode
            control_mode_ = []
            if isinstance(control_mode, list):
                for cmode in control_mode:
                    if cmode is None:
                        control_mode_.append(-1)
                    else:
                        control_mode_.append(cmode)
            control_mode = torch.tensor(control_mode_).to(device, dtype=torch.long)
            control_mode = control_mode.reshape([-1, 1])

        # 4. Prepare latent variables
        num_channels_latents = self.transformer.config.in_channels // 4
        latents, latent_image_ids = self.prepare_latents(
            batch_size * num_images_per_prompt,
            num_channels_latents,
            height,
            width,
            prompt_embeds.dtype,
            device,
            generator,
            latents,
        )

        # 5. Prepare timesteps
        sigmas = np.linspace(1.0, 1 / num_inference_steps, num_inference_steps)
        image_seq_len = latents.shape[1]
        mu = calculate_shift(
            image_seq_len,
            self.scheduler.config.base_image_seq_len,
            self.scheduler.config.max_image_seq_len,
            self.scheduler.config.base_shift,
            self.scheduler.config.max_shift,
        )
        timesteps, num_inference_steps = retrieve_timesteps(
            self.scheduler,
            num_inference_steps,
            device,
            timesteps,
            sigmas,
            mu=mu,
        )

        num_warmup_steps = max(len(timesteps) - num_inference_steps * self.scheduler.order, 0)
        self._num_timesteps = len(timesteps)

        # 6. Denoising loop
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                if self.interrupt:
                    continue

                r = self.get_rank_by_timestep(
                    timestep=t,
                    max_timestep=self.scheduler.num_train_timesteps,
                    max_rank=self.max_rank,
                    min_rank=self.min_rank,
                )
                sigma_mask = torch.zeros((1, self.max_rank))
                sigma_mask[:, :r] = 1.0

                self._joint_attention_kwargs["sigma_mask"] = sigma_mask.detach().to(
                    self.transformer.device
                )

                # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
                timestep = t.expand(latents.shape[0]).to(latents.dtype)

                # handle guidance
                if self.transformer.config.guidance_embeds:
                    guidance = torch.tensor([guidance_scale], device=device)
                    guidance = guidance.expand(latents.shape[0])
                else:
                    guidance = None

                if i < switch_t:
                    controlnet_block_samples, controlnet_single_block_samples = self.controlnet(
                        hidden_states=latents,
                        controlnet_cond=control_image,
                        controlnet_mode=control_mode,
                        conditioning_scale=controlnet_conditioning_scale,
                        timestep=timestep / 1000,
                        guidance=guidance,
                        pooled_projections=pooled_prompt_embeds,
                        encoder_hidden_states=prompt_embeds,
                        txt_ids=text_ids,
                        img_ids=latent_image_ids,
                        joint_attention_kwargs={},
                        return_dict=False,
                    )
                    noise_pred = self.transformer(
                        hidden_states=latents,
                        timestep=timestep / 1000,
                        guidance=guidance,
                        pooled_projections=pooled_prompt_embeds,
                        encoder_hidden_states=prompt_embeds,
                        controlnet_block_samples=controlnet_block_samples,
                        controlnet_single_block_samples=controlnet_single_block_samples,
                        txt_ids=text_ids,
                        img_ids=latent_image_ids,
                        joint_attention_kwargs=self.joint_attention_kwargs,
                        return_dict=False,
                    )[0]
                else:
                    noise_pred = self.transformer(
                        hidden_states=latents,
                        timestep=timestep / 1000,
                        guidance=guidance,
                        pooled_projections=pooled_prompt_embeds,
                        encoder_hidden_states=prompt_embeds,
                        txt_ids=text_ids,
                        img_ids=latent_image_ids,
                        joint_attention_kwargs=self.joint_attention_kwargs,
                        return_dict=False,
                    )[0]

                # compute the previous noisy sample x_t -> x_t-1
                latents_dtype = latents.dtype
                latents = self.scheduler.step(noise_pred, t, latents, return_dict=False)[0]

                if latents.dtype != latents_dtype:
                    if torch.backends.mps.is_available():
                        # some platforms (eg. apple mps) misbehave due to a pytorch bug: https://github.com/pytorch/pytorch/pull/99272
                        latents = latents.to(latents_dtype)

                if callback_on_step_end is not None:
                    callback_kwargs = {}
                    for k in callback_on_step_end_tensor_inputs:
                        callback_kwargs[k] = locals()[k]
                    callback_outputs = callback_on_step_end(self, i, t, callback_kwargs)

                    latents = callback_outputs.pop("latents", latents)
                    prompt_embeds = callback_outputs.pop("prompt_embeds", prompt_embeds)

                # call the callback, if provided
                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()

                if XLA_AVAILABLE:
                    xm.mark_step()

        if output_type == "latent":
            image = latents

        else:
            latents = self._unpack_latents(latents, height, width, self.vae_scale_factor)
            latents = (latents / self.vae.config.scaling_factor) + self.vae.config.shift_factor

            image = self.vae.decode(latents, return_dict=False)[0]
            image = self.image_processor.postprocess(image, output_type=output_type)

        # Offload all models
        self.maybe_free_model_hooks()

        if not return_dict:
            return (image,)

        return FluxPipelineOutput(images=image)