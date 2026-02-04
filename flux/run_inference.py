import os
import yaml
import argparse
import random
import numpy as np
from PIL import Image
import tqdm
import gc

import torch
import torchvision

from diffusers.pipelines import FluxPipeline
from diffusers.loaders import AttnProcsLayers

from model.lora import (
    FluxLoraAttnProcessor,
    LoRALinearLayer,
    set_flux_transformer_attn_processor,
)
from model.utils import unwrap_model, load_text_encoders, get_layer_by_name
from model.pipeline import TLoRAFluxPipeline


def parse_args(input_args=None):
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument(
        "--exp",
        type=str,
        default=None,
        required=True,
    )
    parser.add_argument(
        "--prompts",
        type=str,
        default=None,
        required=True,
    )
    parser.add_argument(
        "--checkpoint_idx",
        type=int,
        default=700,
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
    )
    args = parser.parse_args()
    return args


def get_seed(prompt, i, seed):
    h = 0
    for el in prompt:
        h += ord(el)
    h += i
    return h + seed


def generate_with_prompt(pipe, prompt, config, args, num_images_per_prompt=5, batch_size=1):
    n_batches = (num_images_per_prompt - 1) // batch_size + 1
    images = []
    pipe_kwargs = {
        "guidance_scale": 3.5,
        "num_inference_steps": 28,
    }
    for i in range(n_batches):
        seed = get_seed(prompt, i, args.seed)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        generator = torch.Generator(device="cuda")
        generator = generator.manual_seed(seed)
        images_batch = pipe(
            prompt=prompt.format(f"{config['placeholder_token']} {config['class_name']}"),
            generator=generator,
            num_images_per_prompt=batch_size,
            **pipe_kwargs,
        ).images
        images += images_batch

        gc.collect()
        torch.cuda.empty_cache()
    return images


def save_images(images, path):
    os.makedirs(path, exist_ok=True)
    for idx, image in enumerate(images):
        image.save(os.path.join(path, f"{idx}.png"))


def generate_with_prompt_list(pipe, config, args, prompts, num_images_per_prompt=5, batch_size=1):
    checkpoint_path = os.path.join(config["output_dir"], f"checkpoint-{args.checkpoint_idx}")
    samples_path = os.path.join(
        checkpoint_path, "samples", "ns28gs3.5", "version_0",
    )
    os.makedirs(samples_path, exist_ok=True)
    for prompt in tqdm.tqdm(prompts):
        formatted_prompt = prompt.format(f"{config['placeholder_token']}")
        path = os.path.join(samples_path, formatted_prompt)

        images = generate_with_prompt(
            pipe, prompt, config, args, num_images_per_prompt, batch_size
        )
        save_images(images, path)


def main(args):
    with open(os.path.join(args.exp, 'logs', 'hparams.yml'), "r", encoding="utf-8") as config_file:
        config = yaml.safe_load(config_file)

    if config['tlora']:
        pipe = TLoRAFluxPipeline.from_pretrained(
            config['pretrained_model_name_or_path'], torch_dtype=torch.bfloat16,
            max_rank=config['rank'], min_rank=config['min_rank'],
        ).to('cuda')
    else:
        pipe = FluxPipeline.from_pretrained(config['pretrained_model_name_or_path'], torch_dtype=torch.bfloat16).to('cuda')

    set_flux_transformer_attn_processor(
        pipe.transformer,
        set_attn_proc_func=lambda name, dh, nh, ap: FluxLoraAttnProcessor(
            hidden_size=pipe.transformer.inner_dim, rank=config['rank'], 
            lora_linear_layer=LoRALinearLayer,
            do_training=False, 
        ),
    )
    lora_layers = AttnProcsLayers(pipe.transformer.attn_processors)
    lora_layers.load_state_dict(
        torch.load(
            os.path.join(config["output_dir"], f"checkpoint-{args.checkpoint_idx}", 'lora.pt') 
        )
    )
    lora_layers.to(device='cuda', dtype=torch.float32)

    prompt_set = args.prompts.split('#')
    generate_with_prompt_list(pipe, config, args, prompt_set, num_images_per_prompt=5, batch_size=1)


if __name__ == "__main__":
    args = parse_args()
    main(args)
