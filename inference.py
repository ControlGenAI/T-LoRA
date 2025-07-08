import argparse
import yaml

from tlora.inferencer_sdxl import inferencers

import warnings
warnings.filterwarnings("ignore")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str, required=True, help="Path to hparams.yml")
    parser.add_argument("--checkpoint_idx", type=str, default=None, required=False)
    parser.add_argument("--prompts", type=str, default='a photo of {0}', required=True)
    parser.add_argument("--num_images_per_prompt", type=int, default=5, help="Number of generated images for each prompt")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_inference_steps", type=int, default=25)
    parser.add_argument("--guidance_scale", type=float, default=5.0)
    parser.add_argument("--replace_inference_output", action="store_true", default=False)
    parser.add_argument("--version", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main(args):
    prompts = args.prompts.split('#')

    with open(args.config_path, "r", encoding="utf-8") as config_file:
        config = yaml.safe_load(config_file)
    inferencer = inferencers[config["trainer_class"]](config, args, prompts)

    inferencer.setup()
    inferencer.generate()


if __name__ == "__main__":
    args = parse_args()
    main(args)
