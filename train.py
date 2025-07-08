import argparse
import sys

from tlora.trainer_sdxl import trainers

import warnings
warnings.filterwarnings("ignore")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Simple example of a inference script."
    )
    parser.add_argument("--trainer_type", type=str, required=True)
    parser.add_argument("--trainer_class", type=str, required=True)
    parser.add_argument("--project_name", type=str, default="tlora")
    parser.add_argument("--wandb_api_key", type=str, required=False, default=None)
    parser.add_argument("--seed", type=int, default=0)

    parser.add_argument("--pretrained_model_name_or_path", type=str, default="stabilityai/stable-diffusion-xl-base-1.0")
    parser.add_argument("--mixed_precision", type=str, default="no", choices=["no", "fp16", "bf16"])
    parser.add_argument("--revision", type=str, default=None)

    parser.add_argument("--num_train_epochs", type=int, default=1000)
    parser.add_argument("--checkpointing_steps", type=int, default=200)

    parser.add_argument("--train_data_dir", type=str, default=None, required=True)
    parser.add_argument("--train_batch_size", type=int, default=1)
    parser.add_argument("--dataloader_num_workers", type=int, default=1)
    parser.add_argument("--resolution", type=int, default=1024)
    parser.add_argument("--output_dir", type=str, required=True)

    parser.add_argument("--class_data_dir", type=str, default=None, required=False, help="A folder containing the training data of class images.")
    parser.add_argument("--prior_loss_weight", type=float, default=1.0, help="The weight of prior preservation loss.")
    parser.add_argument("--with_prior_preservation", default=False, action="store_true", help="Flag to add prior preservation loss.")

    parser.add_argument("--class_name", type=str, required=True)
    parser.add_argument("--placeholder_token", type=str, required=True)
    parser.add_argument("--validation_prompts", type=str, default=None)
    parser.add_argument("--num_val_imgs_per_prompt", type=int, default=5)

    parser.add_argument("--lora_rank", type=int, default=4)
    parser.add_argument("--min_rank", type=int, default=1)
    parser.add_argument("--sig_type", type=str, required=False, default="last", choices=["principal", "last", "middle"])
    parser.add_argument("--alpha_rank_scale", type=float, default=1.0)

    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Initial learning rate (after the potential warmup period) to use.")
    parser.add_argument("--adam_beta1", type=float, default=0.9, help="The beta1 parameter for the Adam optimizer.")
    parser.add_argument("--adam_beta2", type=float, default=0.999, help="The beta2 parameter for the Adam optimizer.")
    parser.add_argument("--adam_weight_decay", type=float, default=1e-04, help="Weight decay to use.")
    parser.add_argument("--adam_epsilon", type=float, default=1e-08, help="Epsilon value for the Adam optimizer")

    parser.add_argument("--one_image", type=str, default=None)

    args = parser.parse_args()
    args.argv = [sys.executable] + sys.argv

    return args


def main(args):
    trainer = trainers[args.trainer_class](args)
    trainer.setup()
    trainer.train()


if __name__ == "__main__":
    args = parse_args()
    main(args)
