# T-LoRA: Single Image Diffusion Model Customization Without Overfitting

<a href="https://arxiv.org/abs/2507.05964"><img src="https://img.shields.io/badge/arXiv-2502.06606-b31b1b.svg" height=22.5></a><!-- <a href="https://arxiv.org/abs/2502.06606"><img src="https://img.shields.io/badge/arXiv-2502.06606-b31b1b.svg" height=22.5></a> -->
[![License](https://img.shields.io/github/license/AIRI-Institute/al_toolbox)](./LICENSE)

>While diffusion model fine-tuning offers a powerful approach for customizing pre-trained models to generate specific objects, it frequently suffers from overfitting when training samples are limited, compromising both generalization capability and output diversity. This paper tackles the challenging yet most impactful task of adapting a diffusion model using just a single concept image, as single-image customization holds the greatest practical potential. We introduce T-LoRA, a Timestep-Dependent Low-Rank Adaptation framework specifically designed for diffusion model personalization. In our work we show that higher diffusion timesteps are more prone to overfitting than lower ones, necessitating a timestep-sensitive fine-tuning strategy. T-LoRA incorporates two key innovations: (1) a dynamic fine-tuning strategy that adjusts rank-constrained updates based on diffusion timesteps, and (2) a weight parametrization technique that ensures independence between adapter components through orthogonal initialization. Extensive experiments show that T-LoRA and its individual components outperform standard LoRA and other diffusion model personalization techniques. They achieve a superior balance between concept fidelity and text alignment, highlighting the potential of T-LoRA in data-limited and resource-constrained scenarios.
>


![image](docs/teaser.png)

## 📌 Updates

- [08/07/2025] 🔥🔥🔥 T-LoRA release

## 📌 Prerequisites

To run our method, please ensure you meet the following hardware and software requirements:
- Operating System: Linux
- NVIDIA GPU + CUDA CuDNN
- Conda 24.1.0+ or Python 3.11+

🤗 [Diffusers](https://github.com/huggingface/diffusers) library is used as the foundation for our method implementation.

## 📌 Setup

* Clone this repo:
```bash
git clone https://github.com/ControlGenAI/T-LoRA.git
cd T-LoRA
```

* Setup the environment. Conda environment `tlora` will be created and you can use it.
```bash
conda env create -f tlora_env.yml
conda activate tlora
```

## 📌 Training

You can launch T-LoRA training with the dog-example sourced from [DreamBooth Dataset](https://github.com/google/dreambooth):

```bash

export MODEL_NAME="stabilityai/stable-diffusion-xl-base-1.0"
export INSTANCE_DIR="dog_example"
export OUTPUT_DIR="trained-tlora_dog"
export API_KEY="your-wandb-api-key"

accelerate launch train.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --train_data_dir=$INSTANCE_DIR \
  --output_dir=$OUTPUT_DIR \
  --mixed_precision="no" \
  --trainer_type="ortho_lora" \  # choose "lora" in case you want to train Vanilla T-LoRA
  --trainer_class="sdxl_tlora" \
  --num_train_epochs=800 \
  --checkpointing_steps=100 \
  --resolution=1024 \
  --wandb_api_key=$API_KEY \  # remove if you prefer not to log during training 
  --validation_prompts="a {0} lying in the bed#a {0} swimming#a {0} dressed as a ballerina" \  # a string of prompts separated by #
  --num_val_imgs_per_prompt=3 \
  --placeholder_token="sks" \
  --class_name="dog" \
  --seed=0 \
  --lora_rank=64 \
  --min_rank=32 \
  --sig_type="last" \
  --one_image="02.jpg" \ # remove if you prefer full dataset training
```

We are utilizing the following flags in the command mentioned above

* `wandb_api_key=$API_KEY` will ensure the training runs are tracked on [Weights and Biases](https://wandb.ai/site). To use it, make sure to install Weights and Biases with `pip install wandb` and provide your API key as an argument.
* `validation_prompts` - this flag allows you to specify a string of validation prompts separated by #, enabling the script to perform several validation inference runs.
* use `trainer_type="ortho_lora"` if you prefer to train T-LoRA and `trainer_type="lora"` in case you want to train Vanilla T-LoRA. `sig_type="last"` flag ensures that training starts from weights initialized with the last singular values of the matrices.
* `one_image="00.jpg"`- this flag initiates training using a single selected image. Remove this flag if you prefer to train on the full dataset.

The optimal number of training steps may vary depending on the specific concept. However, based on our experiments, 800 iterations generally yield good results for many concepts.
We conducted our experiments using a single GPU setup on the Nvidia A100. Training each individual model is expected to take approximately 30 minutes.


After the training you will obtain the experiment folder in the following structure:

```
trained-tlora_dog
└───*****-****-dog_example_tortho_lora64
    └───checkpoint-100
        └───pytorch_lora_weights.safetensors
    └───checkpoint-200
    │   ...
    │
    └───logs
        └───hparams.yml

```
## 📌 Inference

```bash

export CONFIG_PATH="trained-tlora_dog/*****-****-dog_example_tortho_lora64/logs/hparams.yml"

python inference.py \
  --config_path=$CONFIG_PATH \
  --checkpoint_idx=800 \
  --guidance_scale=5.0 \
  --num_inference_steps=25 \
  --prompts="a {0} riding a bike#a {0} dressed as a ballerina#a {0} dressed in a superhero cape, soaring through the skies above a bustling city during a sunset" \  # a string of prompts separated by #
  --num_images_per_prompt=5 \
  --version=0 \
  --seed=0 \
```

This command will generate images in the corresponding checkpoint folder.

## 🙏 Acknowledgements
We sincerely thank the 🤗 [Huggingface](https://huggingface.co) community for their open-source code and contributions.

## 📌 Citation

If our work assists your research, feel free to give us a star ⭐ and cite us using:
```
@misc{soboleva2025tlorasingleimagediffusion,
      title={T-LoRA: Single Image Diffusion Model Customization Without Overfitting}, 
      author={Vera Soboleva and Aibek Alanov and Andrey Kuznetsov and Konstantin Sobolev},
      year={2025},
      eprint={2507.05964},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2507.05964}, 
}
```
