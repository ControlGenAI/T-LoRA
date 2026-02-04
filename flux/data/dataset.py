import random
import itertools
from PIL import Image
from pathlib import Path

import torch
from torch.utils.data import Dataset
import torchvision.transforms as T

from .transforms import aug


class DreamBoothDataset(Dataset):
    def __init__(
        self,
        instance_prompt,
        instance_data_root,
        resolution=1024,
        repeats=1,
        rotate=False,
        custom_instance_prompts=False,
        device='cuda',
        one_image=None,
        mask_dir=None,
    ):
        self.rotate = rotate
        self.custom_instance_prompts = custom_instance_prompts
        self.resolution = resolution

        self.instance_data_root = Path(instance_data_root)
        self.mask_dir = mask_dir

        if not self.instance_data_root.exists():
            raise ValueError("Instance images root doesn't exists.")

        names = [
            str(path) for path in list(Path(instance_data_root).iterdir()) if str(path).endswith('png') or str(path).endswith('jpg') or str(path).endswith('jpeg')
        ]
        if one_image:
            names = [n for n in names if str(n).endswith(one_image)]

        self.instance_prompt = instance_prompt
        self.image_names = []
        for i, img in enumerate(names):
            self.image_names.extend(itertools.repeat(names[i], repeats))

        self.num_instance_images = len(self.image_names)
        self._length = self.num_instance_images

    def __len__(self):
        return self._length

    def __getitem__(self, index):
        example = {}

        image_name = self.image_names[index % self.num_instance_images]

        image, mask = aug(image_name, self.mask_dir, self.resolution)
        image = T.Normalize([0.5], [0.5])(image)

        if self.mask_dir:
            example["mask"] = mask

        example["instance_images"] = image
        example["instance_prompt"] = self.instance_prompt
        return example


def collate_fn(examples):
    pixel_values = [example["instance_images"] for example in examples]
    prompts = [example["instance_prompt"] for example in examples]

    pixel_values = torch.stack(pixel_values)
    pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()

    batch = {"pixel_values": pixel_values, "prompts": prompts}

    if "mask" in examples[0]:
        masks = [example["mask"] for example in examples]
        masks = torch.stack(masks)
        batch["mask"] = masks
        
    return batch
