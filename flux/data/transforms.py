import os
from PIL import Image
from PIL.ImageOps import exif_transpose

import torch
import torchvision.transforms as T
import torch.nn.functional as F
from torchvision.transforms.functional import crop
from kornia.geometry.transform import get_affine_matrix2d, warp_affine


def aug(image_url, mask_dir, resolution):
    image = Image.open(image_url).convert('RGB')
    image = exif_transpose(image)
    image = T.Resize(resolution, interpolation=T.InterpolationMode.LANCZOS)(image)

    if mask_dir:
        mask = Image.open(os.path.join(mask_dir, image_url.split('/')[-1].split('.')[0] + '.png'))
        mask = T.Resize(resolution, interpolation=T.InterpolationMode.LANCZOS)(mask)

    train_crop = T.RandomCrop(resolution)
    y1, x1, h, w = train_crop.get_params(image, (resolution, resolution))
    image = crop(image, y1, x1, h, w)
    image = T.ToTensor()(image)

    if mask_dir:
        mask = crop(mask, y1, x1, h, w)
        mask = T.Resize(128, interpolation=T.InterpolationMode.LANCZOS)(mask)
        mask = T.ToTensor()(mask)
        return image, mask
    else:
        return image, mask_dir