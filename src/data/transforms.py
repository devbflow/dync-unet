import numbers
import random

import albumentations as A
import numpy as np
import torchvision.transforms.v2.functional as TF
from torchvision.transforms.v2 import InterpolationMode


class ToTensor:
    """OpenEarthMap-RGB custom to_tensor"""

    def __init__(self, classes):
        self.classes = classes

    def __call__(self, sample):
        msks = [(sample["mask"] == v) * 1 for v in self.classes]
        sample["mask"] = TF.to_dtype(TF.to_image(np.stack(msks, axis=-1)), scale=True)
        sample["image"] = TF.to_dtype(TF.to_image(sample["image"]), scale=True)
        return sample


class Rotate:
    """OpenEarthMap-RGB custom rotate"""

    def __init__(self, degrees=(-180, 180)):
        self.degrees = degrees

    def __call__(self, sample):
        angle = random.uniform(*self.degrees)

        img = TF.rotate(sample["image"], angle, InterpolationMode.BICUBIC)
        msk = TF.rotate(sample["mask"], angle, InterpolationMode.NEAREST)
        return {"image": img, "mask": msk}


class Crop:
    """OpenEarthMap-RGB custom crop"""

    def __init__(self, size):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size

    def __call__(self, sample):
        h, w = sample["mask"].size

        if h > self.size[0] and w > self.size[1]:
            i = random.randrange(0, h - self.size[0])
            j = random.randrange(0, w - self.size[1])
            img = TF.crop(sample["image"], i, j, *self.size)
            msk = TF.crop(sample["mask"], i, j, *self.size)
        else:
            img = TF.resize(sample["image"], self.size, InterpolationMode.BICUBIC)
            msk = TF.resize(sample["mask"], self.size, InterpolationMode.NEAREST)

        return {"image": img, "mask": msk}


class Resize:
    """OpenEarthMap-RGB custom resize"""

    def __init__(self, size):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size

    def __call__(self, sample):
        img = TF.resize(sample["image"], self.size, InterpolationMode.BICUBIC)
        msk = TF.resize(sample["mask"], self.size, InterpolationMode.NEAREST)

        return {"image": img, "mask": msk}


def train_augm3(sample: dict, size: int = 512):
    """OpenEarthMap-SAR train augmentations"""
    augms = [
        A.PadIfNeeded(size, size, border_mode=0, fill=0, p=1.0),
        A.RandomCrop(size, size, p=1.0),
    ]
    return A.Compose(augms)(image=sample["image"], mask=sample["mask"])


def valid_augm(sample: dict, size: int = 512):
    """OpenEarthMap-SAR validation augmentations"""
    augms = [A.Resize(height=size, width=size, p=1.0)]
    return A.Compose(augms)(image=sample["image"], mask=sample["mask"])
