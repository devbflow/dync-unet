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
        sample["mask"] = TF.to_dtype(TF.to_image(np.stack(msks, axis=-1)), scale=False)
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


class ToTensorOEMSAR:
    def __init__(self, classes):
        self.classes = classes

    def __call__(self, sample):
        msks = [(sample["mask"] == v) for v in self.classes]
        msk = np.stack(msks, axis=-1)
        background = 1 - msk.sum(axis=-1, keepdims=True)
        sample["mask"] = TF.to_dtype(
            TF.to_image(np.concatenate((background, msk), axis=-1)), scale=False
        )

        for key in [k for k in sample if k != "mask"]:
            sample[key] = TF.to_dtype(TF.to_image(sample[key]), scale=True)
        return sample


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


def whu_train_augm(sample: dict, target_size: int = 512) -> dict:
    """Training augmentation for WHU-Opt-SAR.

    Pipeline: pad -> random crop -> random rotate (-180° to +180°).

    Args:
        sample: dict with keys "image" (numpy array) and "mask".
        target_size: Target patch size.

    Returns:
        Augmented image/mask pair.
    """
    img = sample["image"]
    msk = sample["mask"]

    # h, w = msk.shape[:2]

    # Pad to target_size if smaller
    # pad_h = max(0, target_size - h)
    # pad_w = max(0, target_size - w)

    # padded_img = TF.pad(img, (pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2))
    # padded_msk = TF.pad(msk, (pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2))

    # Rotation (-180° to +180°)
    angle = random.uniform(-180.0, 180.0)
    img = TF.rotate(img, angle, InterpolationMode.BICUBIC)
    msk = TF.rotate(msk, angle, InterpolationMode.NEAREST)

    # Random crop to target_size
    i = random.randint(0, max(1, img.size[1] - target_size))
    j = random.randint(0, max(1, img.size[0] - target_size))

    img = TF.crop(img, i, j, target_size, target_size)
    msk = TF.crop(msk, i, j, target_size, target_size)

    sample["image"] = img
    sample["mask"] = msk

    return sample


def whu_valid_augm(sample: dict, target_size: int = 512) -> dict:
    """Validation augmentation for WHU-Opt-SAR.

    Pipeline: pad -> uniform random crop (no rotation).

    Args:
        sample: dict with keys "image" (numpy array) and "mask".
        target_size: Target patch size.

    Returns:
        Cropped image/mask pair.
    """
    img = sample["image"]
    msk = sample["mask"]

    # h, w = msk.shape[:2]

    # Pad if smaller than target_size
    # pad_h = max(0, target_size - h)
    # pad_w = max(0, target_size - w)

    # padded_img = TF.pad(img, (pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2))
    # padded_msk = TF.pad(msk, (pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2))

    # Random crop to target_size
    i = random.randint(0, max(1, img.size[1] - target_size))
    j = random.randint(0, max(1, img.size[0] - target_size))

    cropped_img = TF.crop(img, i, j, target_size, target_size)
    cropped_msk = TF.crop(msk, i, j, target_size, target_size)

    sample["image"] = cropped_img
    sample["mask"] = cropped_msk

    return sample
