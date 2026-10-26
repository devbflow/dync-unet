import math

import numpy as np
import rasterio
from PIL import Image
from torch.utils.data import Dataset

from . import transforms


def load_multiband(path: str):
    src = rasterio.open(path, "r")
    return (np.moveaxis(src.read(), 0, -1)).astype(np.uint8)


def load_grayscale(path: str):
    src = rasterio.open(path, "r")
    return (src.read(1)).astype(np.uint8)


class OpenEarthMapRGBDataset(Dataset):
    """
    OpenEarthMap RGB dataset

    Args:
        fn_list (str): List containing image names
        classes (int): list of of class-code
        augm (Classes): transformation pipeline (e.g. Rotate, Crop, etc.)
    """

    def __init__(self, img_list: list, n_classes: int = 9, testing=False, augm=None):
        self.fn_imgs = [str(f) for f in img_list]
        self.fn_msks = [f.replace("images", "labels") for f in self.fn_imgs]
        self.augm = augm
        self.testing = testing
        self.classes = np.arange(n_classes).tolist()
        self.to_tensor = transforms.ToTensor(classes=self.classes)

        self.load_multiband = load_multiband
        self.load_grayscale = load_grayscale

    def __getitem__(self, idx):
        img = Image.fromarray(self.load_multiband(self.fn_imgs[idx]))

        if not self.testing:
            msk = Image.fromarray(self.load_grayscale(self.fn_msks[idx]))
        else:
            msk = Image.fromarray(np.zeros(img.size[:2], dtype="uint8"))

        if self.augm is not None:
            data = self.augm({"image": img, "mask": msk})
        else:
            h, w = msk.size
            power_h = math.ceil(np.log2(h) / np.log2(2))
            power_w = math.ceil(np.log2(w) / np.log2(2))
            if 2**power_h != h or 2**power_w != w:
                img = img.resize((2**power_w, 2**power_h), resample=Image.BICUBIC)
                msk = msk.resize((2**power_w, 2**power_h), resample=Image.NEAREST)
            data = {"image": img, "mask": msk}

        data = self.to_tensor(
            {
                "image": np.array(data["image"], dtype="uint8"),
                "mask": np.array(data["mask"], dtype="uint8"),
            }
        )
        return data["image"], data["mask"], self.fn_imgs[idx]

    def __len__(self):
        return len(self.fn_imgs)


class OpenEarthMapSARDataset(Dataset):
    def __init__(self, label_list, data_type="rgb", size=512, train=False):
        self.fns = label_list
        assert data_type.lower() in ("rgb", "sar"), (
            f"'{data_type}' not supported, must be 'rgb' or 'sar'."
        )
        self.data_type = data_type.lower()
        self.augm = transforms.train_augm3 if train else transforms.valid_augm
        self.size = size
        self.train = train
        self.classes = list(range(1, 9))
        self.to_tensor = transforms.ToTensorOEMSAR(classes=self.classes)
        self.load_fn = load_multiband if self.data_type == "rgb" else load_grayscale
        self.load_grayscale = load_grayscale

    def __getitem__(self, idx):
        img = self.load_fn(self.fns[idx].replace("labels", f"{self.data_type}_images"))
        msk = self.load_grayscale(self.fns[idx])

        data = self.augm({"image": img, "mask": msk}, self.size)
        data = self.to_tensor(data)

        return data["image"], data["mask"], self.fns[idx]

    def __len__(self):
        return len(self.fns)


class WHUOptSARDataset(Dataset):
    """WHU-Opt-SAR dataset for RGB/optical and SAR image fusion.

    Args:
        data_dir: Base directory containing 'optical/', 'sar/', and 'lbl/' subfolders.
        data_type: Either "rgb" to use only 3 RGB channels from RGBA images, or
                   "rgb+nir" to use all 4 channels (RGB+NIR) or "sar" for 1 channel images. Default is "rgb".
        crop_size: Target patch size for training/validation patches (default 512).

    Notes:
        - Optical images are in RGBA format (R, G, B, NIR). When data_type="rgb",
          only the first 3 channels are used. When "rgb+nir", all 4 channels are used.
        - SAR images are single-band (grayscale).
    """

    def __init__(
        self,
        label_list,
        data_type: str = "rgb",
        crop_size: int = 512,
        train: bool = False,
    ):
        self.fns = label_list
        self.data_type = data_type.lower()
        self.data_type_dir = (
            "optical" if self.data_type in ("rgb", "rgb+nir") else "sar"
        )

        assert data_type in ("rgb", "rgb+nir", "sar"), (
            f"data_type must be 'rgb', 'sar' or 'rgb+nir', got '{data_type}'."
        )
        self.crop_size = crop_size
        self.load_fn = (
            load_multiband if self.data_type in ("rgb", "rgb+nir") else load_grayscale
        )
        self.load_grayscale = load_grayscale
        self.augm = transforms.whu_train_augm if train else transforms.whu_valid_augm
        self.to_tensor = transforms.ToTensor(classes=[0, 10, 20, 30, 40, 50, 60, 70])

    def __getitem__(self, idx):
        with rasterio.Env(CPL_LOG="/dev/null"):
            img = self.load_fn(self.fns[idx].replace("lbl", self.data_type_dir))
        msk = Image.fromarray(self.load_grayscale(self.fns[idx]))
        if self.data_type == "rgb":
            img = img[:, :, :3]  # Use only RGB channels, drop NIR channel

        img = Image.fromarray(img)

        data = self.augm({"image": img, "mask": msk}, target_size=self.crop_size)
        data = self.to_tensor(
            {
                "image": np.array(data["image"], dtype="uint8"),
                "mask": np.array(data["mask"], dtype="uint8"),
            }
        )
        return data["image"], data["mask"], self.fns[idx]

    def __len__(self):
        return len(self.fns)
