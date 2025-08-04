import os
from pathlib import Path

import lightning as pl
import numpy as np
import torchvision
from torch.utils.data import DataLoader

from .dataset import OpenEarthMapRGBDataset
from .transforms import Crop, Resize, Rotate


class OpenEarthMapRGBDataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_dir: str,
        batch_size: int = 32,
        img_size: int = 512,
        num_workers: int = 4,
    ):
        super().__init__()

        self.save_hyperparameters()

        self.data_dir = os.path.join(data_dir, "OpenEarthMap-RGB")
        self.batch_size = batch_size
        self.img_size = img_size
        self.num_workers = num_workers

        self.train_split = os.path.join(self.data_dir, "train.txt")
        self.val_split = os.path.join(self.data_dir, "val.txt")
        self.test_split = os.path.join(self.data_dir, "test.txt")

        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

    def setup(self, stage: str):
        filenames = [
            f for f in Path(self.data_dir).rglob("*.tif") if "/images/" in str(f)
        ]

        if stage == "fit":
            train_filenames = [
                str(f)
                for f in filenames
                if f.name in np.loadtxt(self.train_split, dtype=str)
            ]
            train_augm = torchvision.transforms.Compose([Rotate(), Crop(self.img_size)])
            self.train_dataset = OpenEarthMapRGBDataset(
                train_filenames, augm=train_augm
            )

            print(
                f"[{self.__class__.__name__}] setup(fit) called. Train size: {len(self.train_dataset)}"
            )

        if stage == "validate" or stage == "fit":
            val_filenames = [
                str(f)
                for f in filenames
                if f.name in np.loadtxt(self.val_split, dtype=str)
            ]
            val_augm = torchvision.transforms.Compose([Resize(self.img_size)])
            self.val_dataset = OpenEarthMapRGBDataset(val_filenames, augm=val_augm)

            print(
                f"[{self.__class__.__name__}] setup(validate) called. Val size: {len(self.val_dataset)}"
            )

        if stage == "test":
            test_filenames = [
                str(f)
                for f in filenames
                if f.name in np.loadtxt(self.test_split, dtype=str)
            ]
            self.test_dataset = OpenEarthMapRGBDataset(test_filenames, testing=True)

            print(
                f"[{self.__class__.__name__}] setup(test) called. Test size: {len(self.test_dataset)}"
            )

    def train_dataloader(self):
        if self.train_dataset is None:
            raise RuntimeError("train_dataset is not set. Call setup('fit') first.")
        print(f"[{self.__class__.__name__}] train_dataloader called.")
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            drop_last=True,
            pin_memory=True,
        )

    def val_dataloader(self):
        if self.val_dataset is None:
            raise RuntimeError("val_dataset is not set. Call setup('validate') first.")
        print(f"[{self.__class__.__name__}] val_dataloader called.")
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def test_dataloader(self):
        if self.test_dataset is None:
            raise RuntimeError("test_dataset is not set. Call setup('test') first.")
        print(f"[{self.__class__.__name__}] test_dataloader called.")
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
        )
