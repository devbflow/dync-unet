import os
from pathlib import Path

import lightning as pl
import numpy as np
import torchvision
from torch.utils.data import DataLoader

from .dataset import OpenEarthMapRGBDataset, OpenEarthMapSARDataset, WHUOptSARDataset
from .transforms import Crop, Resize, Rotate


class OpenEarthMapRGBDataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_dir: str,
        batch_size: int = 32,
        img_size: int = 512,
        num_workers: int = 4,
        **kwargs,
    ):
        super().__init__()

        self.save_hyperparameters()

        self.data_dir = data_dir
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

        if stage == "test" or stage == "predict":
            test_filenames = [
                str(f)
                for f in filenames
                if f.name in np.loadtxt(self.test_split, dtype=str)
            ]

            if stage == "predict":
                # when predicting, only do small batch of test files
                test_filenames = np.random.default_rng().choice(test_filenames, 32)
            test_augm = torchvision.transforms.Compose([Resize(self.img_size)])

            self.test_dataset = OpenEarthMapRGBDataset(
                test_filenames, augm=test_augm, testing=True
            )

            print(
                f"[{self.__class__.__name__}] setup(test/predict) called. Test size: {len(self.test_dataset)}"
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

    def predict_dataloader(self):
        if self.test_dataset is None:
            raise RuntimeError("test_dataset is not set. Call setup('predict') first.")
        print(f"[{self.__class__.__name__}] predict_dataloader called.")
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
        )


class OpenEarthMapSARDataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_dir: str,
        batch_size: int = 32,
        img_size: int = 512,
        num_workers: int = 4,
        data_type: str = "sar",
        **kwargs,
    ):
        super().__init__()

        self.save_hyperparameters()

        self.data_dir = data_dir
        self.batch_size = batch_size
        self.img_size = img_size
        self.num_workers = num_workers
        self.data_type = data_type

        self.train_split = os.path.join(self.data_dir, "train.txt")
        self.val_split = os.path.join(self.data_dir, "val.txt")
        self.labels_path = os.path.join(self.data_dir, "train", "labels")

    def setup(self, stage: str):
        filenames = os.listdir(self.labels_path)

        if stage == "fit":
            train_labels = [
                os.path.join(self.labels_path, name)
                for name in filenames
                if name in np.loadtxt(self.train_split, dtype=str)
            ]
            self.train_dataset = OpenEarthMapSARDataset(
                label_list=train_labels,
                data_type=self.data_type,
                size=self.img_size,
                train=True,
            )

            print(
                f"[{self.__class__.__name__}] setup(fit) called. Train size: {len(self.train_dataset)}"
            )

        if stage == "validate" or stage == "fit":
            val_labels = [
                os.path.join(self.labels_path, name)
                for name in filenames
                if name in np.loadtxt(self.val_split, dtype=str)
            ]
            self.val_dataset = OpenEarthMapSARDataset(
                label_list=val_labels,
                data_type=self.data_type,
                size=self.img_size,
                train=False,
            )

            print(
                f"[{self.__class__.__name__}] setup(validate) called. Val size: {len(self.val_dataset)}"
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
        raise NotImplementedError("Test dataloader not supported for OpenEarthMap-SAR")

    def predict_dataloader(self):
        raise NotImplementedError("Prediction not supported for OpenEarthMap-SAR")


class WHUOptSARDataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_dir: str,
        batch_size: int = 32,
        img_size: int = 512,
        num_workers: int = 4,
        data_type: str = "rgb",
        **kwargs,
    ):
        super().__init__()

        self.save_hyperparameters()

        self.data_dir = data_dir
        self.batch_size = batch_size
        self.img_size = img_size
        self.num_workers = num_workers
        self.data_type = data_type

        self.train_split = os.path.join(self.data_dir, "train_crops.txt")
        self.val_split = os.path.join(self.data_dir, "val_crops.txt")
        self.labels_path = os.path.join(self.data_dir, "lbl")

    def setup(self, stage: str):
        filenames = os.listdir(self.labels_path)

        if stage == "fit":
            train_labels = [
                os.path.join(self.labels_path, name)
                for name in filenames
                if name in np.loadtxt(self.train_split, dtype=str)
            ]
            self.train_dataset = WHUOptSARDataset(
                label_list=train_labels,
                data_type=self.data_type,
                crop_size=self.img_size,
                train=True,
            )

            print(
                f"[{self.__class__.__name__}] setup(fit) called. Train size: {len(self.train_dataset)}"
            )

        if stage == "validate" or stage == "fit":
            val_labels = [
                os.path.join(self.labels_path, name)
                for name in filenames
                if name in np.loadtxt(self.val_split, dtype=str)
            ]
            self.val_dataset = WHUOptSARDataset(
                label_list=val_labels,
                data_type=self.data_type,
                crop_size=self.img_size,
                train=False,
            )

            print(
                f"[{self.__class__.__name__}] setup(validate) called. Val size: {len(self.val_dataset)}"
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
        raise NotImplementedError("Test dataloader not supported for OpenEarthMap-SAR")

    def predict_dataloader(self):
        raise NotImplementedError("Prediction not supported for OpenEarthMap-SAR")
