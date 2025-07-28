from lightning.pytorch.cli import LightningCLI

from data.datamodule import OpenEarthMapRGBDataModule
from model.lightning_model import UNetLightningModule


def main():
    cli = LightningCLI(UNetLightningModule, OpenEarthMapRGBDataModule)


if __name__ == "__main__":
    main()
