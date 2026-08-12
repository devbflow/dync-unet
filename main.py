from lightning.pytorch.cli import LightningCLI

from src.model.lightning_model import UNetLightningModule


def main():
    cli = LightningCLI(UNetLightningModule)


if __name__ == "__main__":
    main()
