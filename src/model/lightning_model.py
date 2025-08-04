import lightning as pl
import torch
import torch.nn as nn
import torch.optim as optim

from .unet_model import UNet


class UNetLightningModule(pl.LightningModule):
    def __init__(
        self,
        conv_type: str,
        in_channels: int,
        num_classes: int,
        num_layers: int = 4,
        initial_planes: int = 64,
        lr: float = 1e-3,
        K: int | None = None,
    ):
        super().__init__()

        self.save_hyperparameters()
        assert conv_type in ("static", "dynamic")

        self.model = UNet(
            in_channels=in_channels,
            num_classes=num_classes,
            initial_planes=initial_planes,
            num_layers=num_layers,
            bilinear=True,
            conv_type=conv_type,
            K=K,
        )

        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, x) -> torch.Tensor:
        return self.model(x)

    def training_step(self, batch, batch_idx):
        imgs, masks, _ = batch
        y_hat = self.forward(imgs)

        loss = self.loss_fn(y_hat, masks.to(torch.float32))

        self.log("train_loss", loss, on_step=True, on_epoch=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        imgs, masks, _ = batch
        y_hat = self.forward(imgs)
        loss = self.loss_fn(y_hat, masks.to(torch.float32))

        self.log("val_loss", loss, on_step=False, on_epoch=True, logger=True)
        return loss

    def test_step(self, batch, batch_idx):
        imgs, masks, _ = batch
        y_hat = self.forward(imgs)
        loss = self.loss_fn(y_hat, masks.to(torch.float32))

        self.log("test_loss", loss, on_step=False, on_epoch=True, logger=True)
        return loss

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.hparams.lr)
        return optimizer
