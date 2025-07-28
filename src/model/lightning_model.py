import lightning as pl
import torch.nn as nn
import torch.optim as optim

from unet_model import UNet


class UNetLightningModule(pl.LightningModule):
    def __init__(
        self, conv_type: str, in_channels: int, num_classes: int, lr: float = 1e-3
    ):
        super().__init__()

        self.save_hyperparameters()

        self.model = UNet(
            in_channels=in_channels,
            num_classes=num_classes,
            initial_planes=64,
            num_layers=4,
            bilinear=True,
            conv_type=conv_type,
            K=None,
        )

        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        imgs, masks, _ = batch
        y_hat = self.forward(imgs)
        loss = self.loss_fn(y_hat, masks)

        self.log("train_loss", loss, on_step=True, on_epoch=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        imgs, masks, _ = batch
        y_hat = self.forward(imgs)
        loss = self.loss_fn(y_hat, masks)

        self.log("val_loss", loss, on_step=False, on_epoch=True, logger=True)
        return loss

    def test_step(self, batch, batch_idx):
        imgs, masks, _ = batch
        y_hat = self.forward(imgs)
        loss = self.loss_fn(y_hat, masks)

        self.log("test_loss", loss, on_step=False, on_epoch=True, logger=True)
        return loss

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.hparams.lr)
        return optimizer
