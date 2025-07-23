import torch.nn as nn
import torch.optim as optim
import lightning as pl

from unet_model import UNet


class UNetLightningModule(pl.LightningModule):
    def __init__(self, learning_rate: float = 1e-3):
        super().__init__()

        self.save_hyperparameters()

        self.model = UNet(
            in_channels=...,
            num_classes=...,
            initial_planes=64,
            num_layers=4,
            bilinear=True,
            conv_type="static",
            K=None,
        )

        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, x):
        """
        Defines the forward pass of the model.
        This is typically what you'd call when doing inference.
        """
        return self.model(x)

    def training_step(self, batch, batch_idx):
        """
        Defines the logic for a single training step.
        It receives a batch of data and the batch index.
        """
        x, y = batch
        y_hat = self.forward(x)
        loss = self.loss_fn(y_hat, y)

        self.log(
            "train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True
        )
        return loss

    def validation_step(self, batch, batch_idx):
        """
        Defines the logic for a single validation step.
        It receives a batch of data and the batch index.
        """
        x, y = batch
        y_hat = self.forward(x)
        loss = self.loss_fn(y_hat, y)

        self.log(
            "val_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True
        )
        return loss

    def test_step(self, batch, batch_idx):
        """
        Defines the logic for a single test step.
        It receives a batch of data and the batch index.
        """
        x, y = batch
        y_hat = self.forward(x)
        loss = self.loss_fn(y_hat, y)

        self.log(
            "test_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True
        )
        return loss

    def configure_optimizers(self):
        """
        Defines the optimizer(s) and learning rate scheduler(s).
        """
        optimizer = optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
        # You can also add a learning rate scheduler here, e.g.:
        # scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.1)
        # return [optimizer], [scheduler]
        return optimizer
