import lightning as pl
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from src.utils.metrics import FBetaScore, IoU
from src.utils.visualization import draw_segment_image

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
        log_img_every_n_epoch: int = 50,
    ):
        super().__init__()

        self.save_hyperparameters()
        assert conv_type in ("static", "dynamic"), (
            f"'{conv_type}' not supported. Should be 'static' or 'dynamic'."
        )

        self.model = UNet(
            in_channels=in_channels,
            num_classes=num_classes,
            initial_planes=initial_planes,
            num_layers=num_layers,
            upsample=True,
            conv_type=conv_type,
            K=K,
        )

        self.loss_fn = nn.CrossEntropyLoss()
        self.f_score = FBetaScore()
        self.iou = IoU()

    def forward(self, x) -> torch.Tensor:
        return self.model(x)

    def training_step(self, batch, batch_idx):
        input_imgs, masks, _ = batch
        y_hat = self.forward(input_imgs)

        loss = self.loss_fn(y_hat, masks.to(torch.float32))
        iou = self.iou(y_hat, masks.to(torch.float32))
        fscore = self.f_score(y_hat, masks.to(torch.float32))

        self.log_dict(
            {"train_loss": loss, "train_mIoU": iou, "train_fscore": fscore},
            on_step=True,
            on_epoch=True,
            logger=True,
        )

        if batch_idx == 0 and self.current_epoch % self.hparams.log_img_every_n_epoch == 0:
            y_hat = np.argmax(y_hat.detach().cpu().numpy(), axis=1, keepdims=True)
            segment_images = draw_segment_image(y_hat, "oem").squeeze()
            self._log_images(
                "train",
                self._stack_segment_with_input(segment=segment_images, input_images=input_imgs),
                self.current_epoch,
            )

        return loss

    def validation_step(self, batch, batch_idx):
        input_imgs, masks, _ = batch
        y_hat = self.forward(input_imgs)

        loss = self.loss_fn(y_hat, masks.to(torch.float32))
        iou = self.iou(y_hat, masks.to(torch.float32))
        fscore = self.f_score(y_hat, masks.to(torch.float32))

        self.log_dict(
            {"val_loss": loss, "val_mIoU": iou, "val_fscore": fscore},
            on_step=False,
            on_epoch=True,
            logger=True,
        )

        if batch_idx == 0 and self.current_epoch % self.hparams.log_img_every_n_epoch == 0:
            y_hat = np.argmax(y_hat.detach().cpu().numpy(), axis=1, keepdims=True)
            segment_images = draw_segment_image(y_hat, "oem").squeeze()
            self._log_images(
                "val",
                self._stack_segment_with_input(segment=segment_images, input_images=input_imgs),
                self.current_epoch,
            )

        return loss

    def test_step(self, batch, batch_idx):
        input_imgs, _, _ = batch
        with torch.no_grad():
            y_hat = self.forward(input_imgs)
        y_hat = np.argmax(y_hat.cpu().numpy(), axis=1, keepdims=True)

        if batch_idx % 50 == 0:
            segment_images = draw_segment_image(y_hat, "oem").squeeze()
            self._log_images(
                "test",
                self._stack_segment_with_input(segment=segment_images, input_images=input_imgs),
                batch_idx,
            )

    def predict_step(self, batch, batch_idx):
        input_imgs, _, _ = batch
        with torch.no_grad():
            y_hat = self.forward(input_imgs)
        y_hat = np.argmax(y_hat.cpu().numpy(), axis=1, keepdims=True)

        segment_images = draw_segment_image(y_hat, "oem").squeeze()
        self._log_images(
            "predict",
            self._stack_segment_with_input(segment=segment_images, input_images=input_imgs),
            batch_idx,
        )

    def _log_images(self, mode, images, step):
        if len(images.shape) == 4:
            self.logger.experiment.add_images(f"{mode}_image", images, step, dataformats="NHWC")
        else:
            self.logger.experiment.add_image(f"{mode}_image", images, step, dataformats="HWC")

    def _stack_segment_with_input(
        self, segment: np.ndarray, input_images: torch.Tensor
    ) -> np.ndarray:
        """Stack image tensor with segment image ndarray for visualization."""
        return np.concat(
            (
                segment,
                (input_images.detach().cpu().permute(0, 2, 3, 1).numpy() * 255).astype("uint8"),
            ),
            axis=0,
            dtype=np.uint8,
        )

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.hparams.lr)
        return optimizer
