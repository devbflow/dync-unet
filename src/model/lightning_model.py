import io

import lightning as pl
import numpy as np
import torch
import torchvision
from matplotlib import pyplot as plt
from PIL import Image
from torch import nn, optim
from torchmetrics.classification import MulticlassConfusionMatrix

from src.utils.metrics import FBetaScore, IoU
from src.utils.visualization import (
    draw_segment_image,
    get_num_classes,
    map_classes_to_labels,
)

from .unet_model import UNet


class UNetLightningModule(pl.LightningModule):
    def __init__(
        self,
        conv_type: str,
        in_channels: int,
        dataset: str,
        num_layers: int = 4,
        initial_planes: int = 64,
        lr: float = 1e-3,
        K: int | None = None,
        downsample_size: int | None = None,
        kernel_gen_hidden: int | None = None,
        demodulate: bool = True,
        reduction: int | None = None,
        rank: int | None = None,
        log_img_every_n_epoch: int = 50,
        debug: bool = False,
    ):
        super().__init__()

        self.save_hyperparameters()
        assert conv_type in (
            "static",
            "att-dynamic",
            "wp-dynamic",
            "wm-dynamic",
            "lora-dynamic",
            "depth-dynamic",
        ), (
            f"'{conv_type}' not supported. Should be 'static' or 'att/wp/wm/lora/depth-dynamic'."
        )
        # torch.set_float32_matmul_precision("medium")

        num_classes = get_num_classes(dataset)
        self.model = UNet(
            in_channels=in_channels,
            num_classes=num_classes,
            initial_planes=initial_planes,
            num_layers=num_layers,
            upsample=True,
            conv_type=conv_type,
            K=K,
            downsample_size=downsample_size,
            kernel_gen_hidden=kernel_gen_hidden,
            demodulate=demodulate,
            rank=rank,
            reduction=reduction,
        )

        self.loss_fn = nn.CrossEntropyLoss()
        self.f_score = FBetaScore()
        self.iou = IoU()
        self.train_conf_matrix = MulticlassConfusionMatrix(
            num_classes, normalize="true"
        )
        self.val_conf_matrix = MulticlassConfusionMatrix(num_classes, normalize="true")

    def forward(self, x) -> torch.Tensor:
        return self.model(x)

    def training_step(self, batch, batch_idx):
        input_imgs, masks, _ = batch
        pred = self.forward(input_imgs)

        masks = masks.to(torch.float32)
        loss = self.loss_fn(pred, masks)
        # print(f"LOSS: {loss}")
        # print(f"MASKS Shape/Min/Max: {masks.shape} / {masks.min()} / {masks.max()}")
        # for c in range(masks.shape[1]):
        #     print(f"CLASS {c}:")
        #     print("MIN:", masks[0, c, :, :].min(), "MAX:", masks[0, c, :, :].max())
        # print(f"INPUT Shape/Min/Max: {input_imgs.shape} / {input_imgs.min()} /{input_imgs.max()}")
        # print(f"PRED Shape/Min/Max: {pred.shape} / {pred.min()} / {pred.max()}")
        with torch.no_grad():
            iou = self.iou(pred, masks)
            fscore = self.f_score(pred, masks)
            self.train_conf_matrix.update(pred.argmax(dim=1), masks.argmax(dim=1))

        self.log_dict(
            {"train_loss": loss, "train_mIoU": iou, "train_fscore": fscore},
            on_step=True,
            on_epoch=True,
            logger=True,
        )

        if (
            batch_idx == 0
            and self.current_epoch % self.hparams.log_img_every_n_epoch == 0
        ):
            pred = np.argmax(pred.detach().cpu().numpy(), axis=1, keepdims=True)
            segment_pred = draw_segment_image(pred, self.hparams.dataset).squeeze()
            segment_target = draw_segment_image(
                np.argmax(masks.detach().cpu().numpy(), axis=1, keepdims=True), "oem"
            ).squeeze()
            self._log_images(
                "train",
                self._stack_segment_with_input(
                    segment_pred=segment_pred,
                    input_images=input_imgs,
                    segment_target=segment_target,
                ),
                self.current_epoch,
            )
        return loss

    def on_train_end(self) -> None:
        self._log_confusion_matrix(mode="train")

    def validation_step(self, batch, batch_idx):
        input_imgs, masks, _ = batch
        pred = self.forward(input_imgs)

        masks = masks.to(torch.float32)
        loss = self.loss_fn(pred, masks)
        iou = self.iou(pred, masks)
        fscore = self.f_score(pred, masks)
        self.val_conf_matrix.update(pred.argmax(dim=1), masks.argmax(dim=1))

        self.log_dict(
            {"val_loss": loss, "val_mIoU": iou, "val_fscore": fscore},
            on_step=False,
            on_epoch=True,
            logger=True,
        )

        if (
            batch_idx == 0
            and self.current_epoch % self.hparams.log_img_every_n_epoch == 0
        ):
            pred = np.argmax(pred.detach().cpu().numpy(), axis=1, keepdims=True)
            segment_pred = draw_segment_image(pred, self.hparams.dataset).squeeze()
            segment_target = draw_segment_image(
                np.argmax(masks.detach().cpu().numpy(), axis=1, keepdims=True),
                self.hparams.dataset,
            ).squeeze()
            self._log_images(
                "val",
                self._stack_segment_with_input(
                    segment_pred=segment_pred,
                    input_images=input_imgs,
                    segment_target=segment_target,
                ),
                self.current_epoch,
            )

        return loss

    def on_validation_end(self) -> None:
        self._log_confusion_matrix(mode="val")

    def test_step(self, batch, batch_idx):
        input_imgs, _, _ = batch
        with torch.no_grad():
            pred = self.forward(input_imgs)
        pred = np.argmax(pred.cpu().numpy(), axis=1, keepdims=True)

        if batch_idx % 50 == 0:
            segment_images = draw_segment_image(pred, self.hparams.dataset).squeeze()
            self._log_images(
                "test",
                self._stack_segment_with_input(
                    segment_pred=segment_images, input_images=input_imgs
                ),
                batch_idx,
            )

    def predict_step(self, batch, batch_idx):
        input_imgs, _, _ = batch
        with torch.no_grad():
            pred = self.forward(input_imgs)
        pred = np.argmax(pred.cpu().numpy(), axis=1, keepdims=True)

        segment_images = draw_segment_image(pred, self.hparams.dataset).squeeze()
        self._log_images(
            "predict",
            self._stack_segment_with_input(
                segment_pred=segment_images, input_images=input_imgs
            ),
            batch_idx,
        )

    def _log_images(self, mode, images, step):
        if len(images.shape) == 4:
            self.logger.experiment.add_images(
                f"{mode}_image", images, step, dataformats="NHWC"
            )
        else:
            self.logger.experiment.add_image(
                f"{mode}_image", images, step, dataformats="HWC"
            )

    def _stack_segment_with_input(
        self,
        segment_pred: np.ndarray,
        input_images: torch.Tensor,
        segment_target: np.ndarray | None = None,
    ) -> np.ndarray:
        """Stack image tensor with segment image ndarray for visualization."""
        arrays = [
            segment_pred,
            (input_images.detach().cpu().permute(0, 2, 3, 1).numpy() * 255).astype(
                "uint8"
            ),
        ]
        # grayscale images need to be converted to rgb
        if arrays[1].shape[-1] == 1:
            arrays[1] = arrays[1].repeat(3, axis=-1)

        if segment_target is not None:
            arrays.append(segment_target)

        return np.concatenate(arrays, axis=0, dtype=np.uint8)

    def _log_confusion_matrix(self, mode: str):
        """Log (train or val) confusion matrix.

        Args:
            mode (str): train or val, depending on mode
        """
        conf_matrix = (
            self.train_conf_matrix if mode == "train" else self.val_conf_matrix
        )

        fig, ax = plt.subplots(figsize=(12, 8))
        ax.set_title("(Normalized) Class Confusion Matrix")
        fig, ax = conf_matrix.plot(
            ax=ax, labels=map_classes_to_labels(self.hparams.dataset)
        )

        buf = io.BytesIO()
        fig.savefig(buf, format="png", bbox_inches="tight")
        plt.close(fig)
        buf.seek(0)
        im = torchvision.transforms.ToTensor()(Image.open(buf))

        self.logger.experiment.add_image(
            "train_conf_matrix" if mode == "train" else "val_conf_matrix",
            im,
            global_step=self.current_epoch,
        )

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.hparams.lr)
        return optimizer
