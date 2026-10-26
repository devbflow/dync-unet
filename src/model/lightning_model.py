import io

import lightning as pl
import numpy as np
import torch
import torchvision
from matplotlib import pyplot as plt
from PIL import Image
from torch import nn, optim
from torchmetrics.classification import MulticlassConfusionMatrix, MulticlassF1Score
from torchmetrics.segmentation import MeanIoU

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
        dyn_placement: str = "full",
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
            dyn_placement=dyn_placement,
            K=K,
            downsample_size=downsample_size,
            kernel_gen_hidden=kernel_gen_hidden,
            demodulate=demodulate,
            rank=rank,
            reduction=reduction,
        )

        self.loss_fn = nn.CrossEntropyLoss()
        self.f_score = MulticlassF1Score(num_classes=num_classes, average="macro")
        self.iou = MeanIoU(num_classes=num_classes, input_format="index")
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
        with torch.no_grad():
            pred_disc = pred.argmax(dim=1)
            masks_disc = masks.argmax(dim=1)
            iou = self.iou(pred_disc, masks_disc)
            fscore = self.f_score(pred_disc, masks_disc)
            self.train_conf_matrix.update(pred_disc, masks_disc)

        self.log_dict(
            {"train_loss": loss, "train_mIoU": iou, "train_fscore": fscore},
            on_step=True,
            on_epoch=True,
            logger=True,
            batch_size=input_imgs.shape[0],
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
                batch_size=input_imgs.shape[0],
            )
        return loss

    def on_train_end(self) -> None:
        self._log_confusion_matrix(mode="train")

    def validation_step(self, batch, batch_idx):
        input_imgs, masks, _ = batch
        pred = self.forward(input_imgs)
        pred_disc = pred.argmax(dim=1)

        masks = masks.to(torch.float32)
        masks_disc = masks.argmax(dim=1)

        loss = self.loss_fn(pred, masks)
        iou = self.iou(pred_disc, masks_disc)
        fscore = self.f_score(pred_disc, masks_disc)
        self.val_conf_matrix.update(pred_disc, masks_disc)

        self.log_dict(
            {"val_loss": loss, "val_mIoU": iou, "val_fscore": fscore},
            on_step=False,
            on_epoch=True,
            logger=True,
            batch_size=input_imgs.shape[0],
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
                batch_size=input_imgs.shape[0],
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
                batch_size=input_imgs.shape[0],
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
            batch_size=input_imgs.shape[0],
        )

    def _log_images(self, mode, images, step, batch_size: int):
        # images shape: (3 * B, H, W, C) as uint8 numpy array
        tensor = torch.from_numpy(images).permute(0, 3, 1, 2).float() / 255.0
        grid = torchvision.utils.make_grid(tensor, nrow=batch_size, normalize=False)
        self.logger.experiment.add_image(f"{mode}_image", grid, step)

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
