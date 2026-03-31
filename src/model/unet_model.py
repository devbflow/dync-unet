import torch
import torch.nn as nn
import torch.nn.functional as F

from .dynconv import AttDynConv, WPDynConv, WMDynConv


def get_conv_and_kwargs(
    conv_type: str = "static",
    K: int | None = None,
    downsample_size: int | None = None,
    kernel_gen_hidden: int | None = None,
    demodulate: bool = True,
) -> tuple[type[nn.Module], dict]:
    """Helper function to get correct convolution type and respective necessary kwargs."""

    if conv_type == "static":
        return nn.Conv2d, {"kernel_size": 3, "stride": 1, "padding": 1, "bias": False}
    elif conv_type == "att-dynamic":
        if K is None:
            raise ValueError(
                "For attention dynamic convolutions, number of kernels 'K' must be provided."
            )
        return AttDynConv, {"kernel_size": 3, "stride": 1, "padding": 1, "K": K}
    elif conv_type == "wp-dynamic":
        if downsample_size is None or kernel_gen_hidden is None:
            raise ValueError(
                "For weight prediction dynamic convolutions, 'downsample_size' and 'kernel_gen_hidden' must be provided."
            )
        return WPDynConv, {
            "kernel_size": 3,
            "downsample_size": downsample_size,
            "kernel_gen_hidden": kernel_gen_hidden,
        }
    elif conv_type == "wm-dynamic":
        if downsample_size is None or kernel_gen_hidden is None:
            raise ValueError(
                "For weight modulation dynamic convolutions, 'downsample_size' and 'kernel_gen_hidden' must be provided."
            )
        return WMDynConv, {
            "kernel_size": 3,
            "downsample_size": downsample_size,
            "kernel_gen_hidden": kernel_gen_hidden,
            "demodulate": demodulate,
        }
    else:
        raise ValueError(
            f"Unsupported convolution type: '{conv_type}'. Supported types: ['static', 'attention-dynamic', 'input-dynamic', 'wm-dynamic']."
        )


class DoubleConv(nn.Module):
    """(Conv2d -> BN -> ReLU) * 2.
    Conv2d can be static or dynamic convolution, depending on given conv_type.
    """

    def __init__(
        self,
        conv_type: type[nn.Module],
        in_channels: int,
        out_channels: int,
        conv_kwargs: dict,
    ):
        super().__init__()

        self.double_conv = nn.Sequential(
            conv_type(in_channels, out_channels, **conv_kwargs),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            conv_type(out_channels, out_channels, **conv_kwargs),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor):
        return self.double_conv(x)


class DownBlock(nn.Module):
    """MaxPool2d followed by DoubleConv"""

    def __init__(
        self,
        conv_type: type[nn.Module],
        in_channels: int,
        out_channels: int,
        conv_kwargs: dict,
    ):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(conv_type, in_channels, out_channels, conv_kwargs),
        )

    def forward(self, x: torch.Tensor):
        return self.maxpool_conv(x)


class UpBlock(nn.Module):
    """Upsampling followed by DoubleConv"""

    def __init__(
        self,
        conv_type: type[nn.Module],
        in_channels: int,
        out_channels: int,
        conv_kwargs: dict,
        upsample: bool = True,
    ):
        super().__init__()
        if upsample:
            self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(
                in_channels // 2, in_channels // 2, kernel_size=2, stride=2
            )

        self.conv = DoubleConv(conv_type, in_channels, out_channels, conv_kwargs)

    def forward(self, x: torch.Tensor, x_residual: torch.Tensor):
        x = self.up(x)

        # Pad x if its spatial dimensions are smaller than residual due to pooling/upconv
        diffY = x_residual.size()[2] - x.size()[2]
        diffX = x_residual.size()[3] - x.size()[3]

        x = F.pad(x, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])

        x = torch.cat([x_residual, x], dim=1)
        return self.conv(x)


class UNet(nn.Module):
    """UNet implementation supporting multiple types of convolutional layers.
    Supported are:
        - 'static' == standard convolution (torch.nn.Conv2d)
        - 'dynamic' == dynamic convolution (DynConv)
    """

    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        initial_planes: int = 64,
        num_layers: int = 4,
        upsample: bool = True,
        conv_type: str = "static",
        K: int | None = None,
        downsample_size: int | None = None,
        kernel_gen_hidden: int | None = None,
        demodulate: bool = True,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.num_layers = num_layers
        self.upsample = upsample
        self.conv, self.conv_kwargs = get_conv_and_kwargs(
            conv_type, K, downsample_size, kernel_gen_hidden, demodulate
        )

        self.inc = DoubleConv(self.conv, in_channels, initial_planes, self.conv_kwargs)

        # Encoder
        self.enc_modules = nn.ModuleList(
            [
                DownBlock(
                    self.conv,
                    initial_planes * 2**i,
                    initial_planes * 2 ** (i + 1),
                    self.conv_kwargs,
                )
                for i in range(self.num_layers)
            ]
        )

        # Decoder
        self.dec_modules = nn.ModuleList(
            [
                UpBlock(
                    self.conv,
                    initial_planes * 2**j + initial_planes * 2 ** (j - 1),
                    initial_planes * 2 ** (j - 1),
                    self.conv_kwargs,
                    upsample,
                )
                for j in range(self.num_layers, 0, -1)
            ]
        )

        self.out = nn.Conv2d(initial_planes, num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor):
        x_residuals = []

        x = self.inc(x)

        # Encoder
        for i in range(len(self.enc_modules)):
            x_residuals.append(x)
            x = self.enc_modules[i](x)

        # Decoder
        for j in range(len(self.dec_modules)):
            x = self.dec_modules[j](x, x_residuals[-(j + 1)])

        logits = self.out(x)
        return logits
