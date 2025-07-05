import torch
import torch.nn as nn
import torch.nn.functional as F


def get_conv_and_kwargs(
    conv_type: str = "static", K: int = None
) -> tuple[nn.Module, dict]:
    """Helper function to get correct convolution type and respective necessary kwargs.

    Args:
        conv_type (str): (supported) type of convolution. 'static' or 'dynamic'. Defaults to 'static'.
        K (int, optional): number of kernels for dynamic conv. Defaults to None.

    Raises:
        ValueError: if conv_type is unsupported or required arguments are missing

    Returns:
        tuple[nn.Module, dict]: convolution Module and specific kwargs
    """
    if conv_type == "static":
        return nn.Conv2d, {}
    elif conv_type == "dynamic":
        if K is None:
            raise ValueError(
                "For dynamic convolutions, number of kernels 'K' must be provided."
            )
        return DynConv, {"K": K}
    else:
        raise ValueError(
            f"Unsupported convolution type: '{conv_type}'. Supported types: 'static' or 'dynamic'."
        )


class DynConv(nn.Module):
    """
    Dynamic Convolution layer as proposed in "Dynamic Convolution: Attention Over Convolution Kernels".
    It aggregates multiple parallel convolution kernels dynamically based on input-dependent attention.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
        bias: bool = False,
        K: int = 4,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.K = K

        # K sets of convolution kernels and biases
        self.weight = nn.Parameter(
            torch.randn(K, out_channels, in_channels, kernel_size, kernel_size)
        )
        if bias:
            self.bias = nn.Parameter(torch.randn(K, out_channels))
        else:
            self.register_parameter("bias", None)

        # Attention mechanism (similar to Squeeze-and-Excitation)
        self.attention_pooling = nn.AdaptiveAvgPool2d(1)
        self.attention_fc1 = nn.Linear(in_channels, in_channels // 4)
        self.attention_relu = nn.ReLU(inplace=True)
        self.attention_fc2 = nn.Linear(in_channels // 4, K)

    def forward(self, x: torch.Tensor):
        N, C, H, W = x.size()

        # Compute attention weights
        attention_input = self.attention_pooling(x).view(N, C)
        attention_weights = self.attention_fc2(
            self.attention_relu(self.attention_fc1(attention_input))
        )
        attention_weights = F.softmax(attention_weights, dim=1)

        # Aggregate kernels and biases
        aggregated_weight = torch.sum(
            attention_weights.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
            * self.weight,
            dim=1,
        )

        aggregated_bias = None
        if self.bias is not None:
            aggregated_bias = torch.sum(
                attention_weights.unsqueeze(-1) * self.bias, dim=1
            )

        # Reshape for grouped convolution to apply batch-specific kernels
        x = x.view(1, N * C, H, W)
        aggregated_weight = aggregated_weight.view(
            N * self.out_channels, self.in_channels, self.kernel_size, self.kernel_size
        )

        if aggregated_bias is not None:
            aggregated_bias = aggregated_bias.view(N * self.out_channels)

        out = F.conv2d(
            x, aggregated_weight, aggregated_bias, self.stride, self.padding, groups=N
        )
        out = out.view(N, self.out_channels, out.size(-2), out.size(-1))

        return out


class DoubleConv(nn.Module):
    """(Conv2d -> BN -> ReLU) * 2.
    Conv2d can be static or dynamic convolution, depending on given conv_type.
    """

    def __init__(
        self,
        conv_type: nn.Module,
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
        conv_type: nn.Module,
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
        conv_type: nn.Module,
        in_channels: int,
        out_channels: int,
        conv_kwargs: dict,
        bilinear: bool = True,
    ):
        super().__init__()
        if bilinear:
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
        bilinear: bool = True,
        conv_type: str = "static",
        K: int = None,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.num_layers = num_layers
        self.bilinear = bilinear
        self.conv, self.conv_kwargs = get_conv_and_kwargs(conv_type, K)

        self.inc = DoubleConv(self.conv, in_channels, initial_planes, self.conv_kwargs)

        # Encoder
        # FIXME: replace by following line later
        # self.enc_modules = nn.ModuleList([DownBlock(self.conv, initial_planes * 2 ** i, initial_planes * 2 ** (i+1), self.conv_kwargs) for i in range(self.num_layers)])
        self.enc_modules = nn.ModuleList(
            [
                DownBlock(
                    self.conv, initial_planes, initial_planes * 2, self.conv_kwargs
                ),
                DownBlock(
                    self.conv, initial_planes * 2, initial_planes * 4, self.conv_kwargs
                ),
                DownBlock(
                    self.conv, initial_planes * 4, initial_planes * 8, self.conv_kwargs
                ),
                DownBlock(
                    self.conv, initial_planes * 8, initial_planes * 16, self.conv_kwargs
                ),
            ]
        )

        # Decoder
        # FIXME: replace by following line later
        # self.dec_modules = nn.ModuleList([UpBlock(self.conv, initial_planes * 2 ** j + initial_planes * 2 ** (j - 1), initial_planes * 2 ** (j - 1), self.conv_kwargs, bilinear) for j in range(self.num_layers, 0, -1)])
        self.dec_modules = nn.ModuleList(
            [
                UpBlock(
                    self.conv,
                    initial_planes * 16 + initial_planes * 8,
                    initial_planes * 8,
                    self.conv_kwargs,
                    bilinear,
                ),
                UpBlock(
                    self.conv,
                    initial_planes * 8 + initial_planes * 4,
                    initial_planes * 4,
                    self.conv_kwargs,
                    bilinear,
                ),
                UpBlock(
                    self.conv,
                    initial_planes * 4 + initial_planes * 2,
                    initial_planes * 2,
                    self.conv_kwargs,
                    bilinear,
                ),
                UpBlock(
                    self.conv,
                    initial_planes * 2 + initial_planes,
                    initial_planes,
                    self.conv_kwargs,
                    bilinear,
                ),
            ]
        )

        self.out = nn.Conv2d(initial_planes, num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor):
        x_residuals = []

        x = self.inc(x)
        x_residuals.append(x)

        # Encoder
        for i in range(len(self.enc_modules)):
            x = self.enc_modules[i](x)
            x_residuals.append(x)

        # Decoder
        for j in range(len(self.dec_modules)):
            x = self.dec_modules[j](x, x_residuals[-(j + 1)])

        logits = self.out(x)
        return logits
