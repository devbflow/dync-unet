import torch
import torch.nn as nn
import torch.nn.functional as F


class AttDynConv(nn.Module):
    """Dynamic Convolution layer as proposed in "Dynamic Convolution: Attention Over Convolution Kernels".
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
        **kwargs,
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
            attention_weights.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1) * self.weight,
            dim=1,
        )

        aggregated_bias = None
        if self.bias is not None:
            aggregated_bias = torch.sum(attention_weights.unsqueeze(-1) * self.bias, dim=1)

        # Reshape for grouped convolution to apply batch-specific kernels
        x = x.view(1, N * C, H, W)
        aggregated_weight = aggregated_weight.view(
            N * self.out_channels, self.in_channels, self.kernel_size, self.kernel_size
        )

        if aggregated_bias is not None:
            aggregated_bias = aggregated_bias.view(N * self.out_channels)

        out = F.conv2d(x, aggregated_weight, aggregated_bias, self.stride, self.padding, groups=N)
        out = out.view(N, self.out_channels, out.size(-2), out.size(-1))

        return out


class InputDynConv(nn.Module):
    """A module for input-dependent dynamic kernel generation.
    Utilizes an MLP to generate Conv kernel weights based on input."""

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        kernel_gen_hidden=256,
        downsample_size=128,
        **kwargs,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.padding = kernel_size // 2  # "same"
        self.downsample_size = min(downsample_size, 128)

        self.num_kernel_params = out_channels * in_channels * kernel_size * kernel_size

        self.channel_reduce = nn.Conv2d(in_channels, 1, 1)
        self.kernel_generator = nn.Sequential(
            nn.Linear(self.downsample_size * self.downsample_size, kernel_gen_hidden),
            nn.ReLU(),
            nn.Linear(kernel_gen_hidden, self.num_kernel_params),
        )

    def forward(self, x: torch.Tensor):
        """Processes the input tensor x through the dynamic convolution using
        group convolution.

        Args:
            x (torch.Tensor): input tensor of shape (B, C_in, H, W).
        Returns:
            torch.Tensor: output tensor of shape (B, C_out, H, W).
        """
        N, C, H, W = x.size()

        downsampled = F.adaptive_avg_pool2d(x, self.downsample_size)
        downsampled = self.channel_reduce(downsampled)

        flattened_weights = self.kernel_generator(downsampled.flatten(start_dim=-2, end_dim=-1))
        dyn_conv_weights = flattened_weights.reshape(
            N * self.out_channels, C, self.kernel_size, self.kernel_size
        )

        grouped_input = x.view(1, N * C, H, W)

        output_grouped = F.conv2d(
            input=grouped_input,
            weight=dyn_conv_weights,
            bias=None,
            stride=1,
            padding=self.padding,
            groups=N,
        )

        output = output_grouped.view(N, self.out_channels, H, W)
        return output


class WMDynConv(nn.Module):
    """A module for input-dependent, weight modulated dynamic kernel generation, similar to StyleGAN2.
    Utilizes an MLP to modulate kernel over channel dimension based on input."""

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        kernel_gen_hidden=256,
        downsample_size=128,
        demodulate=True,
        eps=1e-8,
        **kwargs,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.padding = kernel_size // 2  # "same"
        self.downsample_size = min(downsample_size, 128)

        self.input_modulation = True
        self.demodulate = demodulate
        self.eps = eps

        self.in_channel_reduce = nn.Conv2d(in_channels, 1, 1)
        self.scale_generator = nn.Sequential(
            nn.Linear(self.downsample_size * self.downsample_size, kernel_gen_hidden),
            nn.ReLU(),
            nn.Linear(kernel_gen_hidden, in_channels if self.input_modulation else out_channels),
        )
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels, kernel_size, kernel_size))

    def forward(self, x: torch.Tensor):
        """Processes the input tensor x through the dynamic convolution using
        group convolution.

        Args:
            x (torch.Tensor): input tensor of shape (B, C_in, H, W).
        Returns:
            torch.Tensor: output tensor of shape (B, C_out, H, W).
        """
        N, C, H, W = x.size()

        downsampled = F.adaptive_avg_pool2d(x, self.downsample_size)
        downsampled = self.in_channel_reduce(downsampled)

        kernel_scale: torch.Tensor = self.scale_generator(
            downsampled.flatten(start_dim=-2, end_dim=-1)
        )
        if self.input_modulation:
            kernel_scale = kernel_scale.view(N, 1, C, 1, 1)
        else:
            kernel_scale = kernel_scale.view(N, self.out_channels, 1, 1, 1)
        weights = self.weight.unsqueeze(0) * kernel_scale
        if self.demodulate:
            dcoefs = torch.rsqrt(weights.pow(2).sum(dim=[2, 3, 4]) + self.eps)
            weights = weights * dcoefs.view(N, self.out_channels, 1, 1, 1)

        weights = weights.view(N * self.out_channels, C, self.kernel_size, self.kernel_size)

        grouped_input = x.view(1, N * C, H, W)

        output_grouped = F.conv2d(
            input=grouped_input,
            weight=weights,
            bias=None,
            stride=1,
            padding=self.padding,
            groups=N,
        )

        output = output_grouped.view(N, self.out_channels, H, W)
        return output
