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


class WPDynConv(nn.Module):
    """A module for weight-prediction dynamic kernel generation.
    Utilizes an MLP to generate/predict Conv kernel weights based on input."""

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

        flattened_weights = self.kernel_generator(
            downsampled.flatten(start_dim=-2, end_dim=-1)
        )
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


class DepthwiseDynConv(nn.Module):
    """
    Dynamic Convolution using Dynamic Depthwise + Static Pointwise filters.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        reduction: int = 4,
        **kwargs,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = kernel_size // 2

        hidden_dim = max(in_channels // reduction, 16)
        num_dw_params = in_channels * kernel_size * kernel_size

        # Squeeze-and-Excitation style global context generator
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.kernel_generator = nn.Sequential(
            nn.Linear(in_channels, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, num_dw_params),
        )

        # Static Pointwise Conv for inter-channel interaction
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape

        # 1. Predict Depthwise Kernel Weights per sample in batch
        context = self.gap(x).view(B, C)
        dw_weights = self.kernel_generator(context)  # (B, C * K * K)
        dw_weights = dw_weights.view(B * C, 1, self.kernel_size, self.kernel_size)

        # 2. Reshape input for batch-wise grouped convolution
        x_grouped = x.view(1, B * C, H, W)

        # 3. Dynamic Depthwise Conv (groups = B * C)
        out_dw = F.conv2d(
            x_grouped,
            weight=dw_weights,
            bias=None,
            stride=self.stride,
            padding=self.padding,
            groups=B * C,
        )

        # 4. Reshape back and apply static Pointwise Conv
        out_dw = out_dw.view(B, C, H, W)
        out = self.conv(out_dw)
        return out


class LoRaDynConv(nn.Module):
    """
    Dynamic Kernel Generator using static base weights combined with
    low-rank dynamic adaptation matrices (W_dyn = W_base + U @ V).
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        rank: int = 4,
        reduction: int = 4,
        **kwargs,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = kernel_size // 2
        self.rank = rank

        # Static base weight tensor
        self.base_weight = nn.Parameter(
            torch.randn(out_channels, in_channels, kernel_size, kernel_size)
        )
        nn.init.kaiming_normal_(self.base_weight, mode="fan_out", nonlinearity="relu")

        # Hypernet predicting low-rank U and V factors
        hidden_dim = max(in_channels // reduction, 16)
        self.gap = nn.AdaptiveAvgPool2d(1)

        u_dim = out_channels * rank
        v_dim = rank * in_channels * kernel_size * kernel_size

        self.hypernet = nn.Sequential(
            nn.Linear(in_channels, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, u_dim + v_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape

        # Global context vector
        context = self.gap(x).view(B, C)
        dynamic_params = self.hypernet(context)

        u_dim = self.out_channels * self.rank
        u_flat, v_flat = dynamic_params.split(
            [u_dim, dynamic_params.shape[1] - u_dim], dim=1
        )

        # Reshape factors: U -> (B, C_out, rank), V -> (B, rank, C_in * K * K)
        U = u_flat.view(B, self.out_channels, self.rank)
        V = v_flat.view(
            B, self.rank, self.in_channels * self.kernel_size * self.kernel_size
        )

        # Compute dynamic residual offset delta_W = U @ V -> (B, C_out, C_in, K, K)
        delta_W = torch.bmm(U, V).view(
            B, self.out_channels, self.in_channels, self.kernel_size, self.kernel_size
        )

        # Aggregate static base weight with dynamic residual offset
        weights = self.base_weight.unsqueeze(0) + delta_W
        weights = weights.view(
            B * self.out_channels, self.in_channels, self.kernel_size, self.kernel_size
        )

        x_grouped = x.view(1, B * C, H, W)
        out = F.conv2d(
            x_grouped,
            weight=weights,
            bias=None,
            stride=self.stride,
            padding=self.padding,
            groups=B,
        )

        return out.view(B, self.out_channels, H, W)


class WMDynConv(nn.Module):
    """A module for weight-modulation dynamic kernel generation, similar to StyleGAN2.
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

        self.input_modulation = False
        self.demodulate = demodulate
        self.eps = eps

        self.in_channel_reduce = nn.Conv2d(in_channels, 1, 1)
        self.scale_generator = nn.Sequential(
            nn.Linear(self.downsample_size * self.downsample_size, kernel_gen_hidden),
            nn.ReLU(),
            nn.Linear(
                kernel_gen_hidden,
                in_channels if self.input_modulation else out_channels,
            ),
        )
        self.weight = nn.Parameter(
            torch.randn(out_channels, in_channels, kernel_size, kernel_size)
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

        weights = weights.view(
            N * self.out_channels, C, self.kernel_size, self.kernel_size
        )

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
