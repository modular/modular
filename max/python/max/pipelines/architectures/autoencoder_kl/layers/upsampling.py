# ===----------------------------------------------------------------------=== #
# Copyright (c) 2025, Modular Inc. All rights reserved.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions:
# https://llvm.org/LICENSE.txt
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ===----------------------------------------------------------------------=== #

"""Upsampling utilities for MAX framework."""

import max.nn as nn
from max.dtype import DType
from max.experimental import tensor
from max.graph import DeviceRef, TensorValue, ops


class Interpolate2DNearest(nn.Module):
    """2D nearest-neighbor upsampling module.

    This is a workaround implementation because MAX framework does not have
    a native `interpolate` operation. The workaround uses reshape and broadcast
    operations to achieve nearest-neighbor upsampling by a factor of 2.

    Note:
        This workaround can be removed once MAX framework adds native interpolate support.
    """

    def __init__(
        self,
        scale_factor: int = 2,
        device: DeviceRef = None,
        dtype: DType = None,
    ):
        """Initialize 2D nearest-neighbor interpolation module.

        Args:
            scale_factor: Upsampling factor (currently only 2 is supported).
            device: Device reference for creating intermediate tensors.
            dtype: Data type for intermediate tensors.
        """
        super().__init__()
        if scale_factor != 2:
            raise NotImplementedError(
                f"Only scale_factor=2 is currently supported, got {scale_factor}"
            )

        self.scale_factor = scale_factor
        self.device = device
        self.dtype = dtype

    def __call__(self, x: TensorValue) -> TensorValue:
        """Upsample a 2D tensor using nearest-neighbor interpolation.

        Args:
            x: Input tensor of shape [N, C, H, W].

        Returns:
            Upsampled tensor of shape [N, C, H*scale_factor, W*scale_factor].
        """
        n, c, h, w = x.shape
        target_shape = [n, c, h * self.scale_factor, w * self.scale_factor]

        x_reshaped = ops.reshape(x, [n, c, h, 1, w, 1])

        ones = tensor.Tensor.ones(
            shape=(1, 1, 1, self.scale_factor, 1, self.scale_factor),
            dtype=self.dtype,
            device=self.device,
        )
        x_expanded = x_reshaped * ones

        x = ops.reshape(x_expanded, target_shape)

        return x


class Upsample2D(nn.Module):
    """2D upsampling module with optional convolution.

    This module performs 2D upsampling using nearest-neighbor interpolation
    (via Interpolate2DNearest workaround) followed by an optional convolution layer.
    """

    def __init__(
        self,
        channels: int,
        use_conv: bool = False,
        use_conv_transpose: bool = False,
        out_channels: int | None = None,
        name: str = "conv",
        kernel_size: int | None = None,
        padding: int = 1,
        bias: bool = True,
        interpolate: bool = True,
        device: DeviceRef | None = None,
        dtype: DType | None = None,
    ) -> None:
        """Initialize 2D upsampling module.

        Args:
            channels: Number of input channels.
            use_conv: Whether to apply a convolution after upsampling.
            use_conv_transpose: Whether to use transposed convolution (not supported yet).
            out_channels: Number of output channels. If None, uses channels.
            name: Name for the convolution layer (unused, kept for compatibility).
            kernel_size: Kernel size for the convolution.
            padding: Padding for the convolution.
            bias: Whether to use bias in the convolution.
            interpolate: Whether to perform interpolation upsampling.
            device: Device reference.
            dtype: Data type.
        """
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.use_conv_transpose = use_conv_transpose
        self.interpolate = interpolate
        self.device = device
        self.dtype = dtype

        self.interpolate_module = None
        if interpolate:
            self.interpolate_module = Interpolate2DNearest(
                scale_factor=2, device=device, dtype=dtype
            )

        self.conv = None
        if use_conv_transpose:
            raise NotImplementedError(
                "Upsample2D does not support use_conv_transpose=True yet."
            )
        elif use_conv:
            if kernel_size is None:
                kernel_size = 3
            self.conv = nn.Conv2d(
                kernel_size=kernel_size,
                in_channels=self.channels,
                out_channels=self.out_channels,
                dtype=dtype,
                stride=1,
                padding=padding,
                has_bias=bias,
                device=device,
                permute=True,
            )

    def __call__(self, x: TensorValue) -> TensorValue:
        """Apply 2D upsampling with optional convolution.

        Args:
            x: Input tensor of shape [N, C, H, W].

        Returns:
            Upsampled tensor, optionally convolved.
        """
        if self.interpolate_module is not None:
            x = self.interpolate_module(x)

        if self.use_conv:
            x = self.conv(x)

        return x
