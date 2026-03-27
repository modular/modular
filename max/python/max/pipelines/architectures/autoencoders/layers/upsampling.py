# ===----------------------------------------------------------------------=== #
# Copyright (c) 2026, Modular Inc. All rights reserved.
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

from max.dtype import DType
from max.graph import DeviceRef, TensorValue, ops
from max.nn.conv import Conv2d
from max.nn.layer import Module


def interpolate_2d_nearest(
    x: TensorValue,
    scale_factor: int = 2,
) -> TensorValue:
    """Upsamples a 2D tensor using nearest-neighbor interpolation.

    This is a workaround implementation because MAX framework resize does not
    currently support NEAREST mode in this path. The workaround uses reshape
    and broadcast operations to achieve nearest-neighbor upsampling by a factor
    of 2.

    Note:
        This workaround can be removed once native nearest-neighbor resize is
        available in this path.
    """

    if x.rank != 4:
        raise ValueError(f"Input tensor must have rank 4, got {x.rank}")
    if scale_factor != 2:
        raise NotImplementedError(
            f"Only scale_factor=2 is currently supported, got {scale_factor}"
        )

    batch, channels, height, width = x.shape
    x = ops.reshape(x, [batch, channels, height, 1, width, 1])
    ones = ops.broadcast_to(
        ops.constant(1.0, dtype=x.dtype, device=x.device),
        [1, 1, 1, scale_factor, 1, scale_factor],
    )
    return ops.reshape(
        x * ones,
        [batch, channels, height * scale_factor, width * scale_factor],
    )


class Upsample2D(Module):
    """2D upsampling module with optional convolution.

    This module performs 2D upsampling using nearest-neighbor interpolation
    followed by an optional convolution layer.
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
        if dtype is None:
            raise ValueError("dtype must be set for Upsample2D")
        if device is None:
            raise ValueError("device must be set for Upsample2D")
        if use_conv_transpose:
            raise NotImplementedError(
                "Upsample2D does not support use_conv_transpose=True yet."
            )

        self.channels = channels
        self.out_channels = out_channels or channels
        self.interpolate = interpolate
        self.use_conv = use_conv
        self.use_conv_transpose = use_conv_transpose
        self.device = device
        self.dtype = dtype
        self.name = name
        self.conv: Conv2d | None = None
        if self.use_conv:
            self.conv = Conv2d(
                kernel_size=3 if kernel_size is None else kernel_size,
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
        if self.interpolate:
            x = interpolate_2d_nearest(x, scale_factor=2)
        if self.use_conv and self.conv is not None:
            x = self.conv(x)
        return x
