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
"""A Module for convolutional layers."""

from __future__ import annotations

from typing import Literal

from max.driver import Accelerator, accelerator_api
from max.dtype import DType
from max.experimental import functional as F
from max.experimental import random
from max.experimental.tensor import Tensor
from max.graph import DeviceRef
from max.graph.type import FilterLayout
from max.nn.module_v3.module import Module


class Conv2d(Module[[Tensor], Tensor]):
    """A 2D convolution layer.

    This is a Conv2d implementation that uses Tensor instead of Weight objects.

    Example:
        .. code-block:: python

            from max.nn.module_v3 import Conv2d
            from max.experimental.tensor import Tensor

            conv = Conv2d(
                kernel_size=3,
                in_channels=3,
                out_channels=64,
                has_bias=True,
                permute=True,
            )

            x = Tensor.ones([1, 3, 32, 32])
            result = conv(x)
    """

    weight: Tensor
    """The weight tensor with shape [out_channels, in_channels // num_groups, kernel_height, kernel_width]."""

    bias: Tensor | Literal[0]
    """The bias tensor with shape [out_channels] (or 0 if bias is disabled)."""

    def __init__(
        self,
        kernel_size: int | tuple[int, int],
        in_channels: int,
        out_channels: int,
        dtype: DType | None = None,
        stride: int | tuple[int, int] = 1,
        padding: int | tuple[int, int] | tuple[int, int, int, int] = 0,
        dilation: int | tuple[int, int] = 1,
        num_groups: int = 1,
        device: DeviceRef | None = None,
        has_bias: bool = False,
        permute: bool = False,
        name: str | None = None,
    ):
        """Initialize Conv2d layer.

        Args:
            kernel_size: Size of the convolving kernel. Can be a single int (square kernel) or tuple (height, width).
            in_channels: Number of channels in the input image.
            out_channels: Number of channels produced by the convolution.
            dtype: The data type for both weights and bias. In v3, this is optional as Tensor manages dtype automatically.
            stride: Stride of the convolution for height and width dimensions.
                Can be int (applied to both dimensions) or tuple (stride_h, stride_w). Default: 1
            padding: Padding added to input. Can be int (applied to all sides),
                tuple of 2 ints (pad_h, pad_w), or tuple of 4 ints (pad_top, pad_bottom, pad_left, pad_right) to support asymmetric padding. Default: 0
            dilation: Spacing between kernel elements for height and width dimensions.
                Can be int (applied to both dimensions) or tuple (dilation_h, dilation_w). Default: 1
            num_groups: Number of blocked connections from input channels to output channels.
                Input channels and output channels are divided into groups. Default: 1
            device: The target device for computation. In v3, this is optional as Tensor manages device automatically.
            has_bias: If true, adds a learnable bias vector to the layer.
                Defaults to :obj:`False`.
            permute: If true, permutes weights from PyTorch format to MAX format.
                PyTorch order: (out_channels, in_channels / num_groups, height, width).
                MAX API order: (height, width, in_channels / num_groups, out_channels).
                Defaults to :obj:`False`.
            name: Base name for weights. In v3, this is stored but not used for Weight naming.
                Defaults to :obj:`None`.
        """
        # Store configuration for easy reconstruction
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.dtype = dtype
        self.device = device
        self.permute = permute
        self.num_groups = num_groups
        self.has_bias = has_bias
        self.name = name

        # Handle kernel_size as int or tuple
        if isinstance(kernel_size, int):
            kernel_height = kernel_width = kernel_size
            self.kernel_size = (kernel_size, kernel_size)
        else:
            kernel_height, kernel_width = kernel_size
            self.kernel_size = kernel_size

        self.weight = random.normal(
            [
                out_channels,
                in_channels // num_groups,
                kernel_height,
                kernel_width,
            ]
            if self.permute
            else [
                kernel_height,
                kernel_width,
                in_channels // num_groups,
                out_channels,
            ],
            dtype=self.dtype,
            device=self.device.to_device() if self.device is not None else None,
        )

        if has_bias:
            self.bias = random.normal(
                [out_channels],
                dtype=self.dtype,
                device=self.device.to_device()
                if self.device is not None
                else None,
            )
        else:
            self.bias = 0

        # Convert scalar parameters to tuples as needed
        self.stride = (stride, stride) if isinstance(stride, int) else stride

        if isinstance(padding, int):
            padding = (padding, padding, padding, padding)
        elif len(padding) == 2:
            # Convert (pad_h, pad_w) to (pad_top, pad_bottom, pad_left, pad_right)
            pad_h, pad_w = padding
            padding = (pad_h, pad_h, pad_w, pad_w)

        self.padding = padding

        if isinstance(dilation, int):
            dilation = (dilation, dilation)
        self.dilation = dilation

        if (
            isinstance(self.weight, Tensor)
            and hasattr(self.weight, "quantization_encoding")
            and self.weight.quantization_encoding is not None
        ):
            raise ValueError("Conv2d not implemented with weight quantization.")

    def forward(self, x: Tensor) -> Tensor:
        """Apply 2D convolution to input.

        Args:
            x: Input tensor. Shape depends on `permute`:
                - If permute=True: [batch_size, in_channels, height, width]
                - If permute=False: [batch_size, height, width, in_channels]

        Returns:
            Output tensor. Shape depends on `permute`:
                - If permute=True: [batch_size, out_channels, new_height, new_width]
                - If permute=False: [batch_size, new_height, new_width, out_channels]
        """
        # Move weight to same device as input
        weight = self.weight.to(x.device)

        is_nvidia_gpu = (
            isinstance(x.device, Accelerator) and accelerator_api() == "cuda"
        )

        if self.permute:
            # Input: [batch_size, in_channels, height, width] -> [batch_size, height, width, in_channels]
            x = F.permute(x, [0, 2, 3, 1])

            # GPU supports FCRS but CPU doesn't. On CPU, permute from
            # FCRS to RSCF format.
            if not is_nvidia_gpu:
                # Permute weight from [out_channels, in_channels // num_groups, height, width]
                # to [height, width, in_channels // num_groups, out_channels] (RSCF)
                weight = F.permute(weight, [2, 3, 1, 0])

        output = F.conv2d(
            x,
            weight,
            self.stride,
            self.dilation,
            self.padding,
            self.num_groups,
            self.bias if isinstance(self.bias, Tensor) else None,
            filter_layout=FilterLayout.FCRS
            if (self.permute and is_nvidia_gpu)
            else FilterLayout.RSCF,
        )

        if self.permute:
            # Output: [batch_size, new_height, new_width, out_channels] -> [batch_size, out_channels, new_height, new_width]
            output = F.permute(output, [0, 3, 1, 2])

        return output
