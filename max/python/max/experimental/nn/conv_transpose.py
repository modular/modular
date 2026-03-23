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
"""Transposed convolution layers for eager tensor modules."""

from __future__ import annotations

from typing import Literal

from max.driver import Device
from max.dtype import DType
from max.experimental import functional as F
from max.experimental import random
from max.experimental.nn.module import Module
from max.experimental.tensor import Tensor
from max.graph import DeviceRef
from max.graph.type import ConvInputLayout, FilterLayout


class ConvTranspose2d(Module[[Tensor], Tensor]):
    """A 2D transposed convolution layer."""

    weight: Tensor
    """The weight tensor with shape [in_channels, out_channels, kernel_height, kernel_width]."""

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
        output_padding: int | tuple[int, int] = 0,
        device: Device | DeviceRef | None = None,
        has_bias: bool = False,
        permute: bool = False,
        name: str | None = None,
    ):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.dtype = dtype
        self.device = device
        self.permute = permute
        self.has_bias = has_bias
        self.name = name

        if isinstance(kernel_size, int):
            kernel_height = kernel_width = kernel_size
            self.kernel_size = (kernel_size, kernel_size)
        else:
            kernel_height, kernel_width = kernel_size
            self.kernel_size = kernel_size

        self.weight = random.normal(
            [
                in_channels,
                out_channels,
                kernel_height,
                kernel_width,
            ]
            if self.permute
            else [
                kernel_height,
                kernel_width,
                out_channels,
                in_channels,
            ],
            dtype=self.dtype,
            device=self.device,
        )

        if has_bias:
            self.bias = random.normal(
                [out_channels],
                dtype=self.dtype,
                device=self.device,
            )
        else:
            self.bias = 0

        self.stride = (stride, stride) if isinstance(stride, int) else stride

        if isinstance(output_padding, int):
            output_padding = (output_padding, output_padding)
        self.output_padding = output_padding

        if isinstance(padding, int):
            padding = (padding, padding, padding, padding)
        elif len(padding) == 2:
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
            raise ValueError(
                "ConvTranspose2d not implemented with weight quantization."
            )

    def forward(self, x: Tensor) -> Tensor:
        weight = self.weight.to(x.device)
        bias = self.bias.to(x.device) if isinstance(self.bias, Tensor) else None

        return F.conv2d_transpose(
            x,
            weight,
            stride=self.stride,
            dilation=self.dilation,
            padding=self.padding,
            output_paddings=self.output_padding,
            bias=bias,
            input_layout=ConvInputLayout.NCHW
            if self.permute
            else ConvInputLayout.NHWC,
            filter_layout=FilterLayout.CFRS
            if self.permute
            else FilterLayout.RSCF,
        )
