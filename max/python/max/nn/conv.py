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

from max import functional as F
from max import random
from max.driver import Accelerator, accelerator_api
from max.dtype import DType
from max.graph import DeviceRef
from max.graph.type import ConvInputLayout, FilterLayout
from max.nn.module import Module
from max.tensor import Tensor


class Conv2d(Module[[Tensor], Tensor]):
    """A 2D convolution layer.

    This is a Conv2d implementation that uses Tensor instead of Weight objects.

    Example:
        .. code-block:: python

            from max.nn import Conv2d
            from max.tensor import Tensor

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


class Conv1d(Module[[Tensor], Tensor]):
    """A 1D convolution layer.

    This is a Conv1d implementation that uses Tensor instead of Weight objects.
    It internally uses Conv2d by unsqueezing the input to 4D and squeezing the output back.

    Example:
        .. code-block:: python

            from max.nn import Conv1d
            from max.tensor import Tensor

            conv = Conv1d(
                kernel_size=3,
                in_channels=3,
                out_channels=64,
                has_bias=True,
                permute=True,
            )

            x = Tensor.ones([1, 3, 32])  # [batch, channels, length]
            result = conv(x)
    """

    weight: Tensor
    """The weight tensor with shape [out_channels, in_channels // num_groups, kernel_size]."""

    bias: Tensor | Literal[0]
    """The bias tensor with shape [out_channels] (or 0 if bias is disabled)."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        dtype: DType | None = None,
        stride: int = 1,
        padding: int | str | tuple[int, int] = 0,
        dilation: int = 1,
        num_groups: int = 1,
        device: DeviceRef | None = None,
        has_bias: bool = False,
        permute: bool = True,
        name: str | None = None,
    ):
        """Initialize Conv1d layer.

        Args:
            in_channels: Number of channels in the input signal.
            out_channels: Number of channels produced by the convolution.
            kernel_size: Size of the convolving kernel.
            dtype: The data type for both weights and bias.
            stride: Stride of the convolution. Default: 1
            padding: Padding added to input. Can be int (applied to both sides),
                tuple of 2 ints (pad_left, pad_right), or "same" for same-size output. Default: 0
            dilation: Spacing between kernel elements. Default: 1
            num_groups: Number of blocked connections from input channels to output channels. Default: 1
            device: The target device for computation.
            has_bias: If true, adds a learnable bias vector to the layer. Defaults to False.
            permute: If true, expects PyTorch format (batch, channels, length).
                Defaults to True for PyTorch compatibility.
            name: Base name for weights. Defaults to None.
        """
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.dtype = dtype
        self.device = device
        self.permute = permute
        self.num_groups = num_groups
        self.has_bias = has_bias
        self.name = name
        self.stride = stride
        self.dilation = dilation

        # Handle padding
        if isinstance(padding, str):
            if padding == "same":
                # For "same" padding, we calculate it dynamically
                self.padding_mode = "same"
                self.padding = (0, 0)
            else:
                raise ValueError(f"Unsupported padding mode: {padding}")
        elif isinstance(padding, int):
            self.padding_mode = None
            self.padding = (padding, padding)
        else:
            self.padding_mode = None
            self.padding = padding

        self.weight = random.normal(
            [out_channels, in_channels // num_groups, kernel_size]
            if self.permute
            else [kernel_size, in_channels // num_groups, out_channels],
            dtype=self.dtype,
            device=self.device.to_device() if self.device is not None else None,
        )

        if has_bias:
            self.bias = random.normal(
                [out_channels],
                dtype=self.dtype,
                device=self.device.to_device() if self.device is not None else None,
            )
        else:
            self.bias = 0

    def forward(self, x: Tensor) -> Tensor:
        """Apply 1D convolution to input.

        Args:
            x: Input tensor. Shape depends on `permute`:
                - If permute=True: [batch_size, in_channels, length]
                - If permute=False: [batch_size, length, in_channels]

        Returns:
            Output tensor. Shape depends on `permute`:
                - If permute=True: [batch_size, out_channels, new_length]
                - If permute=False: [batch_size, new_length, out_channels]
        """
        weight = self.weight.to(x.device)

        is_nvidia_gpu = (
            isinstance(x.device, Accelerator) and accelerator_api() == "cuda"
        )

        # Handle "same" padding by calculating padding dynamically
        if self.padding_mode == "same":
            input_length = x.shape[-1] if self.permute else x.shape[1]
            effective_kernel = (self.kernel_size - 1) * self.dilation + 1
            total_padding = max(effective_kernel - 1, 0)
            pad_left = total_padding // 2
            pad_right = total_padding - pad_left
            padding = (pad_left, pad_right)
        else:
            padding = self.padding

        if self.permute:
            # Input: [batch_size, in_channels, length] -> [batch_size, in_channels, 1, length]
            x = x.unsqueeze(2)
            # Weight: [out_channels, in_channels // num_groups, kernel_size]
            #      -> [out_channels, in_channels // num_groups, 1, kernel_size]
            weight = weight.unsqueeze(2)

            if not is_nvidia_gpu:
                # Permute for CPU: [out_channels, in_channels // num_groups, 1, kernel_size]
                #               -> [1, kernel_size, in_channels // num_groups, out_channels]
                weight = F.permute(weight, [2, 3, 1, 0])
                # Input: [batch, channels, 1, length] -> [batch, 1, length, channels]
                x = F.permute(x, [0, 2, 3, 1])
        else:
            # Input: [batch_size, length, in_channels] -> [batch_size, 1, length, in_channels]
            x = x.unsqueeze(1)
            # Weight: [kernel_size, in_channels // num_groups, out_channels]
            #      -> [1, kernel_size, in_channels // num_groups, out_channels]
            weight = weight.unsqueeze(0)

        # Convert 1D padding to 2D: (pad_left, pad_right) -> (0, 0, pad_left, pad_right)
        pad_left, pad_right = padding
        padding_2d = (0, 0, pad_left, pad_right)

        output = F.conv2d(
            x,
            weight,
            (1, self.stride),
            (1, self.dilation),
            padding_2d,
            self.num_groups,
            self.bias if isinstance(self.bias, Tensor) else None,
            filter_layout=FilterLayout.FCRS
            if (self.permute and is_nvidia_gpu)
            else FilterLayout.RSCF,
        )

        if self.permute:
            if not is_nvidia_gpu:
                # Output: [batch, 1, new_length, out_channels] -> [batch, out_channels, 1, new_length]
                output = F.permute(output, [0, 3, 1, 2])
            # Remove dummy height dimension: [batch, out_channels, 1, new_length] -> [batch, out_channels, new_length]
            output = output.squeeze(2)
        else:
            # Remove dummy height dimension: [batch, 1, new_length, out_channels] -> [batch, new_length, out_channels]
            output = output.squeeze(1)

        return output


class ConvTranspose1d(Module[[Tensor], Tensor]):
    """A 1D transposed convolution layer.

    This is a ConvTranspose1d implementation that uses Tensor instead of Weight objects.
    It internally uses conv2d_transpose by unsqueezing the input to 4D and squeezing the output back.

    Example:
        .. code-block:: python

            from max.nn import ConvTranspose1d
            from max.tensor import Tensor

            conv = ConvTranspose1d(
                kernel_size=4,
                in_channels=64,
                out_channels=32,
                stride=2,
                has_bias=True,
                permute=True,
            )

            x = Tensor.ones([1, 64, 16])  # [batch, channels, length]
            result = conv(x)  # [batch, 32, 32]
    """

    weight: Tensor
    """The weight tensor with shape [in_channels, out_channels, kernel_size]."""

    bias: Tensor | None
    """The bias tensor with shape [out_channels] (or None if bias is disabled)."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        dtype: DType | None = None,
        stride: int = 1,
        padding: int | tuple[int, int] = 0,
        output_padding: int = 0,
        dilation: int = 1,
        device: DeviceRef | None = None,
        has_bias: bool = False,
        permute: bool = True,
        name: str | None = None,
    ):
        """Initialize ConvTranspose1d layer.

        Args:
            in_channels: Number of channels in the input signal.
            out_channels: Number of channels produced by the convolution.
            kernel_size: Size of the convolving kernel.
            dtype: The data type for both weights and bias.
            stride: Stride of the convolution. Default: 1
            padding: Padding subtracted from output. Default: 0
            output_padding: Additional size added to one side of the output. Default: 0
            dilation: Spacing between kernel elements. Default: 1
            device: The target device for computation.
            has_bias: If true, adds a learnable bias vector to the layer. Defaults to False.
            permute: If true, expects PyTorch format (batch, channels, length).
                Defaults to True for PyTorch compatibility.
            name: Base name for weights. Defaults to None.
        """
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.dtype = dtype
        self.device = device
        self.permute = permute
        self.has_bias = has_bias
        self.name = name
        self.stride = stride
        self.dilation = dilation
        self.output_padding = output_padding

        # Handle padding
        if isinstance(padding, int):
            self.padding = (padding, padding)
        else:
            self.padding = padding

        self.weight = random.normal(
            [in_channels, out_channels, kernel_size]
            if self.permute
            else [kernel_size, out_channels, in_channels],
            dtype=self.dtype,
            device=self.device.to_device() if self.device is not None else None,
        )

        if has_bias:
            self.bias = random.normal(
                [out_channels],
                dtype=self.dtype,
                device=self.device.to_device() if self.device is not None else None,
            )
        else:
            self.bias = None

    def forward(self, x: Tensor) -> Tensor:
        """Apply 1D transposed convolution to input.

        Args:
            x: Input tensor. Shape depends on `permute`:
                - If permute=True: [batch_size, in_channels, length]
                - If permute=False: [batch_size, length, in_channels]

        Returns:
            Output tensor. Shape depends on `permute`:
                - If permute=True: [batch_size, out_channels, new_length]
                - If permute=False: [batch_size, new_length, out_channels]
        """
        weight = self.weight.to(x.device)

        if self.permute:
            # Input: [batch, in_channels, length] -> [batch, in_channels, 1, length]
            x = x.unsqueeze(2)
            # Weight: [in_channels, out_channels, kernel_size]
            #      -> [in_channels, out_channels, 1, kernel_size]
            weight = weight.unsqueeze(2)
        else:
            # Input: [batch, length, in_channels] -> [batch, 1, length, in_channels]
            x = x.unsqueeze(1)
            # Weight: [kernel_size, out_channels, in_channels]
            #      -> [1, kernel_size, out_channels, in_channels]
            weight = weight.unsqueeze(0)

        # Convert 1D parameters to 2D
        pad_left, pad_right = self.padding
        padding_2d = (0, 0, pad_left, pad_right)

        output = F.conv2d_transpose(
            x,
            weight,
            stride=(1, self.stride),
            dilation=(1, self.dilation),
            padding=padding_2d,
            output_paddings=(0, self.output_padding),
            bias=self.bias,
            input_layout=ConvInputLayout.NCHW if self.permute else ConvInputLayout.NHWC,
            filter_layout=FilterLayout.CFRS if self.permute else FilterLayout.RSCF,
        )

        if self.permute:
            # Remove dummy height dimension: [batch, out_channels, 1, new_length] -> [batch, out_channels, new_length]
            output = output.squeeze(2)
        else:
            # Remove dummy height dimension: [batch, 1, new_length, out_channels] -> [batch, new_length, out_channels]
            output = output.squeeze(1)

        return output

class Conv3d(Module[[Tensor], Tensor]):
    """A 3D convolution layer.

    Example:
        .. code-block:: python

            from max.nn import Conv3d
            from max.tensor import Tensor

            conv = Conv3d(
                kernel_size=3,
                in_channels=3,
                out_channels=64,
                has_bias=True,
                permute=True,
            )

            x = Tensor.ones([1, 8, 32, 32, 3])
            result = conv(x)
    """

    weight: Tensor
    """The weight tensor with shape [depth, height, width, in_channels / num_groups, out_channels]."""

    bias: Tensor | Literal[0]
    """The bias tensor with shape [out_channels] (or 0 if bias is disabled)."""

    def __init__(
        self,
        kernel_size: int | tuple[int, int, int],
        in_channels: int,
        out_channels: int,
        dtype: DType | None = None,
        stride: int | tuple[int, int, int] = 1,
        padding: int | tuple[int, int, int] | tuple[int, int, int, int, int, int] = 0,
        dilation: int | tuple[int, int, int] = 1,
        num_groups: int = 1,
        device: DeviceRef | None = None,
        has_bias: bool = False,
        permute: bool = False,
        name: str | None = None,
    ):
        """Initialize Conv3d layer.

        Args:
            kernel_size: Size of the convolving kernel.
            in_channels: Number of channels in the input.
            out_channels: Number of channels produced by the convolution.
            dtype: The data type for weights and bias.
            stride: Stride of the convolution. Default: 1
            padding: Padding added to input. Default: 0
            dilation: Spacing between kernel elements. Default: 1
            num_groups: Number of groups. Default: 1
            device: The target device.
            has_bias: If true, adds a learnable bias vector. Defaults to False.
            permute: If true, permutes weights from PyTorch format to MAX format.
                Defaults to False.
            name: Base name for weights.
        """
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.dtype = dtype
        self.device = device
        self.permute = permute
        self.num_groups = num_groups
        self.has_bias = has_bias
        self.name = name

        if isinstance(kernel_size, int):
            self.kernel_size = (kernel_size, kernel_size, kernel_size)
        else:
            self.kernel_size = kernel_size

        kd, kh, kw = self.kernel_size

        self.weight = random.normal(
            [out_channels, in_channels // num_groups, kd, kh, kw]
            if self.permute
            else [kd, kh, kw, in_channels // num_groups, out_channels],
            dtype=self.dtype,
            device=self.device.to_device() if self.device is not None else None,
        )

        if has_bias:
            self.bias = random.normal(
                [out_channels],
                dtype=self.dtype,
                device=self.device.to_device() if self.device is not None else None,
            )
        else:
            self.bias = 0

        self.stride = (
            (stride, stride, stride) if isinstance(stride, int) else stride
        )

        if isinstance(padding, int):
            self.padding = (padding, padding, padding, padding, padding, padding)
        elif len(padding) == 3:
            pd, ph, pw = padding
            self.padding = (pd, pd, ph, ph, pw, pw)
        else:
            self.padding = padding

        if isinstance(dilation, int):
            self.dilation = (dilation, dilation, dilation)
        else:
            self.dilation = dilation

    def forward(self, x: Tensor) -> Tensor:
        """Apply 3D convolution to input.

        Args:
            x: Input tensor. Shape depends on `permute`:
                - If permute=True: [batch, in_channels, depth, height, width]
                - If permute=False: [batch, depth, height, width, in_channels]

        Returns:
            Output tensor.
        """
        weight = self.weight.to(x.device)

        if self.permute:
            # PyTorch format: NCDHW -> NDHWC
            x = F.permute(x, [0, 2, 3, 4, 1])
            # Weight: FCDHW -> DHWCF
            weight = F.permute(weight, [2, 3, 4, 1, 0])

        output = F.conv3d(
            x,
            weight,
            self.stride,
            self.dilation,
            self.padding,
            self.num_groups,
            self.bias if isinstance(self.bias, Tensor) else None,
            filter_layout=self.weight.quantization_encoding
            if hasattr(self.weight, "quantization_encoding")
            else None,  # Placeholder for layout logic if needed
        )

        if self.permute:
            # NDHWC -> NCDHW
            output = F.permute(output, [0, 4, 1, 2, 3])

        return output
