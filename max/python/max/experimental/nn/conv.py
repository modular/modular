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

from max.driver import Accelerator, Device, accelerator_api
from max.dtype import DType
from max.experimental import functional as F
from max.experimental import random
from max.experimental.nn.module import Module
from max.experimental.tensor import Tensor
from max.graph import DeviceRef
from max.graph.type import FilterLayout


class Conv2d(Module[[Tensor], Tensor]):
    """A 2D convolution layer.

    This is a Conv2d implementation that uses Tensor instead of Weight objects.

    Example:
        .. code-block:: python

            from max.experimental.nn import Conv2d
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
        device: Device | DeviceRef | None = None,
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

    Implemented by unsqueezing a height=1 dimension and delegating to
    :func:`max.experimental.functional.conv2d`.

    Example:
        .. code-block:: python

            from max.nn.module_v3 import Conv1d
            from max.experimental.tensor import Tensor

            conv = Conv1d(
                in_channels=16,
                out_channels=32,
                kernel_size=3,
                padding=1,
                permute=True,
            )

            x = Tensor.ones([1, 16, 64])  # [N, C, L] with permute=True
            result = conv(x)
    """

    weight: Tensor
    """The weight tensor. Shape [out_channels, in_channels // num_groups, kernel_size]
    when permute=True, or [kernel_size, in_channels // num_groups, out_channels] otherwise."""

    bias: Tensor | Literal[0]
    """The bias tensor with shape [out_channels] (or 0 if bias is disabled)."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        dtype: DType | None = None,
        stride: int = 1,
        padding: int | tuple[int, int] | str = 0,
        dilation: int = 1,
        num_groups: int = 1,
        device: DeviceRef | None = None,
        has_bias: bool = False,
        permute: bool = True,
        name: str | None = None,
    ):
        """Initialize Conv1d layer.

        Args:
            in_channels: Number of channels in the input.
            out_channels: Number of channels produced by the convolution.
            kernel_size: Size of the convolving kernel.
            dtype: The data type for weights and bias.
            stride: Stride of the convolution. Default: 1
            padding: Padding added to both sides of the input. Can be an int,
                a tuple of 2 ints (pad_left, pad_right), or the string "same"
                for same-size output (only valid with stride=1). Default: 0
            dilation: Spacing between kernel elements. Default: 1
            num_groups: Number of blocked connections from input to output channels.
                Default: 1
            device: The target device for computation.
            has_bias: If True, adds a learnable bias. Default: False
            permute: If True, expects PyTorch-style input [N, C, L] and stores
                weights in PyTorch order [C_out, C_in//G, K].
                If False, expects [N, L, C] and stores weights in [K, C_in//G, C_out].
                Defaults to :obj:`True` to match PyTorch NCL convention.
            name: Base name for weights. Default: None
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
        self._padding_arg = padding

        # Resolve padding to a (pad_left, pad_right) tuple.
        # "same" is resolved at forward time since it depends on dilation.
        if isinstance(padding, str):
            if padding != "same":
                raise ValueError(
                    f"Conv1d only supports padding='same' or an integer/tuple, got {padding!r}"
                )
            # Compute same padding: total = dilation * (kernel_size - 1)
            total = dilation * (kernel_size - 1)
            pad_left = total // 2
            pad_right = total - pad_left
            self.padding = (0, 0, pad_left, pad_right)
        elif isinstance(padding, int):
            self.padding = (0, 0, padding, padding)
        else:
            pad_left, pad_right = padding
            self.padding = (0, 0, pad_left, pad_right)

        _dev = self.device.to_device() if self.device is not None else None

        self.weight = random.normal(
            [out_channels, in_channels // num_groups, kernel_size]
            if permute
            else [kernel_size, in_channels // num_groups, out_channels],
            dtype=self.dtype,
            device=_dev,
        )

        if has_bias:
            self.bias = random.normal(
                [out_channels], dtype=self.dtype, device=_dev
            )
        else:
            self.bias = 0

    def forward(self, x: Tensor) -> Tensor:
        """Apply 1D convolution to input.

        Args:
            x: Input tensor.
                - If permute=True: [batch_size, in_channels, length]
                - If permute=False: [batch_size, length, in_channels]

        Returns:
            Output tensor.
                - If permute=True: [batch_size, out_channels, new_length]
                - If permute=False: [batch_size, new_length, out_channels]
        """
        weight = self.weight.to(x.device)
        weight = weight.cast(x.dtype)

        is_nvidia_gpu = False
        try:
            is_nvidia_gpu = (
                isinstance(x.device, Accelerator)
                and accelerator_api() == "cuda"
            )
        except Exception:
            pass

        # Manually simulate dilation by inserting zeros strictly into the weight
        D = self.dilation
        if D > 1:
            K = self.kernel_size
            K_eff = (K - 1) * D + 1
            if self.permute:
                # weight is [C_out, C_in/G, K]
                w_expand = F.unsqueeze(weight, 3)
                zeros = F.mul(w_expand, 0.0)
                pieces = [w_expand] + [zeros] * (D - 1)
                w_interleaved = F.concat(pieces, axis=3)
                weight = F.flatten(w_interleaved, 2, 3)
                eff_indices = F.arange(
                    0, K_eff, 1, dtype=DType.int64, device=x.device
                )
                weight = F.gather(weight, eff_indices, axis=2)
            else:
                # weight is [K, C_in/G, C_out]
                w_expand = F.unsqueeze(weight, 1)
                zeros = F.mul(w_expand, 0.0)
                pieces = [w_expand] + [zeros] * (D - 1)
                w_interleaved = F.concat(pieces, axis=1)
                weight = F.flatten(w_interleaved, 0, 1)
                eff_indices = F.arange(
                    0, K_eff, 1, dtype=DType.int64, device=x.device
                )
                weight = F.gather(weight, eff_indices, axis=0)

        if self.permute:
            # [N, C, L] -> [N, L, C] -> [N, 1, L, C]
            x = F.permute(x, [0, 2, 1])
            x = F.unsqueeze(x, 1)
            # GPU supports FCRS but CPU doesn't. On CPU, permute from
            # FCRS to RSCF format.
            if not is_nvidia_gpu:
                # [C_out, C_in/G, K] -> [K, C_in/G, C_out] -> [1, K, C_in/G, C_out] (RSCF)
                weight = F.permute(weight, [2, 1, 0])
                weight = F.unsqueeze(weight, 0)
            else:
                # Keep in FCRS: [C_out, C_in/G, K] -> [C_out, C_in/G, 1, K]
                weight = F.unsqueeze(weight, 2)
        else:
            # [N, L, C] -> [N, 1, L, C]
            x = F.unsqueeze(x, 1)
            # [K, C_in/G, C_out] -> [1, K, C_in/G, C_out] (RSCF)
            weight = F.unsqueeze(weight, 0)

        output = F.conv2d(
            x,
            weight,
            stride=(1, self.stride),
            dilation=(1, 1),
            padding=self.padding,
            groups=self.num_groups,
            bias=self.bias if isinstance(self.bias, Tensor) else None,
            filter_layout=FilterLayout.FCRS
            if (self.permute and is_nvidia_gpu)
            else FilterLayout.RSCF,
        )

        # [N, 1, L', C_out] -> [N, L', C_out]
        output = F.squeeze(output, 1)

        if self.permute:
            # [N, L', C_out] -> [N, C_out, L']
            output = F.permute(output, [0, 2, 1])

        return output


class ConvTranspose1d(Module[[Tensor], Tensor]):
    """A 1D transposed convolution layer.

    Implemented by unsqueezing a height=1 dimension and delegating to
    :func:`max.experimental.functional.conv2d_transpose`.

    Example:
        .. code-block:: python

            from max.nn.module_v3 import ConvTranspose1d
            from max.experimental.tensor import Tensor

            conv = ConvTranspose1d(
                in_channels=16,
                out_channels=32,
                kernel_size=4,
                stride=2,
                padding=1,
                permute=True,
            )

            x = Tensor.ones([1, 16, 32])  # [N, C_in, L] with permute=True
            result = conv(x)
    """

    weight: Tensor
    """The weight tensor. Shape [in_channels, out_channels // num_groups, kernel_size]
    when permute=True (PyTorch order), or [kernel_size, out_channels, in_channels // num_groups] otherwise (RSCF after unsqueeze)."""

    bias: Tensor | Literal[0]
    """The bias tensor with shape [out_channels] (or 0 if bias is disabled)."""

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
        num_groups: int = 1,
        device: DeviceRef | None = None,
        has_bias: bool = False,
        permute: bool = True,
        name: str | None = None,
    ):
        """Initialize ConvTranspose1d layer.

        Args:
            in_channels: Number of channels in the input.
            out_channels: Number of channels produced by the transposed convolution.
            kernel_size: Size of the convolving kernel.
            dtype: The data type for weights and bias.
            stride: Stride of the convolution. Default: 1
            padding: Amount to crop from each side of the output. Can be an int
                or tuple of 2 ints (pad_left, pad_right). Default: 0
            output_padding: Additional size added to the output. Must be less than
                stride. Default: 0
            dilation: Spacing between kernel elements. Default: 1
            num_groups: Number of blocked connections. Default: 1
            device: The target device for computation.
            has_bias: If True, adds a learnable bias. Default: False
            permute: If True, expects PyTorch-style input [N, C_in, L] and stores
                weights in PyTorch order [C_in, C_out//G, K].
                If False, expects [N, L, C_in] and stores weights in [K, C_out, C_in//G].
                Defaults to :obj:`True` to match PyTorch NCL convention.
            name: Base name for weights. Default: None
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

        if isinstance(padding, int):
            self.padding = (0, 0, padding, padding)
        else:
            pad_left, pad_right = padding
            self.padding = (0, 0, pad_left, pad_right)

        self.output_padding = (0, output_padding)

        _dev = self.device.to_device() if self.device is not None else None

        # PyTorch ConvTranspose1d weight shape: [in_channels, out_channels/groups, K]
        # MAX RSCF (after unsqueeze): [1, K, out_channels, in_channels/groups]
        self.weight = random.normal(
            [in_channels, out_channels // num_groups, kernel_size]
            if permute
            else [kernel_size, out_channels, in_channels // num_groups],
            dtype=self.dtype,
            device=_dev,
        )

        if has_bias:
            self.bias = random.normal(
                [out_channels], dtype=self.dtype, device=_dev
            )
        else:
            self.bias = 0

    def forward(self, x: Tensor) -> Tensor:
        """Apply 1D transposed convolution to input.

        Args:
            x: Input tensor.
                - If permute=True: [batch_size, in_channels, length]
                - If permute=False: [batch_size, length, in_channels]

        Returns:
            Output tensor.
                - If permute=True: [batch_size, out_channels, new_length]
                - If permute=False: [batch_size, new_length, out_channels]
        """
        weight = self.weight.to(x.device)
        weight = weight.cast(x.dtype)

        is_nvidia_gpu = False
        try:
            is_nvidia_gpu = (
                isinstance(x.device, Accelerator)
                and accelerator_api() == "cuda"
            )
        except Exception:
            pass

        # 1. Reverse the weight spatially (axis 2 of [C_in, C_out/G, K])
        K = self.weight.shape[-1]

        # Use MAX's native arange to build sequence [K-1, K-2, ..., 0]
        rev_indices = F.arange(
            start=K - 1, stop=-1, step=-1, dtype=DType.int64, device=x.device
        )
        weight = F.gather(weight, rev_indices, axis=2)

        # 2. Simulate weight dilation (zero-insertion)
        D = self.dilation
        K_eff = (K - 1) * D + 1
        if D > 1:
            # [C_in, C_out/G, K] -> [C_in, C_out/G, K, 1]
            w_expand = F.unsqueeze(weight, 3)
            zeros = F.mul(w_expand, 0.0)
            pieces = [w_expand] + [zeros] * (D - 1)
            # [C_in, C_out/G, K, D] -> [C_in, C_out/G, K * D]
            w_interleaved = F.concat(pieces, axis=3)
            weight = F.flatten(w_interleaved, 2, 3)
            # Drop the trailing D-1 zeros: slice from 0 to K_eff
            eff_indices = F.arange(
                start=0, stop=K_eff, step=1, dtype=DType.int64, device=x.device
            )
            weight = F.gather(weight, eff_indices, axis=2)

        # 3. Normalize input to [N, L, C_in]
        if self.permute:
            # [N, C_in, L] -> [N, L, C_in]
            x = F.permute(x, [0, 2, 1])

        # 4. Zero-insertion for stride
        S = self.stride
        if S > 1:
            x_expand = F.unsqueeze(x, 2)
            zeros = F.mul(x_expand, 0.0)
            pieces = [x_expand] + [zeros] * (S - 1)
            x_interleaved = F.concat(pieces, axis=2)
            x = F.flatten(x_interleaved, 1, 2)

        # 5. Padding calculation
        pad_in_left = (
            self.padding[2]
            if isinstance(self.padding, tuple) and len(self.padding) == 4
            else self.padding
        )
        pad_in_right = (
            self.padding[3]
            if isinstance(self.padding, tuple) and len(self.padding) == 4
            else self.padding
        )
        op_val = (
            self.output_padding[1]
            if isinstance(self.output_padding, tuple)
            and len(self.output_padding) == 2
            else self.output_padding
        )

        pad_left = K_eff - pad_in_left - 1
        pad_right = K_eff - pad_in_right - S + op_val
        conv2d_padding = (0, 0, pad_left, pad_right)

        # 6. Prepare input for 2D conv: [N, L*S, C_in] -> [N, 1, L*S, C_in]
        x = F.unsqueeze(x, 1)

        # 7. Prepare weight for 2D conv
        if is_nvidia_gpu:
            # Want FCRS: [C_out, C_in/G, R(1), S(K_eff)]
            # weight is [C_in, C_out/G, K_eff] -> [C_out/G, C_in, K_eff]
            weight = F.permute(weight, [1, 0, 2])
            weight = F.unsqueeze(weight, 2)
            flayout = FilterLayout.FCRS
        else:
            # Want RSCF: [R(1), S(K_eff), C_in/G, C_out]
            # weight is [C_in, C_out/G, K_eff] -> [K_eff, C_in, C_out/G]
            weight = F.permute(weight, [2, 0, 1])
            weight = F.unsqueeze(weight, 0)
            flayout = FilterLayout.RSCF

        output = F.conv2d(
            x,
            weight,
            stride=(1, 1),
            dilation=(1, 1),  # We simulated dilation manually!
            padding=conv2d_padding,
            groups=self.num_groups,
            bias=self.bias if isinstance(self.bias, Tensor) else None,
            filter_layout=flayout,
        )

        output = F.squeeze(output, 1)

        if self.permute:
            output = F.permute(output, [0, 2, 1])

        return output
