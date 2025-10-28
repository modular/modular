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

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import max.driver as md
from max.dtype import DType
from max.graph import (
    DeviceRef,
    ShardingStrategy,
    TensorValue,
    TensorValueLike,
    Weight,
    ops,
)
from max.graph.type import FilterLayout

from .layer import Layer, Module, Shardable


class Conv2d(Module, Shardable):
    """A 2D convolution over an input signal composed of several input
    planes.

    Example:
        .. code-block:: python

            conv = nn.Conv2d(
                kernel_size=3,
                in_channels=64,
                out_channels=128,
                dtype=DType.float32,
                stride=1,
                padding=0,
                has_bias=False,
                name="conv2d_weight",
                device=DeviceRef.GPU(),
            )
    """

    device: DeviceRef | None
    """The device where matrix operations are performed."""

    filter: Weight
    """The weight matrix stored on CPU with shape (height, width, in_channels / num_groups, out_channels).
    Model init moves the weight to :obj:`device`."""

    stride: tuple[int, int]
    """Controls the stride for the cross-correlation."""

    padding: tuple[int, int, int, int]
    """Controls the amount of padding applied before and after the input for height and width dimensions."""

    dilation: tuple[int, int]
    """Controls the dilation rate."""

    num_groups: int
    """Number of blocked connections from input channels to output channels."""

    bias: Weight | None = None
    """The optional bias vector stored on CPU with shape (out_channels,).
    Model init moves the bias to :obj:`device` if present."""

    permute: bool = False
    """bool controls whether self.filter is permuted from PyTorch order to max order.
    PyTorch order is: (out_channels, in_channels / num_groups, height, width)
    Max API order: (height, width, in_channels / num_groups, out_channels)."""

    def __init__(
        self,
        kernel_size: int | tuple[int, int],
        in_channels: int,
        out_channels: int,
        dtype: DType,
        stride: int | tuple[int, int] = 1,
        padding: int | tuple[int, int] | tuple[int, int, int, int] = 0,
        dilation: int | tuple[int, int] = 1,
        num_groups: int = 1,
        device: DeviceRef | None = None,
        has_bias: bool = False,
        permute: bool = False,
        name: str | None = None,
    ) -> None:
        """Initializes the Conv2d layer with weights and optional bias.

        Args:
            kernel_size: Size of the convolving kernel. Can be a single int (square kernel) or tuple (height, width).
            in_channels: Number of channels in the input image.
            out_channels: Number of channels produced by the convolution.
            dtype: The data type for both weights and bias.
            stride: Stride of the convolution for height and width dimensions.
                Can be int (applied to both dimensions) or tuple (stride_h, stride_w). Default: 1
            padding: Padding added to input. Can be int (applied to all sides),
                tuple of 2 ints (pad_h, pad_w), or tuple of 4 ints (pad_top, pad_bottom, pad_left, pad_right). Default: 0
            dilation: Spacing between kernel elements for height and width dimensions.
                Can be int (applied to both dimensions) or tuple (dilation_h, dilation_w). Default: 1
            num_groups: Number of blocked connections from input channels to output channels.
                Input channels and output channels are divided into groups. Default: 1
            device: The target device for computation. If None, defaults to CPU.
                Weights are initially stored on CPU and moved to target device during computation.
            name: Base name for weights. If provided, weights are named ``{name}.weight`` and
                ``{name}.bias`` (if bias is enabled). If None, uses "weight" and "bias".
            has_bias: If true, adds a learnable bias vector to the layer.
                Defaults to :obj:`False`.
            permute: If true, permutes weights from PyTorch format to MAX format.
                PyTorch order: (out_channels, in_channels / num_groups, height, width).
                MAX API order: (height, width, in_channels / num_groups, out_channels).
                Defaults to :obj:`False`.
        """
        super().__init__()

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

        self.filter = Weight(
            name=f"{name}.weight" if name else "weight",
            dtype=dtype,
            shape=(
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
                ]
            ),
            device=self.device or DeviceRef.CPU(),
        )

        if has_bias:
            self.bias = Weight(
                name=f"{name}.bias" if name else "bias",
                dtype=dtype,
                shape=(out_channels,),
                device=self.device or DeviceRef.CPU(),
            )

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
            isinstance(self.filter, Weight)
            and self.filter.quantization_encoding is not None
        ):
            raise ValueError("Conv2d not implemented with weight quantization.")

    @property
    def sharding_strategy(self) -> ShardingStrategy | None:
        """Get the Conv2d sharding strategy."""
        # Always take the sharding strategy of the conv filter.
        return self.filter.sharding_strategy

    @sharding_strategy.setter
    def sharding_strategy(self, strategy: ShardingStrategy) -> None:
        """Set the sharding strategy for the conv layer.

        Args:
            strategy: The strategy describing the conv's sharding.
        """
        if not strategy.is_replicate:
            raise ValueError(
                "only replicate is supported for Conv2d, currently"
            )

        self.filter.sharding_strategy = strategy
        if self.bias:
            self.bias.sharding_strategy = strategy

    def shard(self, devices: Iterable[DeviceRef]) -> list[Conv2d]:
        """Creates sharded views of this Conv2d layer across multiple devices.

        Args:
            devices: Iterable of devices to place the shards on.

        Returns:
            List of sharded Conv2d instances, one for each device.
        """
        if not self.sharding_strategy:
            raise ValueError(
                "Conv2d layer cannot be sharded because no sharding strategy was provided."
            )
        assert self.sharding_strategy.is_replicate

        # Get sharded weights
        sharded_filters = self.filter.shard(devices)
        sharded_biases = self.bias.shard(devices) if self.bias else None

        shards = []
        for idx, (device, filter_shard) in enumerate(
            zip(devices, sharded_filters, strict=True)
        ):
            # Create new Conv2d with same configuration.
            sharded = Conv2d(
                kernel_size=self.kernel_size,
                in_channels=self.in_channels,
                out_channels=self.out_channels,
                dtype=self.dtype,
                stride=self.stride,
                padding=self.padding,
                dilation=self.dilation,
                num_groups=self.num_groups,
                device=device,
                has_bias=self.has_bias,
                permute=self.permute,
                name=self.name,
            )

            # Replace the weights with sharded versions.
            sharded.filter = filter_shard
            if sharded_biases:
                sharded.bias = sharded_biases[idx]

            shards.append(sharded)

        return shards

    def __call__(self, x: TensorValue) -> TensorValue:
        """Apply 2D convolution to input `x`. Permutes pytorch weights to match max API if permute=True.

        Args:
            x: a tensor of shape [batch_size, height, width, in_channels]
            if self.permute, then input is of shape: [batch_size, in_channels, height, width]
            and will be permuted to match max's expected input shape.

        Returns:
            a tensor of shape [batch_size, new_height, new_width, out_channels]
            if self.permute, then output shape will be [batch_size, out_channels, new_height, new_width]
        """
        weight: TensorValue = self.filter

        is_nvidia_gpu = (
            isinstance(self.device, DeviceRef)
            and self.device.is_gpu()
            and md.accelerator_api() == "cuda"
        )

        if self.permute:
            # Input: [batch_size, in_channels, height, width] -> [batch_size, height, width, in_channels]
            x = ops.permute(x, [0, 2, 3, 1])

            # GPU supports FCRS but CPU doesn't. On CPU, permute from
            # FCRS to RSCF format.
            if not is_nvidia_gpu:
                # Permute weight from [out_channels, in_channels // num_groups, height, width]
                # to [height, width, in_channels // num_groups, out_channels] (RSCF)
                weight = ops.permute(weight, [2, 3, 1, 0])

        output = ops.conv2d(
            x,
            weight,
            self.stride,
            self.dilation,
            self.padding,
            self.num_groups,
            self.bias,
            filter_layout=FilterLayout.FCRS
            if (self.permute and is_nvidia_gpu)
            else FilterLayout.RSCF,
        )

        if self.permute:
            # Output: [batch_size, new_height, new_width, out_channels] -> [batch_size, out_channels, new_height, new_width]
            output = ops.permute(output, [0, 3, 1, 2])

        return output


@dataclass
class Conv2dV1(Layer):
    """A 2D convolution over an input signal composed of several input
    planes.

    DEPRECATED: Use :obj:`Conv2d` instead.
    """

    filter: TensorValueLike
    bias: TensorValueLike | None = None

    stride: int | tuple[int, int] = (1, 1)
    padding: int | tuple[int, int, int, int] = (0, 0, 0, 0)
    dilation: int | tuple[int, int] = (1, 1)
    groups: int = 1

    def __call__(self, x: TensorValue) -> TensorValue:
        # These need to be casted as the underlying ops.conv2d call
        # expects them to only be tuple types.
        if isinstance(self.stride, int):
            self.stride = (self.stride, self.stride)

        if isinstance(self.padding, int):
            self.padding = (
                self.padding,
                self.padding,
                self.padding,
                self.padding,
            )

        if isinstance(self.dilation, int):
            self.dilation = (self.dilation, self.dilation)

        if (
            isinstance(self.filter, Weight)
            and self.filter.quantization_encoding is not None
        ):
            raise ValueError(
                "Conv2dV1 not implemented with weight quantization."
            )
        return ops.conv2d(
            x,
            self.filter,
            self.stride,
            self.dilation,
            self.padding,
            self.groups,
            self.bias,
        )


@dataclass
class Conv1DV1(Layer):
    """A 1D convolution over an input signal composed of several input
    planes.

    DEPRECATED: Use :obj:`Conv1D` instead.
    """

    filter: TensorValueLike  # [kernel_size, in_channels, out_channels]
    bias: TensorValueLike | None = None

    stride: int = 1
    padding: int = 0
    dilation: int = 1
    groups: int = 1

    def __call__(self, x: TensorValueLike) -> TensorValue:
        """
        Args:
            x: a tensor of shape [batch_size, length, in_channels]

        Returns:
            a tensor of shape [batch_size, new_length, out_channels]
            new_length = ((length + 2 * padding - (kernel_size - 1) - 1) / stride) + 1
        """
        # TODO(GEX-327): Support Conv1D in mo rather than implementing it using Conv2dV1.
        # Reshape [batch_size, length, in_channels] to [batch_size, height=1, length, in_channels].
        x = ops.unsqueeze(x, 1)
        # Reshape  [kernel_size, in_channels, out_channels] to [height=1, kernel_size, in_channels, out_channels].
        filter = ops.unsqueeze(self.filter, 0)
        if (
            isinstance(self.filter, Weight)
            and self.filter.quantization_encoding is not None
        ):
            raise ValueError("Conv1D not implemented with weight quantization.")
        else:
            output = ops.conv2d(
                x,
                filter,
                (1, self.stride),
                (1, self.dilation),
                (0, 0, self.padding, self.padding),
                self.groups,
                self.bias,
            )
        # Reshape [batch_size, height=1, new_length, out_channels] to [batch_size, new_length, out_channels].
        return ops.squeeze(output, 1)


class Conv1D(Module):
    """A 1D convolution over an input signal composed of several input
    planes.

    Example:
        .. code-block:: python

            conv = nn.Conv1D(
                kernel_size=3,
                in_channels=64,
                out_channels=128,
                dtype=DType.float32,
                stride=1,
                padding=0,
                has_bias=False,
                name="conv1d_weight",
                device=DeviceRef.GPU(),
            )
    """

    device: DeviceRef | None
    """The device where matrix operations are performed."""

    filter: Weight
    """The weight matrix stored on CPU with shape (kernel_size, in_channels / num_groups, out_channels).
    Model init moves the weight to :obj:`device`."""

    stride: int
    """Controls the stride for the cross-correlation."""

    padding: int
    """Controls the amount of padding applied before and after the input."""

    dilation: int
    """Controls the dilation rate."""

    num_groups: int
    """Number of blocked connections from input channels to output channels."""

    bias: Weight | None = None
    """The optional bias vector stored on CPU with shape (out_channels,).
    Model init moves the bias to :obj:`device` if present."""

    permute: bool = False
    """bool controls whether self.filter is permuted from PyTorch order to max order.
    PyTorch order is: (out_channels, in_channels / num_groups, kernel_size)
    Max API order: (kernel_size, in_channels / num_groups, out_channels)."""

    def __init__(
        self,
        kernel_size: int,
        in_channels: int,
        out_channels: int,
        dtype: DType,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
        num_groups: int = 1,
        device: DeviceRef | None = None,
        has_bias: bool = False,
        permute: bool = False,
        name: str | None = None,
    ) -> None:
        """Initializes the Conv1D layer with weights and optional bias.

        Args:
            kernel_size: Size of the convolving kernel (width dimension).
            in_channels: Number of channels in the input signal.
            out_channels: Number of channels produced by the convolution.
            dtype: The data type for both weights and bias.
            stride: Stride of the convolution. Controls the step size when sliding the kernel. Default: 1
            padding: Padding added to both sides of the input sequence. Default: 0
            dilation: Spacing between kernel elements. Controls the kernel dilation rate. Default: 1
            num_groups: Number of blocked connections from input channels to output channels.
                Input channels and output channels are divided into groups. Default: 1
            device: The target device for computation. If None, defaults to CPU.
                Weights are initially stored on CPU and moved to target device during computation.
            name: Base name for weights. If provided, weights are named ``{name}.weight`` and
                ``{name}.bias`` (if bias is enabled). If None, uses "weight" and "bias".
            has_bias: If true, adds a learnable bias vector to the layer.
                Defaults to :obj:`False`.
            permute: If true, permutes weights from PyTorch format to MAX format.
                PyTorch order: (out_channels, in_channels / num_groups, kernel_size).
                MAX API order: (kernel_size, in_channels / num_groups, out_channels).
                Defaults to :obj:`False`.
        """
        super().__init__()

        self.device = device
        self.permute = permute

        if self.permute:
            self.filter = Weight(
                name=f"{name}.weight" if name else "weight",
                dtype=dtype,
                shape=[out_channels, in_channels // num_groups, kernel_size],
                device=self.device or DeviceRef.CPU(),
            )
        else:
            self.filter = Weight(
                name=f"{name}.weight" if name else "weight",
                dtype=dtype,
                shape=[kernel_size, in_channels // num_groups, out_channels],
                device=self.device or DeviceRef.CPU(),
            )

        if has_bias:
            self.bias = Weight(
                name=f"{name}.bias" if name else "bias",
                dtype=dtype,
                shape=(out_channels,),
                device=self.device or DeviceRef.CPU(),
            )

        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.num_groups = num_groups

        if (
            isinstance(self.filter, Weight)
            and self.filter.quantization_encoding is not None
        ):
            raise ValueError("Conv1D not implemented with weight quantization.")

    def __call__(self, x: TensorValue) -> TensorValue:
        """Applied 1D convolution to input `x`. Permutes pytorch weights to match max API if permute=True.

        Args:
            x: a tensor of shape [batch_size, length, in_channels]
            if self.permute, then input is of shape: [batch_size, in_channels, length]
            and will be permuted to match max's expected input shape.

        Returns:
            a tensor of shape [batch_size, new_length, out_channels]
            if self.permute, then output shape will be [batch_size, out_channels, new_length]
            new_length = ((length + 2 * padding - (kernel_size - 1) - 1) / stride) + 1
        """
        weight: TensorValue = self.filter

        is_nvidia_gpu = (
            isinstance(self.device, DeviceRef)
            and self.device.is_gpu()
            and md.accelerator_api() == "cuda"
        )

        if self.permute:
            x = ops.permute(x, [0, 2, 1])  # [batch_size, length, in_channels]

            # GPU supports FCRS but CPU doesn't. On CPU, permute from
            # FCS to SCF, then add dummy dim to become RSCF.
            if not is_nvidia_gpu:
                weight = ops.unsqueeze(ops.permute(weight, [2, 1, 0]), 0)
            # on GPU, unsqueeze FCS to FCRS
            else:
                weight = ops.unsqueeze(weight, 2)
        # No permute, filer is SCF and unsqueeze to RSCF.
        else:
            weight = ops.unsqueeze(weight, 0)

        # Reshape for Conv2dV1
        x = ops.unsqueeze(x, 1)  # [batch_size, height=1, length, in_channels]

        output = ops.conv2d(
            x,
            weight,
            (1, self.stride),
            (1, self.dilation),
            (0, 0, self.padding, self.padding),
            self.num_groups,
            self.bias,
            filter_layout=FilterLayout.FCRS
            if (self.permute and is_nvidia_gpu)
            else FilterLayout.RSCF,
        )

        # Reshape back from Conv2dV1
        output = ops.squeeze(
            output, 1
        )  # [batch_size, new_length, out_channels]

        if self.permute:
            output = ops.permute(
                output, [0, 2, 1]
            )  # [batch_size, out_channels, new_length]

        return output


@dataclass
class Conv3DV1(Layer):
    """A 3D convolution over an input signal composed of several input
    planes.

    DEPRECATED: Use :obj:`Conv3D` instead.
    """

    filter: TensorValueLike  # [depth, height, width, in_channels / num_groups, out_channels]
    bias: TensorValueLike | None = None  # [out_channels]

    stride: int | tuple[int, int, int] = (1, 1, 1)
    padding: int | tuple[int, int, int, int, int, int] = (
        0,
        0,
        0,
        0,
        0,
        0,
    )
    dilation: int | tuple[int, int, int] = (1, 1, 1)
    groups: int = 1

    def __call__(self, x: TensorValueLike) -> TensorValue:
        """
        Args:
            x: a tensor of shape (batch_size, depth, height, width, in_channels)

        Returns:
             a tensor of shape (batch_size, new_depth, new_height, new_width, out_channels)
        """
        # These need to be casted as the underlying ops.conv3d call
        # expects them to only be tuple types.
        if isinstance(self.stride, int):
            self.stride = (self.stride, self.stride, self.stride)

        if isinstance(self.padding, int):
            self.padding = (
                self.padding,
                self.padding,
                self.padding,
                self.padding,
                self.padding,
                self.padding,
            )

        if isinstance(self.dilation, int):
            self.dilation = (self.dilation, self.dilation, self.dilation)

        if (
            isinstance(self.filter, Weight)
            and self.filter.quantization_encoding is not None
        ):
            raise ValueError("Conv3D not implemented with weight quantization.")
        return ops.conv3d(
            x,
            self.filter,
            self.stride,
            self.dilation,
            self.padding,
            self.groups,
            self.bias,
        )


class Conv3D(Module):
    """A 3D convolution over an input signal composed of several input
    planes.

    Example:
        .. code-block:: python

            conv = nn.Conv3D(
                depth=,
                height=,
                width=,
                in_channels=,
                out_channels=,
                dtype=DType.float32,
                stride=1,
                padding=0,
                has_bias=False,
                name="conv3d_weight",
                device=DeviceRef.GPU(),
            )
    """

    device: DeviceRef | None
    """The device where matrix operations are performed."""

    filter: Weight
    """The weight matrix stored on CPU with shape (depth, height, width, in_channels / num_groups, out_channels).
    Model init moves the weight to :obj:`device`."""

    stride: tuple[int, int, int]
    """Controls the stride for the cross-correlation. """

    padding: tuple[int, int, int, int, int, int]
    """Controls the amount of padding applied before and after the input for depth, height, and width dimensions."""

    dilation: tuple[int, int, int]
    """Controls the dilation rate for depth, height, and width dimensions."""

    num_groups: int
    """Number of blocked connections from input channels to output channels."""

    bias: Weight | None = None
    """The optional bias vector stored on CPU with shape (out_channels,).
    Model init moves the bias to :obj:`device` if present."""

    permute: bool = False
    """bool controls whether self.filter is permuted from PyTorch order to max order.
    PyTorch order is: (out_channels, in_channels / num_groups, depth, height, width)
    Max API order: (depth, height, width, in_channels / num_groups, out_channels). """

    def __init__(
        self,
        depth: int,
        height: int,
        width: int,
        in_channels: int,
        out_channels: int,
        dtype: DType,
        stride: int | tuple[int, int, int] = 1,
        padding: int | tuple[int, int, int, int, int, int] = 0,
        dilation: int | tuple[int, int, int] = 1,
        num_groups: int = 1,
        device: DeviceRef | None = None,
        has_bias: bool = False,
        permute: bool = False,
        name: str | None = None,
    ) -> None:
        """Initializes the Conv3D layer with weights and optional bias.

        Args:
            depth: Depth dimension of the convolution kernel (kernel_size[0]).
            height: Height dimension of the convolution kernel (kernel_size[1]).
            width: Width dimension of the convolution kernel (kernel_size[2]).
            in_channels: Number of channels in the input image.
            out_channels: Number of channels produced by the convolution.
            dtype: The data type for both weights and bias.
            stride: Stride of the convolution for depth, height, and width dimensions.
                Can be int (applied to all dimensions) or tuple of 3 ints. Default: 1
            padding: Padding added to all six sides of the input in order:
                (pad_front, pad_back, pad_top, pad_bottom, pad_left, pad_right).
                Can be int (applied to all sides) or tuple of 6 ints. Default: 0
            dilation: Spacing between kernel elements for depth, height, and width dimensions.
                Can be int (applied to all dimensions) or tuple of 3 ints. Default: 1
            num_groups: Number of blocked connections from input channels to output channels.
                Input channels and output channels are divided into groups. Default: 1.
            device: The target device for computation. If None, defaults to CPU.
                Weights are initially stored on CPU and moved to target device during computation.
            name: Base name for weights. If provided, weights are named ``{name}.weight`` and
                ``{name}.bias`` (if bias is enabled). If None, uses "weight" and "bias".
            has_bias: If true, adds a learnable bias vector to the layer.
                Defaults to :obj:`False`.
            permute: If true, permutes weights from PyTorch format to MAX format.
                PyTorch order: (out_channels, in_channels / num_groups, depth, height, width).
                MAX API order: (depth, height, width, in_channels / num_groups, out_channels).
                Defaults to :obj:`False`.
        """
        super().__init__()

        self.device = device

        self.permute = permute

        if self.permute:
            self.filter = Weight(
                name=f"{name}.weight" if name else "weight",
                dtype=dtype,
                shape=[
                    out_channels,
                    in_channels // num_groups,
                    depth,
                    height,
                    width,
                ],
                device=self.device or DeviceRef.CPU(),
            )
        else:
            self.filter = Weight(
                name=f"{name}.weight" if name else "weight",
                dtype=dtype,
                shape=[
                    depth,
                    height,
                    width,
                    in_channels // num_groups,
                    out_channels,
                ],
                device=self.device or DeviceRef.CPU(),
            )

        if has_bias:
            self.bias = Weight(
                name=f"{name}.bias" if name else "bias",
                dtype=dtype,
                shape=(out_channels,),
                device=self.device or DeviceRef.CPU(),
            )
        # These need to be casted as the underlying ops.conv3d call
        # expects them to only be tuple types.
        if isinstance(stride, int):
            stride = (stride, stride, stride)
        self.stride = stride

        if isinstance(padding, int):
            padding = (
                padding,
                padding,
                padding,
                padding,
                padding,
                padding,
            )
        self.padding = padding

        if isinstance(dilation, int):
            dilation = (dilation, dilation, dilation)
        self.dilation = dilation

        self.num_groups = num_groups

        if (
            isinstance(self.filter, Weight)
            and self.filter.quantization_encoding is not None
        ):
            raise ValueError("Conv3D not implemented with weight quantization.")

    def __call__(self, x: TensorValue) -> TensorValue:
        """Applied 3D convolution to input `x`. Permutes pytorch weights to match max API if permute=True.

        Args:
            x: a tensor of shape (batch_size, depth, height, width, in_channels)
            if self.permute, then input is of shape: (batch_size, in_channels, depth, height, width)
            and will be permuted to match max's expected input shape.

        Returns:
             a tensor of shape (batch_size, new_depth, new_height, new_width, out_channels).
             if self.permute, then the output shape will be (batch_size, out_channels, new_depth, new_height, new_width)
        """
        weight: TensorValue = self.filter
        if self.permute:
            weight = ops.permute(self.filter, [2, 3, 4, 1, 0])
            x = ops.permute(x, [0, 2, 3, 4, 1])

        res = ops.conv3d(
            x,
            weight,
            self.stride,
            self.dilation,
            self.padding,
            self.num_groups,
            self.bias,
        )
        # permute output from (batch_size, depth, height, width, out_channels) to (batch_size, out_channels, depth, height, width).
        if self.permute:
            res = ops.permute(res, [0, 4, 1, 2, 3])
        return res
