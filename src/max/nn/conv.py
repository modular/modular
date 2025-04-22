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

from dataclasses import dataclass
from typing import Optional, Union

from max.dtype import DType
from max.graph import DeviceRef, TensorValue, TensorValueLike, Weight, ops

from .layer import Layer, Module


@dataclass
class Conv2D(Layer):
    """A 2D convolution over an input signal composed of several input
    planes.

    Example:
        .. code-block:: python

            conv = nn.Conv2D(
                filter=filter_2d,
                bias=bias_2d,
                stride=2,
                padding=1
            )
            output = conv(x)
    """

    filter: TensorValueLike
    bias: Optional[TensorValueLike] = None

    stride: Union[int, tuple[int, int]] = (1, 1)
    padding: Union[int, tuple[int, int, int, int]] = (0, 0, 0, 0)
    dilation: Union[int, tuple[int, int]] = (1, 1)
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
            raise ValueError("Conv1D not implemented with weight quantization.")
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
class Conv1D(Layer):
    """A 1D convolution over an input signal composed of several input
    planes.

    Example:
        .. code-block:: python

            conv = nn.Conv1D(
                filter=filter_1d,
                bias=bias_1d,
                stride=1,
                padding=1
            )
    """

    filter: TensorValueLike  # [kernel_size, in_channels, out_channels]
    bias: Optional[TensorValueLike] = None

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
        # TODO(GEX-327): Support Conv1D in mo rather than implementing it using Conv2D.
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


@dataclass
class Conv3D(Layer):
    """A 3D convolution over an input signal composed of several input
    planes.

    Example:
        .. code-block:: python

            conv = nn.Conv3D(
                filter=filter_3d,
                bias=bias_3d,
                stride=1,
                padding=1
            )
    """

    filter: TensorValueLike  # [depth, height, width, in_channels / num_groups, out_channels]
    bias: Optional[TensorValueLike] = None  # [out_channels]

    stride: Union[int, tuple[int, int, int]] = (1, 1, 1)
    padding: Union[int, tuple[int, int, int, int, int, int]] = (
        0,
        0,
        0,
        0,
        0,
        0,
    )
    dilation: Union[int, tuple[int, int, int]] = (1, 1, 1)
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


class Conv3DV2(Module):
    """A 3D convolution over an input signal composed of several input
    planes.

    Example:
        .. code-block:: python

            conv = nn.Conv3DV2(
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

    device: Union[DeviceRef, None]
    """The device where matrix operations are performed."""

    filter: Weight
    """The weight matrix stored on CPU with shape (depth, height, width, in_channels / num_groups, out_channels).
    Model init moves the weight to :obj:`device`."""

    stride: tuple[int, int, int]
    """Controls the stride for the cross-correlation. """

    padding: tuple[int, int, int, int, int, int]
    """Controls the amount of padding applied before and after the input for depth, height, and width dimensions."""

    dilation: tuple[int, int, int]
    """Not implemented yet. Assuming dilation = 1 for now."""

    num_groups: int
    """Not implemented yet. Assuming num_groups = 1 for now."""

    bias: Union[Weight, None] = None
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
        stride: Union[int, tuple[int, int, int]] = 1,
        padding: Union[int, tuple[int, int, int, int, int, int]] = 0,
        dilation: Union[int, tuple[int, int, int]] = 1,
        num_groups: int = 1,
        device: Union[DeviceRef, None] = None,
        has_bias: bool = False,
        permute: bool = False,
        name: Union[str, None] = None,
    ) -> None:
        """Initializes the Conv3D layer with weights and optional bias.

        Args:
            depth: kernel_size[0]
            height: kernel_size[1]
            width: kernel_size[2]
            in_channels: number of channels in the input image.
            out_channels: dimensionality of the output.
            dtype: The data type for both weights and bias.
            stride: Stride of the convolution. Default: 1
            padding:  Padding added to all six sides of the input. Default: 0
            dilation: Spacing between kernel elements. Default: 1
            num_groups:  Number of blocked connections from input channels to output channels. Default: 1.
            device: The target device for computation.
                Weights remain on CPU until moved during computation.
            name: Base name for weights (appended with ``.weight`` and
                ``.bias`` if applicable).
            has_bias: When :obj:`True`, adds a bias vector to the layer.
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
        # permute output from (batch_size, depth, height, width, out_channels) (batch_size, out_channels, depth, height, width).
        if self.permute:
            res = ops.permute(res, [0, 4, 1, 2, 3])
        return res
