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
from typing import Optional, Tuple, Union

from max.graph import TensorValue, TensorValueLike, Weight, ops

from .layer import Layer


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

    stride: Union[int, Tuple[int, int]] = (1, 1)
    padding: Union[int, Tuple[int, int, int, int]] = (0, 0, 0, 0)
    dilation: Union[int, Tuple[int, int]] = (1, 1)
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

    stride: Union[int, Tuple[int, int, int]] = (1, 1, 1)
    padding: Union[int, Tuple[int, int, int, int, int, int]] = (
        0,
        0,
        0,
        0,
        0,
        0,
    )
    dilation: Union[int, Tuple[int, int, int]] = (1, 1, 1)
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
