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
from typing import Union

from max.dtype import DType
from max.graph import DeviceRef, TensorValue, Weight, ops

from .layer import Module


@dataclass
class ConvTranspose1d(Module):
    """A 1D transposed convolution operator over an input image composed of several input planes.
    Example:
        .. code-block:: python

            conv = nn.ConvTranspose1d(
                in_channels,
                out_channels,
                kernel_size,
                stride,
                padding,
                output_padding,
                has_bias=False,
                name="conv3d_weight",
                device=DeviceRef.GPU(),
            )
    """

    device: Union[DeviceRef, None]
    """The device where matrix operations are performed."""

    filter: Weight
    """The weight matrix stored on CPU with shape (kernel_length, out_channels, in_channels).
    Model init moves the weight to :obj:`device`."""

    stride: tuple[int, int]
    """Controls the stride for the cross-correlation. """

    padding: tuple[int, int, int, int]
    """Controls the amount of padding applied before and after the input for depth, height, and width dimensions."""

    dilation: tuple[int, int]
    """Not implemented yet. Assuming dilation = 1 for now."""

    output_padding: tuple[int, int]
    """Additional size added to one side of the output shape. Default: 0"""

    permute: bool
    """bool controls whether self.filter is permuted from PyTorch order to max order.
    PyTorch order is: (in_channels, out_channels, kernel_length)
    Max API order: (kernel_length, out_channels, in_channels). """

    bias: Union[Weight, None] = None
    """The optional bias vector stored on CPU with shape (out_channels,).
    Model init moves the bias to :obj:`device` if present."""

    def __init__(
        self,
        length: int,
        in_channels: int,
        out_channels: int,
        dtype: DType,
        stride: Union[int, tuple[int, int]] = 1,
        padding: Union[int, tuple[int, int, int, int]] = 0,
        dilation: Union[int, tuple[int, int]] = 1,
        output_padding: Union[int, tuple[int, int]] = 0,
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
                    in_channels,
                    out_channels,
                    length,
                ],
                device=self.device or DeviceRef.CPU(),
            )
        else:
            self.filter = Weight(
                name=f"{name}.weight" if name else "weight",
                dtype=dtype,
                shape=[
                    length,
                    out_channels,
                    in_channels,
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
            stride = (1, stride)
        self.stride = stride

        if isinstance(output_padding, int):
            output_padding = (0, output_padding)
        self.output_padding = output_padding

        if isinstance(padding, int):
            padding = (
                0,
                0,
                padding,
                padding,
            )
        self.padding = padding

        if isinstance(dilation, int):
            dilation = (1, dilation)
        self.dilation = dilation

        if (
            isinstance(self.filter, Weight)
            and self.filter.quantization_encoding is not None
        ):
            raise ValueError(
                "ConvTranspose1d not implemented with weight quantization."
            )

    def __call__(self, x: TensorValue) -> TensorValue:
        """Applied ConvTranspose1d to input `x`. Permutes pytorch weights to match max API if permute=True.

        Args:
            x: a tensor of shape (batch_size, length, in_channels)
            if self.permute, then input is of shape: (batch_size, in_channels, length)
            and will be permuted to match max's expected input shape.
            Also, self.filter will be permuted from (kernel_length, in_channels, out_channels) to
            (in_channels, out_channels, kernel_length)

        Returns:
             a tensor of shape (batch_size, new_length, out_channels).
             if self.permute, then the output shape will be (batch_size, out_channels, new_length)
        """
        weight: TensorValue = self.filter

        if self.permute:
            # Reshape (batch_size, in_channels, length) to [batch_size, in_channels, height=1, length].
            x = ops.unsqueeze(x, 2)
            # Reshape (in_channels, out_channels, kernel_length) to [in_channels, out_channels, kernel_height=1, kernel_length,].
            weight = ops.unsqueeze(self.filter, 2)
            # [batch_size, in_channels, height=1, length] to (batch_size, height, length, in_channels)
            x = ops.permute(x, [0, 2, 3, 1])
            # (in_channels, out_channels, kernel_height, kernel_length) to [kernel_height=1, kernel_length, out_channels, in_channels]
            weight = ops.permute(weight, [2, 3, 1, 0])
        else:
            # Reshape (batch_size, length, in_channels) to [batch_size, height=1, length, in_channels].
            x = ops.unsqueeze(x, 1)
            # Reshape [kernel_length, in_channels, out_channels] to [kernel_height=1, kernel_length, out_channels, in_channels].
            weight = ops.unsqueeze(weight, 0)

        res = ops.conv2d_transpose(
            x=x,
            filter=weight,
            stride=self.stride,
            dilation=self.dilation,
            padding=self.padding,
            output_paddings=self.output_padding,
            bias=self.bias,
        )

        if self.permute:
            # permute output from [batch_size, height=1, new_length, out_channels] to (batch_size, out_channels, height=1, new_length).
            res = ops.permute(res, [0, 3, 1, 2])
            # Reshape  (batch_size, out_channels, height=1, new_length). to [batch_size, out_channels, new_length].
            res = ops.squeeze(res, 2)
        else:
            # Reshape [batch_size, height=1, new_length, out_channels] to [batch_size, new_length, out_channels].
            res = ops.squeeze(res, 1)
        return res
