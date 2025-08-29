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
"""Op implementation for conv2d."""

from typing import Optional

from max.mlir.dialects import rmo

from .. import dtype_promotion
from ..graph import Graph
from ..type import ConvInputLayout, FilterLayout, Shape
from ..value import TensorValue, TensorValueLike


def conv2d(
    x: TensorValueLike,
    filter: TensorValueLike,
    stride: tuple[int, int] = (1, 1),
    dilation: tuple[int, int] = (1, 1),
    padding: tuple[int, int, int, int] = (0, 0, 0, 0),
    groups: int = 1,
    bias: Optional[TensorValueLike] = None,
    input_layout: ConvInputLayout = ConvInputLayout.NHWC,
    filter_layout: FilterLayout = FilterLayout.RSCF,
) -> TensorValue:
    """Computes the 2-D convolution product of the input with the given filter, bias,
    strides, dilations, paddings, and groups.

    The op supports 2-D convolution, with the following layout assumptions:

    - input `x` has NHWC layout, i.e.,
      (batch_size, height, width, in_channels)
    - filter has layout RSCF, i.e.,
      (height, width, in_channels / num_groups, out_channels)
    - bias has shape (out_channels,)

    The padding values are expected to take the form (pad_dim1_before,
    pad_dim1_after, pad_dim2_before, pad_dim2_after...) and represent padding
    0's before and after the indicated *spatial* dimensions in `input`. In 2-D
    convolution, dim1 here represents H and dim2 represents W. In Python like
    syntax, padding a 2x3 spatial `input` with [0, 1, 2, 1] would yield:

    .. code-block:: python

        input = [
          [1, 2, 3],
          [4, 5, 6]
        ]
        # Shape is 2x3

        padded_input = [
          [0, 0, 1, 2, 3, 0],
          [0, 0, 4, 5, 6, 0],
          [0, 0, 0, 0, 0, 0]
        ]
        # Shape is 3x6

    This op currently only supports strides and padding on the input.

    Args:
        input: An NHWC input tensor to perform the convolution upon.
        filter: The convolution filter in RSCF layout:
                (height, width, in_channels / num_groups, out_channels).
        stride: The stride of the convolution operation.
        dilation: The spacing between the kernel points.
        padding: The amount of padding applied to the input.
        groups: When greater than 1, divides the convolution into multiple
                parallel convolutions. The number of input and output
                channels must both be divisible by the number of groups.

    Returns:
        A symbolic tensor value with the convolution applied.
    """
    x, filter = dtype_promotion._promote_weak_dtypes(x, filter)

    if bias is not None:
        x, bias = dtype_promotion._promote_weak_dtypes(x, bias)

        if bias.rank != 1:
            raise ValueError(
                "bias for a 2-D convolution must be rank 1 with shape (out_channels,)"
            )

    if x.rank != 4:
        raise ValueError(
            "input to a 2-D convolution must be rank 4 with shape (batch_size,"
            " height, width, in_channels)"
        )

    if filter.rank != 4:
        raise ValueError(
            "filter for a 2-D convolution must be rank 4 with shape (height,"
            " width, in_channels / num_groups, out_channels)"
        )

    conv_output = Graph.current._add_op(
        rmo.conv,
        x,
        filter._with_layout(filter_layout),
        Shape(stride).to_mlir(),
        Shape(dilation).to_mlir(),
        Shape(padding).to_mlir(),
        groups,
        input_layout=input_layout.to_mlir(),
    )[0].tensor

    if bias is not None:
        return Graph.current._add_op(rmo.add, conv_output, bias)[0].tensor
    return conv_output


def conv3d(
    x: TensorValueLike,
    filter: TensorValueLike,
    stride: tuple[int, int, int] = (1, 1, 1),
    dilation: tuple[int, int, int] = (1, 1, 1),
    padding: tuple[int, int, int, int, int, int] = (0, 0, 0, 0, 0, 0),
    groups: int = 1,
    bias: Optional[TensorValueLike] = None,
    input_layout: ConvInputLayout = ConvInputLayout.NHWC,
    filter_layout: FilterLayout = FilterLayout.QRSCF,
) -> TensorValue:
    """Computes the 3-D convolution product of the input with the given filter,
    strides, dilations, paddings, and groups.

    The op supports 3-D convolution, with the following layout assumptions:

    - input has NDHWC layout, i.e.,
      (batch_size, depth, height, width, in_channels)
    - filter has layout RSCF, i.e.,
      (depth, height, width, in_channels / num_groups, out_channels)

    The padding values are expected to take the form (pad_dim1_before,
    pad_dim1_after, pad_dim2_before, pad_dim2_after...) and represent padding
    0's before and after the indicated *spatial* dimensions in `input`. In 3-D
    convolution, dim1 here represents D, dim2 represents H and dim3 represents W. In Python like
    syntax, padding a 2x3 spatial `input` with [0, 1, 2, 1] would yield:

    .. code-block:: python

        input = [
          [1, 2, 3],
          [4, 5, 6]
        ]
        # Shape is 2x3

        padded_input = [
          [0, 0, 1, 2, 3, 0],
          [0, 0, 4, 5, 6, 0],
          [0, 0, 0, 0, 0, 0]
        ]
        # Shape is 3x6

    This op currently only supports strides and padding on the input.

    Args:
        x: An NDHWC input tensor to perform the convolution upon.
        filter: The convolution filter in RSCF layout:
                (depth, height, width, in_channels / num_groups, out_channels).
        stride: The stride of the convolution operation.
        dilation: The spacing between the kernel points.
        padding: The amount of padding applied to the input.
        groups: When greater than 1, divides the convolution into multiple
                parallel convolutions. The number of input and output
                channels must both be divisible by the number of groups.

    Returns:
        A symbolic tensor value with the convolution applied.
        Output shape = (batch_size, depth, height, width, out_channels).
    """
    x, filter = dtype_promotion._promote_weak_dtypes(x, filter)

    if bias is not None:
        x, bias = dtype_promotion._promote_weak_dtypes(x, bias)

        if bias.rank != 1:
            raise ValueError(
                "bias for a 2-D convolution must be rank 1 with shape (out_channels,)"
            )

    if x.rank != 5:
        raise ValueError(
            "input to a 3-D convolution must be rank 5 with shape (batch_size,"
            " depth, height, width, in_channels)"
        )

    if filter.rank != 5:
        raise ValueError(
            "filter for a 3-D convolution must be rank 5 with shape (depth,"
            " height, width, in_channels / num_groups, out_channels)"
        )

    conv_output = Graph.current._add_op(
        rmo.conv,
        x,
        filter._with_layout(filter_layout),
        Shape(stride).to_mlir(),
        Shape(dilation).to_mlir(),
        Shape(padding).to_mlir(),
        groups,
        input_layout=input_layout.to_mlir(),
    )[0].tensor

    if bias is not None:
        return Graph.current._add_op(rmo.add, conv_output, bias)[0].tensor
    return conv_output
