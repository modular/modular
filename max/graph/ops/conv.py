# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
"""Op implementation for conv2d."""

from typing import Optional, Tuple

import numpy as np
from max.dtype import DType
from max.mlir.dialects import rmo

from .. import dtype_promotion
from ..graph import Graph
from ..type import Dim, Shape
from ..value import TensorType, TensorValue, TensorValueLike
from .constant import constant


def _calc_output_dimensions(
    input_shape: Shape,
    filter_shape: Tuple[Dim, Dim],
    padding: Tuple[int, int, int, int],
    stride: Tuple[int, int],
    dilation: Tuple[int, int],
) -> Shape:
    """Computes the final dimensions of a rank-4 tensor after a 2-D convolution.

    When a 2-D filter is applied to a tensor, the final height and width
    dimensions are inset by the size of the filter, minus one, and then grown
    by the padding applied before and after each dimension.

    Args:
        g: The Graph instance.
        input: The shape of the rank-4 input tensor.
        filter_shape: The shape of the filter being applied.
        padding: The padding to apply to the height and width of the input
                 tensor. Values are provided in (before_height, after_height,
                 before_width, after_width) ordering.
        stride: The stride of the filter, in (height, width) ordering.

    Returns:
        The resized shape of the tensor.
    """
    output_shape = input_shape  # output_shape[0] = input_shape[0] = batch_size

    input_height, filter_height = input_shape[1], filter_shape[0]
    output_shape[1] = (
        input_height - filter_height + padding[0] + padding[1] + stride[0]
    ) // stride[0]

    input_width, filter_width = input_shape[2], filter_shape[1]
    output_shape[2] = (
        input_width - filter_width + padding[2] + padding[3] + stride[1]
    ) // stride[1]
    return output_shape


def conv2d(
    x: TensorValueLike,
    filter: TensorValueLike,
    stride: Tuple[int, int] = (1, 1),
    dilation: Tuple[int, int] = (1, 1),
    padding: Tuple[int, int, int, int] = (0, 0, 0, 0),
    groups: int = 1,
    bias: Optional[TensorValueLike] = None,
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
        if x.dtype != bias.dtype:
            raise ValueError(
                "input and bias must resolve to the same strong dtype. input is"
                f" {x.dtype}. bias is {bias.dtype}."
            )
        if bias.rank != 1:
            raise ValueError(
                "bias for a 2-D convolution must be rank 1 with shape (out_channels,)"
            )
    if x.dtype != filter.dtype:
        raise ValueError(
            "input and filter must resolve to the same strong dtype. input is"
            f" {x.dtype}. filter is {filter.dtype}."
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
    if dilation != (1, 1):
        raise ValueError("Non-unit dilation is not implemented yet!")

    stride_constant = constant(np.array([stride[0], stride[1]]), DType.int64)

    dilation_constant = constant(
        np.array([dilation[0], dilation[1]]), DType.int64
    )

    padding_constant = constant(
        np.array([padding[0], padding[1], padding[2], padding[3]]),
        DType.int64,
    )

    groups_constant = constant(groups, DType.int64)

    # TODO(MSDK-1196): create rmo.conv and move this to rmo
    output_shape = _calc_output_dimensions(
        x.shape,
        filter_shape=(filter.shape[0], filter.shape[1]),
        padding=padding,
        stride=stride,
        dilation=dilation,
    )
    output_shape[3] = filter.shape[3]  # out_channels

    conv_output = Graph.current._add_op(
        rmo.mo_conv,
        TensorType(x.dtype, output_shape, x.device).to_mlir(),
        x,
        filter,
        stride_constant,
        dilation_constant,
        padding_constant,
        groups_constant,
    )[0].tensor

    if bias is not None:
        return Graph.current._add_op(rmo.add, conv_output, bias)[0].tensor
    return conv_output


def conv3d(
    x: TensorValueLike,
    filter: TensorValueLike,
    stride: Tuple[int, int, int] = (1, 1, 1),
    dilation: Tuple[int, int, int] = (1, 1, 1),
    padding: Tuple[int, int, int, int, int, int] = (0, 0, 0, 0, 0, 0),
    groups: int = 1,
    bias: Optional[TensorValueLike] = None,
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
    0's before and after the indicated *spatial* dimensions in `input`. In 2-D
    convolution, dim1 here repesents H and dim2 represents W. In Python like
    syntax, padding a 2x3 spatial `input` with [0, 1, 2, 1] would yield:

    ```python
    input = [
      [1, 2, 3],
      [4, 5, 6]
    ]
    # Shape is 2x3

    padded_input = [
      [0, 0, 1, 2, 3, 0],
      [0, 0, 4, 5, 6, 0]
      [0, 0, 0, 0, 0, 0]
    ]
    # Shape is 3x6
    ```

    This op currently only supports strides and padding on the input.

    Args:
        input: An NDHWC input tensor to perform the convolution upon.
        filter: The convolution filter in RSCF layout:
                (height, depth, width, in_channels / num_groups, out_channels).
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
        if x.dtype != bias.dtype:
            raise ValueError(
                "input and bias must resolve to the same strong dtype. input is"
                f" {x.dtype}. bias is {bias.dtype}."
            )
        if bias.rank != 1:
            raise ValueError(
                "bias for a 2-D convolution must be rank 1 with shape (out_channels,)"
            )
    if x.dtype != filter.dtype:
        raise ValueError(
            "input and filter must resolve to the same strong dtype. input is"
            f" {x.dtype}. filter is {filter.dtype}."
        )
    if x.rank != 5:
        raise ValueError(
            "input to a 3-D convolution must be rank 4 with shape (batch_size,"
            " depth, height, width, in_channels)"
        )
    if filter.rank != 5:
        raise ValueError(
            "filter for a 3-D convolution must be rank 4 with shape (depth,"
            " height, width, in_channels / num_groups, out_channels)"
        )
    if dilation != (1, 1, 1):
        raise ValueError("Non-unit dilation is not implemented yet!")

    stride_constant = constant(
        np.array([stride[0], stride[1], stride[2]]), DType.int64
    )

    dilation_constant = constant(
        np.array([dilation[0], dilation[1], dilation[2]]), DType.int64
    )

    padding_constant = constant(
        np.array(
            [
                padding[0],
                padding[1],
                padding[2],
                padding[3],
                padding[4],
                padding[5],
            ]
        ),
        DType.int64,
    )

    groups_constant = constant(groups, DType.int64)

    # TODO(MSDK-1196): create rmo.conv and move this to rmo
    output_shape = x.shape

    input_depth, filter_depth = x.shape[1], filter.shape[0]
    output_shape[1] = (
        input_depth - filter_depth + padding[0] + padding[1] + stride[0]
    ) // stride[0]

    input_height, filter_height = x.shape[2], filter.shape[1]
    output_shape[2] = (
        input_height - filter_height + padding[2] + padding[3] + stride[1]
    ) // stride[1]

    input_width, filter_width = x.shape[3], filter.shape[2]
    output_shape[3] = (
        input_width - filter_width + 1 + padding[4] + padding[5] + stride[2]
    ) // +stride[2]

    output_shape[4] = filter.shape[4]

    conv_output = Graph.current._add_op(
        rmo.mo_conv,
        TensorType(x.dtype, output_shape, x.device).to_mlir(),
        x,
        filter,
        stride_constant,
        dilation_constant,
        padding_constant,
        groups_constant,
    )[0].tensor

    if bias is not None:
        return Graph.current._add_op(rmo.add, conv_output, bias)[0].tensor
    return conv_output
