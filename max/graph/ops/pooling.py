# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
"""Op implementation for pooling (max, avg, etc)."""

from __future__ import annotations

from max.mlir.dialects import rmo

from ..graph import Graph
from ..type import DimLike, Shape
from ..value import TensorValue, TensorValueLike


def avg_pool2d(
    input: TensorValueLike,
    kernel_size: tuple[DimLike, DimLike],
    stride: int | tuple[int, int] = 1,
    dilation: int | tuple[int, int] = 1,
    padding: int | tuple[int, int] = 0,
    ceil_mode: bool = False,
    count_boundary: bool = True,
) -> TensorValue:
    """
    Perform a 2D average pooling operation on the input tensor.

    This function applies a 2D average pooling operation to the input tensor [N, H, W, C].
    The pooling operation slides a window of size `kernel_size` over the input
    tensor, and computes the average value within each window.

    Args:
        input: The input tensor to perform the pooling operation on.
        kernel_size: The size of the sliding blocks.
        stride: The stride of the sliding blocks in the input dimension.
        dilation: The spacing between the kernel elements.
        padding: 0-paddings to be added on both sides of the inputs.
        ceil_mode: If true, use ceil instead of floor to compute the output shape.
        count_boundary: If true, count the padding elements when computing the average.
    """
    input = TensorValue(input)

    if not isinstance(stride, tuple):
        stride = (stride, stride)
    if not isinstance(dilation, tuple):
        dilation = (dilation, dilation)
    if not isinstance(padding, tuple):
        _padding = (padding, padding, padding, padding)
    else:
        _padding = (padding[0], padding[0], padding[1], padding[1])

    return Graph.current._add_op(
        rmo.avg_pool,
        input=input,
        filter_shape=Shape(kernel_size).to_mlir(),
        strides=Shape(stride).to_mlir(),
        dilations=Shape(dilation).to_mlir(),
        paddings=Shape(_padding).to_mlir(),
        ceil_mode=ceil_mode,
        count_boundary=count_boundary,
    )[0].tensor


def max_pool2d(
    input: TensorValueLike,
    kernel_size: tuple[DimLike, DimLike],
    stride: int | tuple[int, int] = 1,
    dilation: int | tuple[int, int] = 1,
    padding: int | tuple[int, int] = 0,
    ceil_mode: bool = False,
) -> TensorValue:
    """
    Perform a 2D max pooling operation on the input tensor.

    This function applies a 2D max pooling operation to the input tensor [N, H, W, C].
    The pooling operation slides a window of size `kernel_size` over the input
    tensor, and selects the maximum value within each window.

    Args:
        input: The input tensor to perform the pooling operation on.
        kernel_size: The size of the sliding blocks.
        stride: The stride of the sliding blocks in the input dimension.
        dilation: The spacing between the kernel elements.
        padding: 0-paddings to be added on both sides of the inputs.
        ceil_mode: If true, use ceil instead of floor to compute the output shape.
    """
    input = TensorValue(input)

    if not isinstance(stride, tuple):
        stride = (stride, stride)
    if not isinstance(dilation, tuple):
        dilation = (dilation, dilation)
    if not isinstance(padding, tuple):
        _padding = (padding, padding, padding, padding)
    else:
        _padding = (padding[0], padding[0], padding[1], padding[1])

    return Graph.current._add_op(
        rmo.max_pool,
        input=input,
        filter_shape=Shape(kernel_size).to_mlir(),
        strides=Shape(stride).to_mlir(),
        dilations=Shape(dilation).to_mlir(),
        paddings=Shape(_padding).to_mlir(),
        ceil_mode=ceil_mode,
    )[0].tensor
