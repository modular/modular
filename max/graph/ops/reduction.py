# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
"""Reduction ops."""

from max.dtype import DType
from max.mlir.dialects import rmo

from ..graph import Graph
from ..type import Dim, Shape, TensorType
from ..value import TensorValue, TensorValueLike
from .constant import constant


def mean(x: TensorValueLike, axis=-1) -> TensorValue:
    """
    Reduces a symbolic tensor using a mean operation.

    Args:
        x: The input tensor for the operation.
        axis: The axis along which to compute the reduction. If negative,
            indexes from the last dimension. For example, a value of -1 will
            compute the reduction along the last dimension.

    Returns:
        A symbolic tensor representing the result of the mean operation.
        The tensor will have the same rank as the input tensor, and the same
        shape except along the ``axis`` dimension which will have size 1.
    """
    gv = TensorValue(x)
    shape = Shape(gv.shape)
    shape[axis] = Dim(1)
    type = TensorType(gv.dtype, shape, gv.device)
    return Graph.current._add_op(
        rmo.mo_mean, type.to_mlir(), gv, constant(axis, DType.int64)
    )[0].tensor


def argmax(x: TensorValueLike, axis=-1) -> TensorValue:
    """
    Reduces a symbolic tensor using an argmax operation.

    Args:
        x: The input tensor for the operation.
        axis: The axis along which to compute the reduction. If negative,
            indexes from the last dimension. For example, a value of -1 will
            compute the reduction along the last dimension.

    Returns:
        A symbolic tensor representing the result of the argmax operation.
        The tensor will have the same rank as the input tensor, and the same
        shape except along the ``axis`` dimension which will have size 1.
    """
    x = TensorValue(x)

    if axis < 0:
        axis += x.rank
    if not 0 <= axis < x.rank:
        raise ValueError(f"Invalid {axis=} for input {x.rank=}")

    shape = Shape(x.shape)
    shape[axis] = Dim(1)
    type = TensorType(DType.int64, shape, x.device)
    return Graph.current._add_op(
        rmo.mo_arg_max, type.to_mlir(), x, constant(axis, DType.int64)
    )[0].tensor
