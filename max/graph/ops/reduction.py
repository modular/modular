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


def sum(x: TensorValueLike, axis=-1) -> TensorValue:
    """
    Reduces a symbolic tensor using a sum operation.

    Args:
        x: The input tensor for the operation.
        axis: The axis along which to compute the reduction. If negative,
            indexes from the last dimension. For example, a value of -1 will
            compute the reduction along the last dimension.

    Returns:
        A symbolic tensor representing the result of the sum operation.
        The tensor will have the same rank as the input tensor, and the same
        shape except along the ``axis`` dimension which will have size 1.
    """
    return _reduce(rmo.mo_reduce_add, x, axis=axis)


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
    return _reduce(rmo.mo_mean, x, axis=axis)


def min(x: TensorValueLike, axis=-1) -> TensorValue:
    """
    Reduces a symbolic tensor using a min operation.

    Args:
        x: The input tensor for the operation.
        axis: The axis along which to compute the reduction. If negative,
            indexes from the last dimension. For example, a value of -1 will
            compute the reduction along the last dimension.

    Returns:
        A symbolic tensor representing the result of the min operation.
        The tensor will have the same rank as the input tensor, and the same
        shape except along the ``axis`` dimension which will have size 1.
    """
    return _reduce(rmo.mo_reduce_min, x, axis=axis)


def max(x: TensorValueLike, axis=-1) -> TensorValue:
    """
    Reduces a symbolic tensor using a max operation.

    Args:
        x: The input tensor for the operation.
        axis: The axis along which to compute the reduction. If negative,
            indexes from the last dimension. For example, a value of -1 will
            compute the reduction along the last dimension.

    Returns:
        A symbolic tensor representing the result of the max operation.
        The tensor will have the same rank as the input tensor, and the same
        shape except along the ``axis`` dimension which will have size 1.
    """
    return _reduce(rmo.mo_reduce_max, x, axis=axis)


def _reduce(op, x: TensorValueLike, axis=-1, out_dtype=None) -> TensorValue:
    """
    Reduces a symbolic tensor using a reduction operation.

    Args:
        x: The input tensor for the operation.
        axis: The axis along which to compute the reduction. If negative,
            indexes from the last dimension. For example, a value of -1 will
            compute the reduction along the last dimension.
        out_dtype: The dtype of the result. Defaults to the dtype of `x`.

    Returns:
        A symbolic tensor representing the result of the argmin or argmax operation.
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
    type = TensorType(out_dtype or x.dtype, shape, x.device)
    return Graph.current._add_op(
        op, type.to_mlir(), x, constant(axis, DType.int64)
    )[0].tensor


def argmin(x: TensorValueLike, axis=-1) -> TensorValue:
    """
    Reduces a symbolic tensor using an argmin operation.

    When provided with a tensor with all identical elements,
    on CPU this will return the first element index in the tensor,
    on GPU this will return an arbitrary index.

    Args:
        x: The input tensor for the operation.
        axis: The axis along which to compute the reduction. If negative,
            indexes from the last dimension. For example, a value of -1 will
            compute the reduction along the last dimension.

    Returns:
        A symbolic tensor representing the result of the argmin operation.
        The tensor will have the same rank as the input tensor, and the same
        shape except along the ``axis`` dimension which will have size 1.
    """
    return _reduce(rmo.mo_arg_min, x, axis, out_dtype=DType.int64)


def argmax(x: TensorValueLike, axis=-1) -> TensorValue:
    """
    Reduces a symbolic tensor using an argmax operation.

    When provided with a tensor with all identical elements,
    on CPU this will return the first element index in the tensor,
    on GPU this will return an arbitrary index.

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
    return _reduce(rmo.mo_arg_max, x, axis, out_dtype=DType.int64)
