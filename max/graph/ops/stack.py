# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
"""Op implementation for stack."""

from collections.abc import Iterable

from max import mlir
from max.mlir.dialects import rmo

from ..graph import Graph
from ..value import TensorValue, TensorValueLike
from .concat import concat
from .unsqueeze import unsqueeze


def stack(vals: Iterable[TensorValueLike], axis: int = 0) -> TensorValue:
    """Stacks a list of tensors along a new axis.

    Args:
        values: A list of symbolic tensor values. Each tensor must have the same
            dtype and rank, and must have the same dimension size for each
            dimension.
        axis: The axis to concatenate along. If negative, indexes relative
            to the end of the tensor shape *plus 1*. For instance,
            ``stack(vs, -1)`` will create and stack along a new axis as the
            last dimension, aad ``stack(vs, -2)`` will create and stack along a new
            dimension which is inserted immediately before the last dimension.

    Returns:
        A new symbolic tensor representing the result of the stack. It will
        have rank ``n+1`` where ``n`` is the rank of each input tensor. Its size
        on each dimension other than ``axis`` will be the same as each input tensors',
        with the new axis inserted. Along the new dimension it will have size
        ``len(values)``.
    """
    vals_coerced = [TensorValue(v) for v in vals]
    if len(vals_coerced) == 0:
        raise ValueError("Expected at least one value to stack")

    rank = len(vals_coerced[0].shape)
    if any(len(v.shape) != rank for v in vals_coerced):
        raise ValueError("all inputs to stack must be the same rank")

    unsqueezed = [unsqueeze(v, axis) for v in vals_coerced]

    # Short circuit to avoid bloating graph with unneeded op.
    if len(unsqueezed) == 1:
        return unsqueezed[0]

    return concat(unsqueezed, axis=axis)


def stack_scalars(vals: Iterable[TensorValue]) -> TensorValue:
    axis = mlir.IntegerAttr.get(mlir.IndexType.get(), 0)

    vals = [v.reshape([1]) if v.shape != [1] else v for v in vals]
    return Graph.current._add_op(rmo.concat, vals, axis=axis)[0].tensor
