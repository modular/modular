# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
"""Op implementation for concat."""

from collections.abc import Iterable

from max import mlir
from max.mlir.dialects import rmo

from ..graph import Graph
from ..value import TensorValue, TensorValueLike


def concat(
    original_vals: Iterable[TensorValueLike],
    axis: int = 0,
) -> TensorValue:
    """Concatenates a list of symbolic tensors along an axis.

    Args:
        original_vals: A list of symbolic tensor values. Each tensor must have the same
            dtype and rank, and must have the same dimension size for each
            dimension other than ``axis``.
        axis: The axis to concatenate along. If negative, indexes relative
            to the end of the tensor shape. For instance, ``concat(vs, -1)``
            will concat along the last dimension.

    Returns:
        A new symbolic tensor representing the concatenation result. It will
        have the same rank as each input tensor, and its dimensions will be the same
        as each input tensor's for each dimension other than `axis`, which will
        have size equal to the sum of all tensor's size for that dimension.
    """
    vals = [TensorValue(v) for v in original_vals]

    if not vals:
        raise ValueError("Must provide at least one value to concat.")
    if not all(val.rank == vals[0].rank for val in vals):
        raise ValueError(f"Concat inputs must all have the same rank. {vals=}")
    if not -vals[0].rank <= axis < vals[0].rank:
        raise IndexError(f"Axis out of range {axis=}, {vals=}")
    for i, dim in enumerate(vals[0].shape):
        if i in (axis, axis + vals[0].rank):
            continue
        if not all(val.shape[i] == dim for val in vals):
            raise ValueError(
                f"Concat inputs differ on non-concat axis {i}: {vals=}"
            )
    if any(v.type.device != vals[0].type.device for v in vals):
        raise ValueError(f"Cannot concat inputs on different devices {vals}")
    axis_attr = mlir.IntegerAttr.get(mlir.IndexType.get(), axis)

    result = Graph.current._add_op(rmo.concat, vals, axis=axis_attr)[0].tensor

    return result
