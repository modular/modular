# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
"""Op implementation for select."""

from max.mlir.dialects import rmo

from .. import dtype_promotion
from ..graph import Graph
from ..value import TensorValue, TensorValueLike


def select(
    cond: TensorValueLike, x: TensorValueLike, y: TensorValueLike
) -> TensorValue:
    """
    Returns ``condition ? x : y`` (element-wise), where ``cond``, ``x`` and ``y``
    are input tensors.

    Args:
        condition: The condition tensor to use for selecting elementwise
                   values.
        x: If the condition is true at a position, the value from the same
           position in this tensor will be selected.
        y: If the condition is false at a position, the value from the same
           position in this tensor will be selected.

    Returns:
        A new symbolic tensor holding either values from either ``x`` or ``y``,
        based on the elements in `condition`.
    """
    x, y = dtype_promotion._promote_weak_dtypes(x, y)
    return Graph.current._add_op(rmo.select, TensorValue(cond), x, y)[0].tensor
