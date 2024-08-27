# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
"""Op implementation for select."""

from max.mlir.dialects import rmo

from ..graph import Graph
from ..value import TensorValue, ValueLike


def select(cond: ValueLike, x: ValueLike, y: ValueLike) -> TensorValue:
    return Graph.current._add_op(
        rmo.select, TensorValue(cond), TensorValue(x), TensorValue(y)
    )[0].tensor
