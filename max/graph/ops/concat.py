# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
"""Op implementation for concat."""

from typing import Iterable

from max import mlir
from max.mlir.dialects import rmo

from ..graph import Graph
from ..graph_value import GraphValue, ValueLike


def concat(vals: Iterable[ValueLike], axis: int = 0):
    vals = [GraphValue(v) for v in vals]

    if not vals:
        raise ValueError("Must provide at least one value to concat.")

    # TODO: assert that all vals have the same rank

    axis_attr = mlir.IntegerAttr.get(mlir.IndexType.get(), axis)

    return Graph.current._add_op(rmo.concat, vals, axis=axis_attr)[0]
