# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
"""Casting ops."""

from max import _graph, mlir
from max.mlir.dialects import rmo

from ..graph import Graph
from ..graph_value import GraphValue
from ..type import ShapeLike, dim


def reshape(x: GraphValue, shape: ShapeLike):
    dims = [dim(d).to_mlir() for d in shape]
    return Graph.current._add_op(
        rmo.reshape, x, new_shape=_graph.shape_attr(mlir.Context.current, dims)
    )[0]
