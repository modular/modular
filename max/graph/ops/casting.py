# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
"""Casting ops."""

from max import _graph, mlir
from max.mlir.dialects import rmo, mo

from ..graph import Graph
from ..graph_value import GraphValue, ValueLike
from ..type import ShapeLike, dim, DType


def reshape(x: GraphValue, shape: ShapeLike):
    dims = [dim(d).to_mlir() for d in shape]
    return Graph.current._add_op(
        rmo.reshape, x, new_shape=_graph.shape_attr(mlir.Context.current, dims)
    )[0]


def cast(x: ValueLike, dtype: DType):
    gv = GraphValue(x)
    if gv.tensor_type.dtype == dtype:
        return gv
    return Graph.current._add_op(
        mo.cast, gv.tensor_type.cast(dtype).to_mlir(), gv
    )[0]
