# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
"""Casting ops."""

from max import mlir
from max.mlir.dialects import rmo

from .. import core as _c
from ..graph import Graph
from ..graph_value import GraphValue
from ..type import ShapeLike, dim


def reshape(x: GraphValue, shape: ShapeLike):
    dims = [dim(d).to_mlir() for d in shape]
    return Graph.current._add_op(
        rmo.reshape, x, new_shape=_c.shape_attr(mlir.Context.current, dims)
    )[0]
