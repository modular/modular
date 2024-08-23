# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
"""Op implementation for broadcast_to."""

from max.mlir.dialects import rmo

from ..graph import Graph
from ..graph_value import GraphValue
from ..type import Shape, ShapeLike


def broadcast_to(x: GraphValue, shape: ShapeLike):
    return Graph.current._add_op(
        rmo.broadcast_to, x, new_shape=Shape(shape).to_mlir()
    )[0]
