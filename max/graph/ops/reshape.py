# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
"""Op implementation for reshape."""

from max.mlir.dialects import rmo

from ..graph import Graph
from ..graph_value import GraphValue, ValueLike
from ..type import Shape, ShapeLike


def reshape(x: ValueLike, shape: ShapeLike):
    return Graph.current._add_op(
        rmo.reshape, GraphValue(x), new_shape=Shape(shape).to_mlir()
    )[0]
