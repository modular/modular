# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
"""Op implementation for reshape."""

from max.mlir.dialects import rmo

from ..graph import Graph
from ..value import TensorValue, ValueLike
from ..type import Shape, ShapeLike


def reshape(x: ValueLike, shape: ShapeLike) -> TensorValue:
    return Graph.current._add_op(
        rmo.reshape, TensorValue(x), new_shape=Shape(shape).to_mlir()
    )[0].tensor
