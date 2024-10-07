# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
"""Op implementation for shape_to_tensor."""

from max.mlir.dialects import rmo

from ..graph import Graph
from ..type import Shape, ShapeLike
from ..value import TensorValue


def shape_to_tensor(shape: ShapeLike) -> TensorValue:
    return Graph.current._add_op(
        rmo.shape_to_tensor, shape=Shape(shape).to_mlir()
    )[0].tensor
