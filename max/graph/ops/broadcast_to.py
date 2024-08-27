# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
"""Op implementation for broadcast_to."""

from max.mlir.dialects import rmo

from ..graph import Graph
from ..value import TensorValue
from ..type import Shape, ShapeLike


def broadcast_to(x: TensorValue, shape: ShapeLike) -> TensorValue:
    """Broadcasts a symbolic tensor.

    Broadcasts the input tensor to the specified shape.
    Dimensions in the input must be one or match the target dimension.

    Args:
        x: The input symbolic tensor to broadcast.
            This tensor may not contain any dynamic dimensions.
        shape: The new shape as a list of dimensions.
            Dynamic dimensions are not allowed.
        location: An optional location for a more specific error message.

    Returns:
        A symbolic tensor with the same elements as the original tensor, but
        in a new shape. Its symbolic shape is the same as :code:`shape`.
    """
    return Graph.current._add_op(
        rmo.broadcast_to, x, new_shape=Shape(shape).to_mlir()
    )[0].tensor
