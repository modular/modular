# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
"""Op implementation for rebind."""

from max import mlir
from max.mlir.dialects import rmo

from ..graph import Graph
from ..value import TensorValue, TensorType, ValueLike
from ..type import ShapeLike


def rebind(x: ValueLike, shape: ShapeLike, message: str) -> TensorValue:
    # TODO(MSDK-662): Add checks to ensure that statically known dims are
    # rebound in a way to keep the size the same.
    v = TensorValue(x)
    message_attr = mlir.StringAttr.get(message)
    return Graph.current._add_op(
        rmo.rebind_tensor_shape,
        TensorType(v.dtype, shape).to_mlir(),
        v,
        message=message_attr,
    )[0].tensor
