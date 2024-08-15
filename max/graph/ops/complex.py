# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
"""Complex ops."""

from max import _graph, mlir
from max.mlir.dialects import rmo, mo

from ..graph import Graph
from ..graph_value import GraphValue, ValueLike, ops
from ..type import ShapeLike, DType, DimLike, Dim, StaticDim
from .casting import reshape


def as_interleaved_complex(x: ValueLike) -> GraphValue:
    g = GraphValue(x)
    shape = g.tensor_type.shape
    last = shape[-1]
    if not isinstance(last, StaticDim):
        raise TypeError("The last dimension must be static.")
    if last.dim % 2 != 0:
        raise ValueError("The last dimension must be divisible by 2.")
    new_shape = shape[:-1] + [last.dim // 2, 2]
    return ops.reshape(g, new_shape)
