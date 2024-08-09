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
from ..type import ShapeLike, dim, DType, DimLike, Dim, _is_static_shape
from .casting import reshape


def as_interleaved_complex(x: ValueLike) -> GraphValue:
    g = GraphValue(x)
    shape = g.tensor_type.shape
    if not _is_static_shape(shape):
        raise TypeError("Shape must be static")
    if shape[-1].dim % 2 != 0:
        raise ValueError("The last dimension must be divisible by 2.")
    new_shape = [x.dim for x in shape[:-1]] + [shape[-1].dim // 2, 2]
    return ops.reshape(g, new_shape)
