# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
"""Complex ops."""

from ..graph_value import GraphValue, ValueLike, ops


def as_interleaved_complex(x: ValueLike) -> GraphValue:
    g = GraphValue(x)
    *dims, last_dim = g.tensor_type.shape
    if not isinstance(last_dim, int):
        raise ValueError(f"Last dim must be static: {last_dim}")
    if last_dim % 2 != 0:
        raise ValueError("The last dimension must be divisible by 2.")
    return ops.reshape(g, dims + [last_dim // 2, 2])
