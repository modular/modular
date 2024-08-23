# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
"""Op implementation for cast."""

from max.dtype import DType
from max.mlir.dialects import mo

from ..graph import Graph
from ..graph_value import GraphValue, ValueLike


def cast(x: ValueLike, dtype: DType):
    gv = GraphValue(x)
    if gv.tensor_type.dtype == dtype:
        return gv
    return Graph.current._add_op(
        mo.cast, gv.tensor_type.cast(dtype).to_mlir(), gv
    )[0]
