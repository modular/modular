# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
"""Op implementation for cast."""

from max.dtype import DType
from max.mlir.dialects import mo

from ..graph import Graph
from ..value import TensorValue, ValueLike


def cast(x: ValueLike, dtype: DType) -> TensorValue:
    gv = TensorValue(x)
    if gv.dtype == dtype:
        return gv
    return Graph.current._add_op(mo.cast, gv.type.cast(dtype).to_mlir(), gv)[
        0
    ].tensor
