# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
"""Op implementation for cast."""

from max.dtype import DType
from max.mlir.dialects import mo

from .constant import constant
from ..graph import Graph
from ..value import TensorValue, ValueLike, _strong_value_like


def cast(x: ValueLike, dtype: DType) -> TensorValue:
    """Casts a symbolic tensor to a different data type.

    Args:
        v: The input tensor to cast.
        dtype: The target dtype to which the tensor is cast.

    Returns:
        A new symbolic tensor with the same shape as the input and the
        specified dtype.
    """
    if not isinstance(x, _strong_value_like):
        # This is a weak value. Cast has an explicit target dtype, just create a constant of that dtype.
        return constant(x, dtype)

    # We have a strong valuelike. Actually load and cast it.
    gv = TensorValue(x)
    if gv.dtype == dtype:
        return gv
    return Graph.current._add_op(mo.cast, gv.type.cast(dtype).to_mlir(), gv)[
        0
    ].tensor
