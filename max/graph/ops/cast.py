# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
"""Op implementation for cast."""

from max.dtype import DType
from max.mlir.dialects import mo

from ..graph import Graph
from ..value import TensorValue


def cast(x: TensorValue, dtype: DType) -> TensorValue:
    """Casts a symbolic tensor to a different data type.

    Args:
        x: The input tensor to cast.
        dtype: The target dtype to which the tensor is cast.

    Returns:
        A new symbolic tensor with the same shape as the input and the
        specified dtype.
    """
    cast_type = x.type.cast(dtype)
    return Graph.current._add_op(mo.cast, cast_type.to_mlir(), x)[0].tensor
