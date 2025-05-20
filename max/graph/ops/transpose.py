# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
"""Op implementation for transpose."""

import numpy as np
from max.dtype import DType
from max.mlir.dialects import rmo

from ..graph import Graph
from ..type import DeviceRef
from ..value import TensorType, TensorValue, TensorValueLike
from .constant import constant


def transpose(x: TensorValueLike, dim_1: int, dim_2: int) -> TensorValue:
    """Transposes two dimensions of a symbolic tensor.
    For more information, see :obj:`~max.graph.TensorValue.transpose()`.

    Args:
        x: The input symbolic tensor to transpose.
        dim_1: One of the two dimensions to transpose. If negative, this indexes
           from the end of the tensor. For example,
           :code:`transpose(v, -1, -2)` transposes the last two dimensions.
        dim_2: The other dimension to transpose. May also be negative to index from
           the end of the tensor.

    Returns:
        A new symbolic tensor with the two specified dimensions transposed.
        It has the same elements and dtype, but the order of the elements
        is different according to the transposition.
    """
    v = TensorValue(x)

    rank = len(v.shape)
    if dim_1 < 0:
        dim_1 += rank
    if dim_2 < 0:
        dim_2 += rank

    new_shape = v.shape
    indices = np.array(range(len(new_shape)))

    new_shape[dim_1], new_shape[dim_2] = new_shape[dim_2], new_shape[dim_1]
    indices[dim_1], indices[dim_2] = dim_2, dim_1

    return Graph.current._add_op(
        rmo.mo_transpose,
        TensorType(dtype=v.dtype, shape=new_shape, device=v.device).to_mlir(),
        v,
        constant(indices, DType.int64, DeviceRef.CPU()),
    )[0].tensor
