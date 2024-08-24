# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
"""Op implementation for transpose."""

import numpy as np
from max.mlir.dialects import rmo

from ..graph import Graph
from ..value import TensorValue, TensorType, ValueLike


def transpose(x: ValueLike, dim_1: int, dim_2: int) -> TensorValue:
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
        TensorType(dtype=v.dtype, shape=new_shape).to_mlir(),
        v,
        TensorValue(indices),
    )[0].tensor
