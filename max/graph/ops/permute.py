# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
"""Op implementation for permute."""

import numpy as np
from max.dtype import DType
from max.mlir.dialects import rmo

from ..graph import Graph
from ..type import DeviceRef
from ..value import TensorType, TensorValue, TensorValueLike
from .constant import constant


def permute(x: TensorValueLike, dims: list[int]) -> TensorValue:
    """Permutes all dimensions of a symbolic tensor.

    Args:
        input: The input symbolic tensor to transpose.
        dims: The desired ordering of the dimensions in the output tensor.

    Returns:
        A new symbolic tensor with the dimensions permuted to match the passed in order.
        It has the same elements and dtype, but the order of the elements
        is different according to the permutation.
    """
    x = TensorValue(x)
    rank = x.rank

    if len(dims) != rank:
        raise ValueError(
            f"The rank of the input ({rank}) does not match the number of dims"
            f" used for ordering ({len(dims)})"
        )

    for d in dims:
        if not -rank <= d < rank:
            raise IndexError(
                f"All dimensions in the ordering must be be between {-rank} and"
                f" {rank - 1} (inclusive), but was {d}"
            )

    dims = [d + rank if d < 0 else d for d in dims]
    if len(set(dims)) != len(dims):
        raise ValueError(
            f"The ordering may not contain duplicate dimensions: {dims}"
        )

    shape = x.shape
    new_shape = [shape[d] for d in dims]

    return Graph.current._add_op(
        rmo.mo_transpose,
        TensorType(dtype=x.dtype, shape=new_shape, device=x.device).to_mlir(),
        x,
        constant(np.array(dims), DType.int64, DeviceRef.CPU()),
    )[0].tensor
