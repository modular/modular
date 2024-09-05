# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
"""Op implementation for tile."""

from typing import Iterable

import numpy as np
from max import mlir
from max.dtype import DType
from max.mlir.dialects import rmo

from .. import dtype_promotion
from .constant import constant
from ..graph import Graph
from ..type import Shape, StaticDim, SymbolicDim, TensorType
from ..value import TensorValue, ValueLike


def tile(x: ValueLike, repeats: Iterable[int]):
    """
    Returns a new Tensor as the result of copying the input tensor N_i times
    on each dimension, where N_i = repeats[i].

    The i-th dimension of output shape will be the ith dimension of input shape
    multiplied by N_i.
    """
    x, = dtype_promotion._promote_weak_dtypes((x,))
    shape = x.shape

    # TODO(MSDK-604): Move all of these checks and shape inference to RMO.
    repeats = list(repeats)
    if len(shape) != len(repeats):
        raise ValueError(
            "Input rank and number of elements in repeats must match:"
            f" {shape=}, {repeats=}"
        )

    if any(count <= 0 for count in repeats):
        raise ValueError(f"Repeats must all be positive: {repeats=}")

    if not all(
        isinstance(dim, StaticDim) or count == 1
        for dim, count in zip(shape, repeats)
    ):
        raise ValueError(
            f"Can't non-trivially tile non-static dimensions: {shape=},"
            f" {repeats=}"
        )

    output_dims = [
        int(dim) * count if isinstance(dim, StaticDim) else dim
        for dim, count in zip(shape, repeats)
    ]

    return Graph.current._add_op(
        rmo.mo_tile,
        TensorType(dtype=x.dtype, shape=output_dims).to_mlir(),
        x,
        constant(np.array(repeats), DType.int64),
    )[0].tensor
