# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
"""Op implementation for tile."""

from collections.abc import Iterable

from max.mlir.dialects import rmo

from .. import dtype_promotion
from ..graph import Graph
from ..type import Dim, DimLike, StaticDim, TensorType
from ..value import TensorValue, TensorValueLike


def tile(x: TensorValueLike, repeats: Iterable[DimLike]):
    """
    Returns a new Tensor as the result of copying the input tensor N_i times
    on each dimension, where N_i = repeats[i].

    The i-th dimension of output shape will be the ith dimension of input shape
    multiplied by N_i.
    """
    x = dtype_promotion._restrict_to_strong_dtypes(x)
    shape = x.shape

    repeats = list(Dim(d) for d in repeats)
    if len(shape) != len(repeats):
        raise ValueError(
            "Input rank and number of elements in repeats must match:"
            f" {shape=}, {repeats=}"
        )

    if any(count.dim <= 0 for count in repeats if isinstance(count, StaticDim)):
        raise ValueError(f"Repeats must all be positive: {repeats=}")

    output_dims = [dim * count for dim, count in zip(shape, repeats)]

    return Graph.current._add_op(
        rmo.mo_tile,
        TensorType(dtype=x.dtype, shape=output_dims, device=x.device).to_mlir(),
        x,
        TensorValue.from_shape(repeats),
    )[0].tensor
