# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
"""Op implementation for band_part."""

from __future__ import annotations

from max.dtype import DType
from max.mlir.dialects import rmo

from ..graph import Graph
from ..type import StaticDim
from ..value import TensorValue, TensorValueLike
from .constant import constant


def band_part(
    x: TensorValueLike,
    num_lower: int | None = None,
    num_upper: int | None = None,
    exclude: bool = False,
) -> TensorValue:
    """Masks out everything except a diagonal band of an input matrix.

    Copies a tensor setting everything outside the central diagonal band of the
    matricies to zero, where all but the last two axes are effectively batches,
    and the last two axes define sub matricies.

    Assumes the input has dimensions [I, J, ..., M, N], then the output tensor
    has the same shape as the input, and the values are given by

    .. code-block:: python

        out[i, j, ..., m, n] = in_band(m, n) * input[i, j,  ..., m, n].

    with the indicator function:

    .. code-block:: python

        in_band(m, n) = ((num_lower is None || (m - n) <= num_lower)) &&
                        (num_upper is None || (n - m) <= num_upper))

    Args:
        input: The input to mask out.
        num_lower: The number of diagonal bands to include below the central
            diagonal. If None, include the entire lower triangle.
        num_upper: The number of diagonal bands to include above the central
            diagonal. If None, include the entire upper triangle.
        exclude: If true, invert the selection of elements to mask. Elements
            in the band are set to zero.

    Returns:
        A symbolic tensor value with the configured selection masked out
        to 0 values, and the remaining values copied from the input tensor.

    Raises:
        ValueError: If the input tensor rank is less than 2, or if num_lower/num_upper
            are out of bounds for statically known dimensions.
    """
    x = TensorValue(x)
    num_lower = -1 if num_lower is None else num_lower
    num_upper = -1 if num_upper is None else num_upper

    if num_lower < -1:
        raise ValueError(f"{num_lower=} must be non-negative")
    if num_upper < -1:
        raise ValueError(f"{num_upper=} must be non-negative")

    if x.rank < 2:
        raise ValueError(
            f"Input tensor {x.shape=} must have at least 2 dimensions"
        )

    # Check for out-of-bounds values for known static dimensions.
    # - m is the "vertical" dimension, and n is the "horizontal" dimension, visually
    # - num_lower is how far "down", so it is compared against m
    # - num_upper is how far "right", so it is compared against n
    *_, m, n = x.shape
    if isinstance(m, StaticDim) and num_lower >= int(m):
        raise ValueError(
            f"{num_lower=} is out of bounds for dimension size {int(m)}"
        )
    if isinstance(n, StaticDim) and num_upper >= int(n):
        raise ValueError(
            f"{num_upper=} is out of bounds for dimension size {int(n)}"
        )

    # Use the same device as the input tensor for constants
    device = x.type.device

    return Graph.current._add_op(
        rmo.mo_linalg_band_part,
        x.type.to_mlir(),
        x,
        constant(num_lower, DType.int64, device),
        constant(num_upper, DType.int64, device),
        constant(exclude, DType.bool, device),
    )[0].tensor
