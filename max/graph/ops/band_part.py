# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
"""Op implementation for band_part."""

from max.dtype import DType
from max.mlir.dialects import rmo

from ..graph import Graph
from ..type import DeviceRef
from ..value import TensorValue, TensorValueLike
from .constant import constant


def band_part(
    x: TensorValueLike, num_lower: int, num_upper: int, exclude: bool = False
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

        in_band(m, n) = ((num_lower < 0 || (m - n) <= num_lower)) &&
                        (num_upper < 0 || (n - m) <= num_upper))

    Args:
        input: The input to mask out.
        num_lower: The number of diagonal bands to include below the central
            diagonal. If -1, include the entire lower triangle.
        num_upper: The number of diagonal bands to include above the central
            diagonal. If -1, include the entire upper triangle.
        exclude: If true, invert the selection of elements to mask. Elements
            in the band are set to zero.

    Returns:
        A symbolic tensor value with the configured selection masked out
        to 0 values, and the remaining values copied from the input tensor.
    """
    x = TensorValue(x)
    return Graph.current._add_op(
        rmo.mo_linalg_band_part,
        x.type.to_mlir(),
        x,
        constant(num_lower, DType.int64, DeviceRef.CPU()),
        constant(num_upper, DType.int64, DeviceRef.CPU()),
        constant(exclude, DType.bool, DeviceRef.CPU()),
    )[0].tensor
