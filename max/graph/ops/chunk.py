# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
"""Op implementation for chunk."""

import math
from typing import MutableSequence

from ..type import _is_static_shape
from ..value import TensorValue
from .slice_tensor import SliceIndex, slice_tensor


def chunk(a: TensorValue, chunks: int, dim: int = 0) -> list[TensorValue]:
    """
    Chunk the tensor into chunks along the specified dim.

    Args:
        a: The tensor to chunk.
        chunks: The number of chunks to split the tensor into.
        dim: The dimension to split the tensor along.

    Returns:
        A list of tensors.

    Example:
        >>> a = TensorValue([1, 2, 3, 4, 5])
        >>> chunk(a, 2, 0)
        [TensorValue([1, 2]), TensorValue([3, 4])]
    """

    if not _is_static_shape(a.shape):
        raise ValueError("the chunk operation only supports static shapes")

    if a.rank == 0:
        return [a]

    # Check if the dimension is out of bounds. We put the error message
    # construction here to make sure we have the correct dim before we start
    # normalizing it.
    err_msg = (
        f"the dimension '{dim}' is out of bounds for tensor of rank {a.rank}"
    )

    if dim < 0:
        dim = a.rank + dim

    if a.rank < dim:
        raise ValueError(err_msg)

    n = int(a.shape[dim])

    # Determine chunk size using ceiling division.
    chunk_size = int(math.ceil(n / chunks))

    result = []
    for i in range(chunks):
        start = i * chunk_size
        if start >= n:
            break
        end = min(start + chunk_size, n)

        # Build a tuple of slice objects for all dimensions.
        slices: MutableSequence[SliceIndex] = [slice(None)] * a.rank
        slices[dim] = slice(start, end)
        result.append(slice_tensor(a, slices))
    return result
