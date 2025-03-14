# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
"""Op implementation for chunk."""

from typing import MutableSequence

from ..type import Shape
from ..value import TensorValue
from .slice_tensor import SliceIndex, slice_tensor


def chunk(a: TensorValue, chunks: int, dim: int = 0) -> list[TensorValue]:
    """
    Chunk the tensor into an exact number of chunks along the specified dim.

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
    # TODO(GEX-1943): once we have control flow in the graph, this can be updated to
    # dynamic chunk counts while still supporting algebraic dims. For now,
    # this will generate exactly chunks or fail.

    if a.rank == 0 and chunks > 1:
        msg = f"Cannot split 1 value into {chunks=}"
        raise ValueError(msg)

    # Check if the dimension is out of bounds. We put the error message
    # construction here to make sure we have the correct dim before we start
    # normalizing it.
    dim_err_msg = f"'{dim=}' is out of bounds for tensor of rank {a.rank}"
    chunks_err_msg = f"chunk: '{dim=}' of {a.shape=} must be exactly divisible into {chunks=}"

    if dim < 0:
        dim = a.rank + dim

    if a.rank < dim:
        raise ValueError(dim_err_msg)

    n = a.shape[dim]

    # Determine chunk size using ceiling division.
    chunk_size = (n + chunks - 1) // chunks
    full_size = chunk_size * chunks

    # Ensure exact divisbility
    target_shape = Shape(a.shape)
    target_shape[dim] = full_size
    # The rebind adds a better compile and runtime time error message if hit.
    a.rebind(target_shape, chunks_err_msg)
    try:
        # The reshape catches errors with static shapes right at build time.
        a = a.reshape(target_shape)
    except ValueError as e:
        # This intentially does not reraise the initial error.
        # The initial error is just noise used to figure out if we divide exactly.
        raise ValueError(chunks_err_msg)

    # Build a tuple of slice objects for all dimensions.
    slices: MutableSequence[SliceIndex] = [slice(None)] * a.rank
    result = []
    for i in range(chunks):
        start = i * chunk_size
        end = start + chunk_size

        slices[dim] = slice(start, end)
        result.append(slice_tensor(a, slices))
    return result
