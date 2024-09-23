# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

from tensor_internal import StaticTensorSpec
from utils import StaticIntTuple
from collections import InlineArray


@always_inline
fn _dot_prod[
    rank: Int
](x: StaticIntTuple[rank], y: StaticIntTuple[rank]) -> Int:
    var offset = 0

    @parameter
    for i in range(rank):
        offset += x[i] * y[i]
    return offset


@always_inline
fn _slice_to_tuple[
    func: fn (Slice) capturing [_] -> Int, rank: Int
](slices: InlineArray[Slice, rank]) -> StaticIntTuple[rank]:
    """Takes a tuple of `Slice`s and returns a tuple of Ints.
    `func` is used to extract the appropriate field (i.e. start, stop or end)
    of the Slice.
    """
    var tuple = StaticIntTuple[rank]()

    @parameter
    for i in range(rank):
        tuple[i] = func(slices[i])
    return tuple


@always_inline
fn _row_major_strides[
    type: DType, rank: Int
](spec: StaticTensorSpec[type, rank]) -> StaticIntTuple[rank]:
    var offset = 1
    var strides = StaticIntTuple[rank]()

    @parameter
    for i in range(rank - 1, -1, -1):
        strides[i] = offset
        offset *= spec.shape[i]
    return strides
