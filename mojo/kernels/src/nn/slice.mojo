# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

from Assert import debug_assert
from Buffer import Buffer, NDBuffer
from DType import DType
from Functional import elementwise
from Index import StaticIntTuple
from LLCL import OutputChainPtr
from List import Dim, DimList
from Math import div_ceil
from Range import range
from TypeUtilities import rebind

# ===----------------------------------------------------------------------===#
# slice
# ===----------------------------------------------------------------------===#
#
# TODO: This should be moved into the Stdlib folder, but currently that would
# create a naming conflict with this file. Placed here temporarily.
#
# ===----------------------------------------------------------------------===#


@register_passable("trivial")
struct slice:
    var start: Int
    var end: Int
    var step: Int

    @always_inline("nodebug")
    fn __init__() -> Self:
        return Self {start: 0, end: -1, step: 1}

    @always_inline("nodebug")
    fn __init__(end: Int) -> Self:
        return Self {start: 0, end: end, step: 1}

    @always_inline("nodebug")
    fn __init__(start: Int, end: Int) -> Self:
        return Self {start: start, end: end, step: 1}

    @always_inline("nodebug")
    fn __init__(start: Int, end: Int, step: Int) -> Self:
        return Self {start: start, end: end, step: step}

    @always_inline("nodebug")
    fn __eq__(self, other: Self) -> Bool:
        return (
            self.start == other.start
            and self.end == other.end
            and self.step == other.step
        )

    @always_inline("nodebug")
    fn __ne__(self, other: Self) -> Bool:
        return not (self == other)

    @always_inline("nodebug")
    fn __len__(self) -> Int:
        debug_assert(self != Self(), "invalid range, end value must be known")

        if self.step > 0:
            return div_ceil(self.end - self.start, self.step)
        else:
            return div_ceil(self.start - self.end, -self.step)


# ===----------------------------------------------------------------------===#
# slice_as_view
# ===----------------------------------------------------------------------===#


fn slice_as_view[
    type: DType, index_type: DType, rank: Int
](
    tensor: NDBuffer[rank, DimList[rank].create_unknown(), type],
    starts: Buffer[Dim(), index_type],
    ends: Buffer[Dim(), index_type],
    steps: Buffer[Dim(), index_type],
) -> NDBuffer[rank, DimList[rank].create_unknown(), type]:

    var new_shape = StaticIntTuple[rank]()
    var new_stride = StaticIntTuple[rank]()

    # The data does not change however we will be addressing a different
    # offset of the data.
    var new_data = tensor.data

    for i in range(rank):
        var start = starts[i].to_int()
        var stop = ends[i].to_int()
        let step = steps[i].to_int()

        if start < 0:
            start = start + tensor.dim(i)

        if stop < 0:
            stop = stop + tensor.dim(i)

        # Allow start and stop to truncate like numpy and torch allow.
        if start < 0:
            start = 0
        elif start >= tensor.dim(i):
            start = tensor.dim(i) - 1

        if stop < 0:
            stop = -1
        elif stop >= tensor.dim(i) and step > 0:
            stop = tensor.dim(i)
        elif stop >= tensor.dim(i) and step < 0:
            stop = tensor.dim(i) - 1

        let new_offset = start * tensor.stride(i)
        new_data = new_data.offset(new_offset)

        # Stride == number of elements to the next index in this dimension.
        # So to step we can just increase the stride.
        new_stride[i] = tensor.stride(i) * step

        # If the steps are positive we traverse from start, if negative from
        # stop.
        new_shape[i] = slice(start, stop, step).__len__()

    # Create the new view
    return NDBuffer[rank, DimList[rank].create_unknown(), type](
        new_data, new_shape, tensor.dynamic_dtype, new_stride
    )


# ===----------------------------------------------------------------------===#
# slice_as_copy
# ===----------------------------------------------------------------------===#


fn slice_as_copy[
    type: DType, index_type: DType, in_rank: Int
](
    output: NDBuffer[in_rank, DimList[in_rank].create_unknown(), type],
    tensor: NDBuffer[in_rank, DimList[in_rank].create_unknown(), type],
    start: Buffer[Dim(), index_type],
    end: Buffer[Dim(), index_type],
    step: Buffer[Dim(), index_type],
    out_chain: OutputChainPtr,
):

    # Apply slice to the tensor
    var sliced = slice_as_view[type, index_type, in_rank](
        tensor, start, end, step
    )

    # Copy lambda sliced view into output buffer.
    @always_inline
    fn copy[simd_width: Int, rank: Int](idx: StaticIntTuple[rank]):
        let index = rebind[StaticIntTuple[in_rank]](idx)
        output.simd_store[simd_width](
            index, sliced.simd_load[simd_width](index)
        )

    # Invoke copy.
    elementwise[in_rank, 1, 1, copy](output.dynamic_shape, out_chain)
