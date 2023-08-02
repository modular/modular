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
# slice_as_view
# ===----------------------------------------------------------------------===#


@always_inline
fn slice_as_view[
    type: DType,
    start_type: DType,
    end_type: DType,
    step_type: DType,
    rank: Int,
](
    tensor: NDBuffer[rank, DimList.create_unknown[rank](), type],
    starts: NDBuffer[1, DimList.create_unknown[1](), start_type],
    ends: NDBuffer[1, DimList.create_unknown[1](), end_type],
    steps: NDBuffer[1, DimList.create_unknown[1](), step_type],
) -> NDBuffer[rank, DimList.create_unknown[rank](), type]:

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
    return NDBuffer[rank, DimList.create_unknown[rank](), type](
        new_data, new_shape, new_stride
    )


# ===----------------------------------------------------------------------===#
# slice_as_copy
# ===----------------------------------------------------------------------===#


@always_inline
fn slice_as_copy[
    type: DType, index_type: DType, in_rank: Int
](
    output: NDBuffer[in_rank, DimList.create_unknown[in_rank](), type],
    tensor: NDBuffer[in_rank, DimList.create_unknown[in_rank](), type],
    start: NDBuffer[1, DimList.create_unknown[1](), index_type],
    end: NDBuffer[1, DimList.create_unknown[1](), index_type],
    step: NDBuffer[1, DimList.create_unknown[1](), index_type],
    out_chain: OutputChainPtr,
):

    # Apply slice to the tensor
    let sliced = slice_as_view(tensor, start, end, step)

    # Copy lambda sliced view into output buffer.
    @always_inline
    @parameter
    fn copy[simd_width: Int, rank: Int](idx: StaticIntTuple[rank]):
        let index = rebind[StaticIntTuple[in_rank]](idx)
        output.simd_store[simd_width](
            index, sliced.simd_load[simd_width](index)
        )

    # Invoke copy.
    elementwise[in_rank, 1, copy](output.dynamic_shape, out_chain)


# ===----------------------------------------------------------------------===#
# slice_shape
# ===----------------------------------------------------------------------===#


@always_inline
fn slice_shape[
    input_rank: Int,
    input_type: DType,
    start_type: DType,
    stop_type: DType,
    step_type: DType,
    single_thread_blocking_override: Bool,
](
    input_buf: NDBuffer[
        input_rank, DimList.create_unknown[input_rank](), input_type
    ],
    start_buf: NDBuffer[1, DimList.create_unknown[1](), start_type],
    stop_buf: NDBuffer[1, DimList.create_unknown[1](), stop_type],
    step_buf: NDBuffer[1, DimList.create_unknown[1](), step_type],
) -> StaticIntTuple[input_rank]:

    # TODO(17512)
    debug_assert(
        input_rank == start_buf.dim(0),
        "start indices size must equal input rank",
    )
    debug_assert(
        input_rank == stop_buf.dim(0), "stop indices size must equal input rank"
    )
    debug_assert(
        input_rank == step_buf.dim(0), "step indices size must equal input rank"
    )
    for axis in range(input_rank):
        debug_assert(step_buf[axis].to_int() != 0, "step must be non-zero")

    # NOTE this assumes `slice_as_view` can handle input with null data pointer
    let output_shape = slice_as_view(
        input_buf, start_buf, stop_buf, step_buf
    ).dynamic_shape
    return output_shape
