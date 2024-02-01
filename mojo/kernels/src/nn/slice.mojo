# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

from math import div_ceil

from algorithm import elementwise
from memory.buffer import Buffer, NDBuffer

from utils.index import StaticIntTuple
from utils.list import Dim, DimList

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

    @unroll
    for i in range(rank):
        var start = int(starts[i])
        var stop = int(ends[i])
        let step = int(steps[i])
        let dim_i = tensor.dim(i)
        debug_assert(step != 0, "step must be nonzero")

        # Normalize the start/stop indices
        if start < 0:
            start = start + dim_i
        if stop < 0:
            stop = stop + dim_i

        # Compute the min/max for clamping start/end
        let idx_min = 0 if step > 0 else -1
        let idx_max = dim_i if step > 0 else dim_i - 1

        # Allow start and stop to truncate like numpy and torch allow.
        if start < idx_min:
            start = idx_min
        elif start > idx_max:
            start = idx_max

        if stop < idx_min:
            stop = idx_min
        elif stop > idx_max:
            stop = idx_max

        let new_offset = start * tensor.stride(i)
        new_data = new_data.offset(new_offset)

        # Stride == number of elements to the next index in this dimension.
        # So to step we can just increase the stride.
        new_stride[i] = tensor.stride(i) * step

        # If the steps are positive we traverse from start, if negative from
        # stop.
        new_shape[i] = len(slice(start, stop, step))

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
    elementwise[in_rank, 1, copy](output.dynamic_shape)


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
) raises -> StaticIntTuple[input_rank]:
    if input_rank != start_buf.dim(0):
        raise Error("start indices size must equal input rank")
    if input_rank != stop_buf.dim(0):
        raise Error("stop indices size must equal input rank")
    if input_rank != step_buf.dim(0):
        raise Error("step indices size must equal input rank")

    @unroll
    for axis in range(input_rank):
        if step_buf[axis] == 0:
            raise Error("step must be non-zero")

    var output_shape = StaticIntTuple[input_rank]()

    @unroll
    for i in range(input_rank):
        var start = int(start_buf[i])
        var stop = int(stop_buf[i])
        let step = int(step_buf[i])
        let dim_i = input_buf.dim(i)

        # Normalize the start/stop indices
        if start < 0:
            start = start + dim_i
        if stop < 0:
            stop = stop + dim_i

        # Compute the min/max for clamping start/end
        let idx_min = 0 if step > 0 else -1
        let idx_max = dim_i if step > 0 else dim_i - 1

        # Allow start and stop to truncate like numpy and torch allow.
        if start < idx_min:
            start = idx_min
        elif start > idx_max:
            start = idx_max

        if stop < idx_min:
            stop = idx_min
        elif stop > idx_max:
            stop = idx_max

        if step > 0 and stop < start:
            raise Error(
                "normalized stop cannot be smaller than start for positive step"
            )

        if step < 0 and start < stop:
            raise Error(
                "normalized start cannot be smaller than stop for negative step"
            )

        output_shape[i] = len(slice(start, stop, step))

    return output_shape
