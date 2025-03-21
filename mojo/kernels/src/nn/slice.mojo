# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

from math import clamp

from algorithm import elementwise
from buffer import NDBuffer
from runtime.asyncrt import DeviceContextPtr
from collections.string import StaticString

from utils._select import _select_register_value as select
from utils.index import IndexList


@always_inline("nodebug")
fn _normalize_and_clamp_dim(start: Int, step: Int, dim_i: Int) -> Int:
    # Normalize the start/stop indices
    var normalized_idx = select(start < 0, start + dim_i, start)

    # Compute the min/max for clamping start/end
    var idx_min = select(step > 0, 0, -1)
    var idx_max = select(step > 0, dim_i, dim_i - 1)

    # Allow start and stop to truncate like numpy and torch allow.
    return clamp(normalized_idx, idx_min, idx_max)


# ===-----------------------------------------------------------------------===#
# slice_dim_as_view
# ===-----------------------------------------------------------------------===#


@always_inline
fn slice_dim_as_view[
    type: DType, rank: Int, dim: Int
](
    tensor: NDBuffer[type, rank, *_, **_], start: Int, end: Int, step: Int
) -> NDBuffer[type, rank, tensor.origin]:
    var new_shape = tensor.get_shape()
    var new_stride = tensor.get_strides()

    var dim_i = tensor.dim(dim)
    var old_stride = tensor.stride(dim)

    # Normalize the start/stop indices
    var clamped_start = _normalize_and_clamp_dim(start, step, dim_i)
    var clamped_stop = _normalize_and_clamp_dim(end, step, dim_i)

    var new_offset = clamped_start * old_stride

    # The data does not change however we will be addressing a different
    # offset of the data.
    var new_data = tensor.data.offset(new_offset)

    # Stride == number of elements to the next index in this dimension.
    # So to step we can just increase the stride.
    new_stride[dim] = old_stride * step

    # If the steps are positive we traverse from start, if negative from
    # stop.
    new_shape[dim] = len(range(clamped_start, clamped_stop, step))

    # Create the new view
    return NDBuffer[type, rank, address_space = tensor.address_space](
        new_data, new_shape, new_stride
    )


# ===-----------------------------------------------------------------------===#
# slice_as_view
# ===-----------------------------------------------------------------------===#


@always_inline
fn slice_as_view[
    type: DType,
    start_type: DType,
    end_type: DType,
    step_type: DType,
    rank: Int,
](
    tensor: NDBuffer[type, rank, *_, **_],
    starts: NDBuffer[start_type, 1, *_, **_],
    ends: NDBuffer[end_type, 1, *_, **_],
    steps: NDBuffer[step_type, 1, *_, **_],
) -> NDBuffer[type, rank, tensor.origin]:
    var new_shape = IndexList[rank]()
    var new_stride = IndexList[rank]()

    # The data does not change however we will be addressing a different
    # offset of the data.
    var new_data = tensor.data

    @parameter
    for i in range(rank):
        var start = Int(starts[i])
        var stop = Int(ends[i])
        var step = Int(steps[i])
        var dim_i = tensor.dim(i)
        var stride_i = tensor.stride(i)

        # Normalize the start/stop indices
        start = _normalize_and_clamp_dim(start, step, dim_i)
        stop = _normalize_and_clamp_dim(stop, step, dim_i)

        var new_offset = start * stride_i
        new_data = new_data.offset(new_offset)

        # Stride == number of elements to the next index in this dimension.
        # So to step we can just increase the stride.
        new_stride[i] = stride_i * step

        # If the steps are positive we traverse from start, if negative from
        # stop.
        new_shape[i] = len(range(start, stop, step))

    # Create the new view
    return NDBuffer[type, rank, address_space = tensor.address_space](
        new_data, new_shape, new_stride
    )


# ===-----------------------------------------------------------------------===#
# copy_to_slice
# ===-----------------------------------------------------------------------===#


@always_inline
fn copy_to_slice[
    type: DType,
    start_type: DType,
    end_type: DType,
    step_type: DType,
    in_rank: Int,
    target: StaticString = "cpu",
](
    buffer: NDBuffer[mut=True, type, in_rank],
    in_slice: NDBuffer[type, in_rank],
    start: NDBuffer[start_type, 1],
    end: NDBuffer[end_type, 1],
    step: NDBuffer[step_type, 1],
    context: DeviceContextPtr = DeviceContextPtr(),
) raises:
    var expected_shape = slice_shape[single_thread_blocking_override=True](
        buffer, start, end, step
    )

    if expected_shape != in_slice.get_shape():
        raise Error(
            "Shape mismatch for mo.mutable.store.slice: expected 'slice'",
            " operand to have shape: ",
            expected_shape,
            " but got: ",
            in_slice.get_shape(),
        )

    var buffer_slice_view = slice_as_view(buffer, start, end, step)

    @always_inline
    @__copy_capture(in_slice, buffer_slice_view)
    @parameter
    fn copy[simd_width: Int, rank: Int](idx: IndexList[rank]):
        var index = rebind[IndexList[in_rank]](idx)
        buffer_slice_view.store[width=simd_width](
            index, in_slice.load[width=simd_width](index)
        )

    elementwise[copy, 1, target=target](buffer_slice_view.get_shape(), context)


# ===-----------------------------------------------------------------------===#
# slice_as_copy
# ===-----------------------------------------------------------------------===#


@always_inline
fn slice_as_copy[
    type: DType, index_type: DType, in_rank: Int
](
    output: NDBuffer[mut=True, type, in_rank],
    tensor: NDBuffer[type, in_rank],
    start: NDBuffer[index_type, 1],
    end: NDBuffer[index_type, 1],
    step: NDBuffer[index_type, 1],
) raises:
    # Apply slice to the tensor
    var sliced = slice_as_view(tensor, start, end, step)

    # Copy lambda sliced view into output buffer.
    @always_inline
    @__copy_capture(sliced)
    @parameter
    fn copy[simd_width: Int, rank: Int](idx: IndexList[rank]):
        var index = rebind[IndexList[in_rank]](idx)
        output.store[width=simd_width](
            index, sliced.load[width=simd_width](index)
        )

    # Invoke copy.
    elementwise[copy, 1](output.dynamic_shape)


# ===-----------------------------------------------------------------------===#
# slice_shape
# ===-----------------------------------------------------------------------===#


@always_inline
fn slice_shape[
    input_rank: Int,
    input_type: DType,
    start_type: DType,
    stop_type: DType,
    step_type: DType,
    single_thread_blocking_override: Bool,
](
    input_buf: NDBuffer[input_type, input_rank],
    start_buf: NDBuffer[start_type, 1],
    stop_buf: NDBuffer[stop_type, 1],
    step_buf: NDBuffer[step_type, 1],
) raises -> IndexList[input_rank]:
    if input_rank != start_buf.dim(0):
        raise Error("[slice] start indices size must equal input rank")
    if input_rank != stop_buf.dim(0):
        raise Error("[slice] stop indices size must equal input rank")
    if input_rank != step_buf.dim(0):
        raise Error("[slice] step indices size must equal input rank")

    for axis in range(input_rank):
        if step_buf[axis] == 0:
            raise Error("[slice] step must be non-zero")

    var output_shape = IndexList[input_rank]()

    for i in range(input_rank):
        var start = Int(start_buf[i])
        var stop = Int(stop_buf[i])
        var step = Int(step_buf[i])
        var dim_i = input_buf.dim(i)

        start = _normalize_and_clamp_dim(start, step, dim_i)
        stop = _normalize_and_clamp_dim(stop, step, dim_i)

        if step > 0 and stop < start:
            raise Error(
                "[slice] normalized stop cannot be smaller than start for"
                " positive step"
            )

        if step < 0 and start < stop:
            raise Error(
                "[slice] normalized start cannot be smaller than stop for"
                " negative step"
            )

        output_shape[i] = len(range(start, stop, step))

    return output_shape
