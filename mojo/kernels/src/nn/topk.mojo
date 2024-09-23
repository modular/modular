# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #


from collections import List
from math import iota

from algorithm.functional import parallelize_over_rows
from algorithm.reduction import _get_nd_indices_from_flat_index
from buffer import NDBuffer
from builtin.sort import _quicksort
from memory import UnsafePointer
from register import mogg_register_shape_func

from utils import Span, StaticIntTuple


@always_inline
fn top_k_shape_impl[
    type: DType,
    rank: Int,
    axis_type: DType,
    single_thread_blocking_override: Bool,
](
    input: NDBuffer[type, rank],
    k_buf: NDBuffer[axis_type, 1],
    axis_buf: NDBuffer[axis_type, 1],
) raises -> StaticIntTuple[rank]:
    """
    Compute the output shape of a  top/bottom k operation.

    Parameters:
        type: Data type of the input buffer.
        rank: Rank of the input.
        axis_type: Type of the axis and K arguments.
        single_thread_blocking_override: If this function can block.

    Args:
        input: The input tensor.
        k_buf: The K value in a tensor.
        axis_buf: The axis value in a tensor.

    Returns:
        The output shape.
    """
    var axis = int(axis_buf[0])
    var k = int(k_buf[0])

    if axis < 0 or axis >= rank:
        raise Error("[top/bottom-k] axis must be within [0, rank]")
    if k < 0 or k > input.get_shape()[axis]:
        raise Error("[top/bottom-k] k must be within [0, input_shape[axis]]")

    var shape = input.get_shape()
    shape[axis] = k

    return shape


@mogg_register_shape_func("mo.top_k")
@always_inline
fn top_k_shape[
    type: DType,
    rank: Int,
    axis_type: DType,
    single_thread_blocking_override: Bool,
](
    input: NDBuffer[type, rank],
    k_buf: NDBuffer[axis_type, 1],
    axis_buf: NDBuffer[axis_type, 1],
) raises -> StaticIntTuple[rank]:
    return top_k_shape_impl[
        single_thread_blocking_override=single_thread_blocking_override
    ](input, k_buf, axis_buf)


@mogg_register_shape_func("mo.bottom_k")
@always_inline
fn bottom_k_shape[
    type: DType,
    rank: Int,
    axis_type: DType,
    single_thread_blocking_override: Bool,
](
    input: NDBuffer[type, rank],
    k_buf: NDBuffer[axis_type, 1],
    axis_buf: NDBuffer[axis_type, 1],
) raises -> StaticIntTuple[rank]:
    return top_k_shape_impl[
        single_thread_blocking_override=single_thread_blocking_override
    ](input, k_buf, axis_buf)


fn top_k[
    rank: Int, type: DType
](
    input: NDBuffer[type, rank],
    k: Int,
    axis: Int,
    largest: Bool,
    out_vals: NDBuffer[type, rank],
    out_idxs: NDBuffer[DType.int64, rank],
    sorted: Bool = True,
):
    """
    Implementation of the Top K algorithm. Returns the top or bottom K elements
    and their index along a specified axis.

    Parameters:
        rank: Rank of the input.
        type: Data type of the input buffer.

    Args:
        input: The input tensor.
        k: Represents the K largest/smallest value.
        axis: On which axis it should operate.
        largest: If true, acts like top K. Otherwise, bottom K.
        out_vals: Output values.
        out_idxs: Output indices.
        sorted: Indicates if the top/bottom K elements are in (stable) sorted order.
    """
    alias grain_size = 1000
    _top_k(
        input,
        k,
        axis,
        largest,
        out_vals,
        out_idxs,
        grain_size,
        sorted,
    )


fn _top_k[
    rank: Int, type: DType
](
    input: NDBuffer[type, rank],
    k: Int,
    axis: Int,
    largest: Bool,
    out_vals: NDBuffer[type, rank],
    out_idxs: NDBuffer[DType.int64, rank],
    parallelism_grain_size: Int,  # impl detail, exposed for testing
    sorted: Bool,
):
    var shape = input.get_shape()

    @__copy_capture(shape)
    @parameter
    fn process_rows(start_row: Int, end_row: Int):
        # Allocate the index list without initializing its elements.
        var idxs = List(
            unsafe_pointer=UnsafePointer[Int64].alloc(shape[axis]),
            size=shape[axis],
            capacity=shape[axis],
        )

        for row_idx in range(start_row, end_row):
            var indices = _get_nd_indices_from_flat_index[rank](
                row_idx, shape, axis
            )
            iota(idxs)

            @parameter
            @always_inline
            fn indices_to_val(idx: Int64) -> Scalar[type]:
                indices[axis] = int(idx)
                return input[indices]

            if largest:

                @parameter
                @always_inline
                fn _val_greater_than(lhs: Int64, rhs: Int64) -> Bool:
                    return indices_to_val(lhs) > indices_to_val(rhs)

                if sorted:
                    sort[_val_greater_than](idxs)
                else:
                    _ = partition[_val_greater_than](idxs, k)
            else:

                @parameter
                @always_inline
                fn _val_less_than(lhs: Int64, rhs: Int64) -> Bool:
                    return indices_to_val(lhs) < indices_to_val(rhs)

                if sorted:
                    sort[_val_less_than](idxs)
                else:
                    _ = partition[_val_less_than](idxs, k)

            if sorted:
                # for duplicate vals, the smaller index needs to appear first
                # _quicksort is not stable, so do another pass to enforce this
                # could use a stable sorting algorithm but the complexity is O(n*log(n)*log(n))
                # this is also what tensorflow and PT do:
                # https://github.com/tensorflow/tensorflow/blob/v2.10.0/tensorflow/core/kernels/topk_op.cc#L171-L172
                var i = 0
                while i < shape[axis] - 1:
                    indices[axis] = int(idxs[i])
                    var curr = input[indices]
                    var num_equal = 1
                    for j in range(i + 1, shape[axis]):
                        indices[axis] = int(idxs[j])
                        var next = input[indices]
                        if curr != next:
                            break
                        num_equal += 1
                    if num_equal > 1:
                        var ptr = idxs.data + i
                        sort(
                            Span[idxs.T, __lifetime_of(idxs)](
                                unsafe_ptr=ptr, len=num_equal
                            )
                        )
                    i += num_equal

            for i in range(k):
                indices[axis] = int(idxs[i])
                var val = input[indices]
                indices[axis] = i
                out_vals[indices] = val
                out_idxs[indices] = idxs[i]

    parallelize_over_rows[process_rows](shape, axis, parallelism_grain_size)
