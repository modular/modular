# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #


from collections import List
from math import exp, iota
from random import random_float64

from algorithm.functional import parallelize_over_rows
from algorithm.reduction import _get_nd_indices_from_flat_index
from buffer import NDBuffer
from builtin.sort import _quicksort
from memory import UnsafePointer, Span
from nn.reshape import reshape
from register import register_internal_shape_func

from utils import IndexList


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
) raises -> IndexList[rank]:
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


@register_internal_shape_func("mo.top_k")
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
) raises -> IndexList[rank]:
    return top_k_shape_impl[
        single_thread_blocking_override=single_thread_blocking_override
    ](input, k_buf, axis_buf)


@register_internal_shape_func("mo.bottom_k")
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
) raises -> IndexList[rank]:
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
            ptr=UnsafePointer[Int64].alloc(shape[axis]),
            length=shape[axis],
            capacity=shape[axis],
        )

        for row_idx in range(start_row, end_row):
            var indices = _get_nd_indices_from_flat_index(row_idx, shape, axis)
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
                            Span[idxs.T, __origin_of(idxs)](
                                ptr=ptr, length=num_equal
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


@always_inline
fn top_k_fused_sampling[
    type: DType,
    rank: Int,
    out_idx_type: DType,
](
    k: Int,
    input: NDBuffer[type, rank],
    out_idxs: NDBuffer[out_idx_type, rank],
) raises:
    """
    Generalized implementation of the Top K algorithm with sampling.
    Returns the sampled index from the innermost dimension of the input
    tensor for each row/subvolume.

    Parameters:
        type: Data type of the input buffer.
        rank: Rank of the input.
        out_idx_type: Data type of the output indices.

    Args:
        k: Int - Represents the K largest values to consider for sampling.
        input: NDBuffer[type, rank] (Any shape)- The input tensor.
        out_idxs: NDBuffer[out_idx_type, rank] (shape of [input_shape[:-1]] + [1]) - The output indices.
    """
    constrained[out_idx_type == DType.int64, "out_idx_type must be int64"]()
    # materialize the out_vals which is of shape [input[:-1]] + [k]
    var out_vals_shape = input.get_shape()
    out_vals_shape[rank - 1] = k
    var out_vals = NDBuffer[type, rank](
        UnsafePointer[Scalar[type]].alloc(out_vals_shape.flattened_length()),
        out_vals_shape,
    )

    _top_k_sampling(
        k,
        input,
        out_vals,
        rebind[NDBuffer[DType.int64, rank]](out_idxs),
    )

    out_vals.data.free()


fn _top_k_sampling[
    type: DType,
    rank: Int,
](
    k: Int,
    input: NDBuffer[type, rank],
    out_vals: NDBuffer[type, rank],
    out_idxs: NDBuffer[DType.int64, rank],
) raises:
    """
    Generalized implementation of the Top K algorithm with sampling.
    Returns the sampled index from the innermost dimension of the input
    tensor for each row/subvolume.

    Parameters:
        type: Data type of the input buffer.
        rank: Rank of the input.

    Args:
        k: Int - Represents the K largest values to consider for sampling.
        input: NDBuffer[type, rank] (Any shape)- The input tensor.
        out_vals: NDBuffer[type, rank] (shape of [input[:-1]] + [k]) - The output values.
        out_idxs: NDBuffer[DType.int64, rank] (shape of [input[:-1]] + [1]) - The output indices.
    """
    # Now reshape for sampling
    var orig_in_shape: IndexList[rank] = input.get_shape()
    var last_dim = orig_in_shape[rank - 1]

    alias internal_rank = 2
    var internal_bs: Int
    var internal_in_shape: IndexList[internal_rank]

    @parameter
    if rank == 1:
        internal_bs = 1
        internal_in_shape = IndexList[internal_rank](1, input.size())
    elif rank == internal_rank:
        internal_bs = orig_in_shape[0]
        internal_in_shape = rebind[IndexList[internal_rank]](orig_in_shape)
    elif rank > internal_rank:
        internal_bs = int(orig_in_shape.flattened_length() / last_dim)
        internal_in_shape = IndexList[internal_rank](internal_bs, last_dim)
    else:
        raise Error("Unsupported input rank. Must be >= 1.")

    internal_out_shape = IndexList[internal_rank](internal_bs, k)
    internal_out_vals = reshape(out_vals, internal_out_shape)  # internal view
    internal_out_idxs_shape = IndexList[internal_rank](internal_bs, 1)
    internal_out_idxs = reshape(
        out_idxs, internal_out_idxs_shape
    )  # internal view
    # End reshape to internal rank

    var out_idxs_tmp = NDBuffer[DType.int64, internal_rank](
        UnsafePointer[Int64].alloc(int(out_vals.size())),
        internal_out_shape,  # topk returns K as last dim
    )
    _top_k[rank=internal_rank, type=type](
        reshape(input, internal_in_shape),
        k,
        axis=internal_rank - 1,  # Always operate on the last axis
        largest=True,
        out_vals=internal_out_vals,
        out_idxs=out_idxs_tmp,
        sorted=True,
        parallelism_grain_size=1,
    )

    # Sample from the top K elements
    for batch in range(internal_bs):
        # Calculate softmax normalization
        var max_val = internal_out_vals[batch, 0]
        var sum_exp = Scalar[type](0)
        var exp_vals = List[Scalar[type]](capacity=k)
        for i in range(k):
            var val = internal_out_vals[batch, i]
            var exp_val = exp(val - max_val)
            exp_vals.append(exp_val)
            sum_exp += exp_val

        # Sample using the normalized probabilities
        var r = sum_exp * random_float64().cast[type]()
        for i in range(k):
            r -= exp_vals[i]
            if r <= 0 or i == k - 1:
                # Store the sampled index and value
                internal_out_idxs[batch, 0] = out_idxs_tmp[batch, i]
                break

    out_idxs_tmp.data.free()
