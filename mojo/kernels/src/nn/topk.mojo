# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #


from math import iota

from algorithm.functional import parallelize_over_rows
from algorithm.reduction import _get_nd_indices_from_flat_index
from algorithm.sort import _quicksort, partition, sort
from memory.buffer import NDBuffer


fn top_k_shape[
    type: DType,
    rank: Int,
    axis_type: DType,
    single_thread_blocking_override: Bool,
](
    input: NDBuffer[rank, DimList.create_unknown[rank](), type],
    k_buf: NDBuffer[1, DimList.create_unknown[1](), axis_type],
    axis_buf: NDBuffer[1, DimList.create_unknown[1](), axis_type],
) -> StaticIntTuple[rank]:
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
    debug_assert(k_buf.size() == 1, "k_buf must be a scalar")
    debug_assert(axis_buf.size() == 1, "axis_buf must be a scalar")

    let axis = axis_buf[0].to_int()
    let k = k_buf[0].to_int()

    debug_assert(axis < rank, "Axis should be less than the rank of the input")
    debug_assert(
        k <= input.get_shape()[axis],
        "K should be less or equal to the size of the axis",
    )

    var shape = input.get_shape()

    shape[axis] = k

    return shape


fn top_k[
    rank: Int, type: DType
](
    input: NDBuffer[rank, DimList.create_unknown[rank](), type],
    k: Int,
    axis: Int,
    largest: Bool,
    out_vals: NDBuffer[rank, DimList.create_unknown[rank](), type],
    out_idxs: NDBuffer[rank, DimList.create_unknown[rank](), DType.int64],
    out_chain: OutputChainPtr,
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
        out_chain: Output chain to notify for errors and ready states.
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
        out_chain,
        grain_size,
        sorted,
    )


fn _top_k[
    rank: Int, type: DType
](
    input: NDBuffer[rank, DimList.create_unknown[rank](), type],
    k: Int,
    axis: Int,
    largest: Bool,
    out_vals: NDBuffer[rank, DimList.create_unknown[rank](), type],
    out_idxs: NDBuffer[rank, DimList.create_unknown[rank](), DType.int64],
    out_chain: OutputChainPtr,
    parallelism_grain_size: Int,  # impl detail, exposed for testing
    sorted: Bool,
):
    let shape = input.get_shape()

    @parameter
    fn process_rows(start_row: Int, end_row: Int):
        var idxs = DynamicVector[Int64](shape[axis])
        idxs.resize(shape[axis])
        for row_idx in range(start_row, end_row):
            var indices = _get_nd_indices_from_flat_index[rank](
                row_idx, shape, axis
            )
            iota[DType.int64](idxs)

            @parameter
            @always_inline
            fn indices_to_val(idx: Int64) -> SIMD[type, 1]:
                indices[axis] = idx.__int__()
                return input[indices]

            if largest:

                @parameter
                @always_inline
                fn _val_greater_than_eq[
                    ty: AnyRegType
                ](lhs: ty, rhs: ty) -> Bool:
                    return indices_to_val(rebind[Int64](lhs)) >= indices_to_val(
                        rebind[Int64](rhs)
                    )

                if sorted:
                    _quicksort[Int64, _val_greater_than_eq](
                        idxs.data, idxs.__len__()
                    )
                else:
                    partition[Int64, _val_greater_than_eq](
                        idxs.data, k, idxs.__len__()
                    )
            else:

                @parameter
                @always_inline
                fn _val_less_than_eq[ty: AnyRegType](lhs: ty, rhs: ty) -> Bool:
                    return indices_to_val(rebind[Int64](lhs)) <= indices_to_val(
                        rebind[Int64](rhs)
                    )

                if sorted:
                    _quicksort[Int64, _val_less_than_eq](
                        idxs.data, idxs.__len__()
                    )
                else:
                    partition[Int64, _val_less_than_eq](
                        idxs.data, k, idxs.__len__()
                    )

            if sorted:
                # for duplicate vals, the smaller index needs to appear first
                # _quicksort is not stable, so do another pass to enforce this
                # could use a stable sorting algorithm but the complexity is O(n*log(n)*log(n))
                # this is also what tensorflow and PT do:
                # https://github.com/tensorflow/tensorflow/blob/v2.10.0/tensorflow/core/kernels/topk_op.cc#L171-L172
                var i = 0
                while i < shape[axis] - 1:
                    indices[axis] = idxs[i].__int__()
                    let curr = input[indices]
                    var num_equal = 1
                    for j in range(i + 1, shape[axis]):
                        indices[axis] = idxs[j].__int__()
                        let next = input[indices]
                        if curr != next:
                            break
                        num_equal += 1
                    if num_equal > 1:
                        var ptr = rebind[Pointer[Int64]](idxs.data.offset(i))
                        sort[DType.int64](ptr, num_equal)
                    i += num_equal

            for i in range(k):
                indices[axis] = idxs[i].__int__()
                let val = input[indices]
                indices[axis] = i
                out_vals[indices] = val
                out_idxs[indices] = idxs[i]

        idxs._del_old()

    parallelize_over_rows[rank, process_rows](
        shape, axis, out_chain, parallelism_grain_size
    )
