# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #


from memory.buffer import NDBuffer
from algorithm.reduction import _get_nd_indices_from_flat_index
from algorithm.sort import _quicksort, sort
from math import iota
from algorithm.functional import parallelize_over_rows


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
):
    alias grain_size = 1000
    _top_k(input, k, axis, largest, out_vals, out_idxs, out_chain, grain_size)


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
                fn _val_greater_than[ty: AnyType](lhs: ty, rhs: ty) -> Bool:
                    return indices_to_val(rebind[Int64](lhs)) > indices_to_val(
                        rebind[Int64](rhs)
                    )

                _quicksort[Int64, _val_greater_than](idxs.data, idxs.__len__())
            else:

                @parameter
                @always_inline
                fn _val_less_than_eq[ty: AnyType](lhs: ty, rhs: ty) -> Bool:
                    return indices_to_val(rebind[Int64](lhs)) <= indices_to_val(
                        rebind[Int64](rhs)
                    )

                _quicksort[Int64, _val_less_than_eq](idxs.data, idxs.__len__())

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
