# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #


from memory.buffer import NDBuffer
from algorithm.reduction import _get_nd_indices_from_flat_index
from algorithm.sort import _quicksort
from math import iota


fn top_k[
    rank: Int, type: DType
](
    input: NDBuffer[rank, DimList.create_unknown[rank](), type],
    k: Int,
    axis: Int,
    largest: Bool,
    out_vals: NDBuffer[rank, DimList.create_unknown[rank](), type],
    out_idxs: NDBuffer[rank, DimList.create_unknown[rank](), DType.int64],
):
    let shape = input.get_shape()
    let total_size = shape.flattened_length()
    let num_vecs = total_size // shape[axis]

    var idxs = DynamicVector[Int64](shape[axis])
    idxs.resize(shape[axis])

    for vec_idx in range(num_vecs):
        var indices = _get_nd_indices_from_flat_index[rank](
            vec_idx, shape, axis
        )
        iota[DType.int64](idxs)

        if largest:

            @parameter
            @always_inline
            fn _val_greater_than[ty: AnyType](lhs: ty, rhs: ty) -> Bool:
                indices[axis] = rebind[Int64](lhs).__int__()
                let lhs_val = input[indices]
                indices[axis] = rebind[Int64](rhs).__int__()
                let rhs_val = input[indices]
                return lhs_val > rhs_val

            _quicksort[Int64, _val_greater_than](idxs.data, idxs.__len__())
        else:

            @parameter
            @always_inline
            fn _val_less_than_eq[ty: AnyType](lhs: ty, rhs: ty) -> Bool:
                indices[axis] = rebind[Int64](lhs).__int__()
                let lhs_val = input[indices]
                indices[axis] = rebind[Int64](rhs).__int__()
                let rhs_val = input[indices]
                return lhs_val <= rhs_val

            _quicksort[Int64, _val_less_than_eq](idxs.data, idxs.__len__())

        @parameter
        @always_inline
        fn _idx_less_than[ty: AnyType](lhs: ty, rhs: ty) -> Bool:
            return rebind[Int64](lhs) < rebind[Int64](rhs)

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
                _quicksort[Int64, _idx_less_than](
                    idxs.data.offset(i), num_equal
                )
            i += num_equal

        for i in range(k):
            indices[axis] = idxs[i].__int__()
            let val = input[indices]
            indices[axis] = i
            out_vals[indices] = val
            out_idxs[indices] = idxs[i]

    idxs._del_old()
