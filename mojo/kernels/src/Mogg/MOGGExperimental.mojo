# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

from algorithm.functional import _elementwise_impl, vectorize_unroll
from memory.buffer import NDBuffer
from MOGG import simd_load, simd_store
from MOGGIntList import IntList
from MOGGTensor import Tensor
from utils.index import StaticIntTuple
from utils._annotations import *


fn _get_start_indices_of_nth_subvolume[
    subvolume_rank: Int, static_shape: DimList
](n: Int, shape: IntList[static_shape]) -> IntList[shape._size_but_unknown]:
    """
    Converts from a flat 1D index into an ND index for the given shape. The
    `subvolume_rank` parameter is used to skip some of the ND dimension so we
    can calculate an index over a subset of the shape. I.E subvolume_rank=1
    will calcualte over (N, B, C, 0) with the last dimension being 0.
    """
    var out = IntList[shape._size_but_unknown].empty(shape.length)
    var curr_index = n

    @parameter
    if shape.has_static_length():
        alias rank = shape._length

        @always_inline
        @parameter
        fn compute_shape[idx: Int]():
            alias i = rank - 1 - idx - subvolume_rank
            out._unsafe_set_dim(i, curr_index % shape[i])
            curr_index //= shape[i]

        unroll[rank - subvolume_rank, compute_shape]()
    else:
        let rank = out.__len__()
        for i in range(rank - subvolume_rank - 1, -1, -1):
            out._unsafe_set_dim(i, curr_index % shape[i])
            curr_index //= shape[i]

    return out ^


@export
fn empty_tensor[
    type: DType,
](shape: IntList, strides: IntList) -> Tensor[
    type,
    shape.static_values,
    strides.static_values,
]:
    let ptr = DTypePointer[type].alloc(shape.nelems())
    return Tensor[type, shape.static_values, strides.static_values](
        ptr, shape, strides
    )


# TODO figure out what to do with deconstructors.
@export
fn _dealloc_tensor(t: Tensor):
    if t.data:
        t.data.free()


# Stand in for elementwise while we experiment with the new tensor api
@always_inline
fn apply_per_element[
    simd_width: Int,
    func: fn[width: Int] (IntList) capturing -> None,
](shape: IntList):
    let rank = shape.__len__()
    let total_size: Int = shape.nelems()
    let inner_loop = shape[shape.__len__() - 1]
    let outer_loop = total_size // inner_loop

    for outer_i in range(outer_loop):
        var indices = _get_start_indices_of_nth_subvolume[1](outer_i, shape)

        @always_inline
        @parameter
        fn func_wrapper[simd_width: Int](idx: Int):
            # The inner most dimension is vectorized, so we set it
            # to the index offset.
            indices._unsafe_set_dim(rank - 1, idx)
            func[simd_width, indices.static_values](indices)

        # We vectorize over the innermost dimension.
        vectorize_unroll[
            simd_width,
            1,
            func_wrapper,
        ](inner_loop)

        # We have to extend the lifetime of the indices as the above parameter
        # capture and use does not extend the lifetime of the object.
        _ = indices


@mogg_register_override("mo.add", 1000)
@mogg_kgen_experiment_kernel()
@export
fn my_add[
    type: DType, rank: Int
](
    out: NDBuffer[rank, DimList.create_unknown[rank](), type],
    x: NDBuffer[rank, DimList.create_unknown[rank](), type],
    y: NDBuffer[rank, DimList.create_unknown[rank](), type],
    out_chain: OutputChainPtr,
):
    @always_inline
    @parameter
    fn func[width: Int, _rank: Int](i: StaticIntTuple[_rank]):
        let indices = rebind[StaticIntTuple[rank]](i)
        let i1 = simd_load[type, 1, rank, x.shape](x, indices)
        let i2 = simd_load[type, 1, rank, y.shape](y, indices)
        let tmp = i1 + i2
        simd_store[type, width, rank](out, indices, tmp)

    _elementwise_impl[rank, 1, True, func, target="cpu"](
        out.dynamic_shape,
        out_chain,
    )


@mogg_register_override("mo.sub", 1000)
@mogg_kgen_experiment_kernel()
@export
fn my_sub[
    type: DType, rank: Int
](
    out: NDBuffer[rank, DimList.create_unknown[rank](), type],
    x: NDBuffer[rank, DimList.create_unknown[rank](), type],
    y: NDBuffer[rank, DimList.create_unknown[rank](), type],
    out_chain: OutputChainPtr,
):
    @always_inline
    @parameter
    fn func[width: Int, _rank: Int](i: StaticIntTuple[_rank]):
        let indices = rebind[StaticIntTuple[rank]](i)
        let i1 = simd_load[type, 1, rank, x.shape](x, indices)
        let i2 = simd_load[type, 1, rank, y.shape](y, indices)
        let tmp = i1 - i2
        simd_store[type, width, rank](out, indices, tmp)

    _elementwise_impl[rank, 1, True, func, target="cpu"](
        out.dynamic_shape,
        out_chain,
    )


@mogg_register_override("no-op", 1000)
@mogg_kgen_experiment_kernel()
@export
fn noop[
    type: DType, rank: Int
](
    out: NDBuffer[rank, DimList.create_unknown[rank](), type],
    x: NDBuffer[rank, DimList.create_unknown[rank](), type],
    y: NDBuffer[rank, DimList.create_unknown[rank](), type],
    out_chain: OutputChainPtr,
):
    pass


@mogg_register_override("diff_dtype", 1000)
@mogg_kgen_experiment_kernel()
@export
fn diff_dtype[
    type1: DType, type2: DType, rank: Int
](
    out: NDBuffer[rank, DimList.create_unknown[rank](), type1],
    x: NDBuffer[rank, DimList.create_unknown[rank](), type1],
    y: NDBuffer[rank, DimList.create_unknown[rank](), type2],
    out_chain: OutputChainPtr,
):
    pass


@mogg_register_override("many_lambdas", 1000)
@mogg_kgen_experiment_kernel()
@export
fn many_lambdas[
    type: DType, rank: Int
](
    out: NDBuffer[rank, DimList.create_unknown[rank](), type],
    x: NDBuffer[rank, DimList.create_unknown[rank](), type],
    y: NDBuffer[rank, DimList.create_unknown[rank](), type],
    out_chain: OutputChainPtr,
):
    # Lambda in lambda
    @always_inline
    @parameter
    fn func1[width: Int, _rank: Int](i: StaticIntTuple[_rank]):
        let indices = rebind[StaticIntTuple[rank]](i)
        let i1 = simd_load[type, 1, rank, x.shape](x, indices)
        let i2 = simd_load[type, 1, rank, y.shape](y, indices)
        let tmp = i1 + i2
        simd_store[type, width, rank](out, indices, tmp)

    # Second lambda
    @always_inline
    @parameter
    fn func2[width: Int, _rank: Int](i: StaticIntTuple[_rank]):
        let indices = rebind[StaticIntTuple[rank]](i)
        let i1 = simd_load[type, 1, rank, x.shape](x, indices)
        let i2 = simd_load[type, 1, rank, y.shape](y, indices)
        let tmp = i1 - i2
        simd_store[type, width, rank](out, indices, tmp)
