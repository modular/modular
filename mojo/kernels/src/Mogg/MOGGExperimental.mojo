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


@mogg_tensor_allocator()
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


@mogg_register("to_tensor")
@export
@always_inline
fn to_tensor[
    type: DType,
    static_shape: DimList = DimList(),
    static_strides: DimList = DimList(),
](
    data: __mlir_type[`!kgen.pointer<scalar<`, type.value, `>>`],
    raw_shape_ptr: __mlir_type.`!kgen.pointer<index>`,
    length: Int,
) -> Tensor[type, static_shape, static_strides]:
    let shape_ptr = Pointer(raw_shape_ptr)

    var shape = IntList[static_shape].empty(length)
    var strides = IntList[static_strides].empty(length)

    var stride: Int = 1

    @parameter
    if shape.has_static_length():
        alias rank = static_shape.__len__()

        @always_inline
        @parameter
        fn body[idx: Int]():
            # Start from the back so we can accumulate the strides.
            let i = rank - 1 - idx
            shape._unsafe_set_dim(i, shape_ptr.load(i))
            strides._unsafe_set_dim(i, stride)
            stride *= shape[i]

        unroll[rank, body]()
    else:
        # Start from the back so we can accumulate the strides.
        for i in range(length - 1, -1, -1):
            shape._unsafe_set_dim(i, shape_ptr.load(i))
            strides._unsafe_set_dim(i, stride)
            stride *= shape[i]

    return Tensor[type, static_shape, static_strides](
        DTypePointer[type](data), shape, strides
    )


@mogg_register_override("mo.add", 1000)
@mogg_kgen_experiment_kernel()
@export
fn my_add(
    x: Tensor, y: Tensor
) -> Tensor[x.type, x.static_shape, x.static_strides]:
    var out = empty_tensor[x.type](x.shape, x.strides)

    @parameter
    @always_inline
    fn func[width: Int](i: IntList):
        let i2 = rebind[SIMD[x.type, width]](y.simd_load[width](i))
        out.simd_store[width](i, x.simd_load[width](i) + i2)

    apply_per_element[1, func](out.shape)
    return out ^


@mogg_register_override("mo.sub", 1000)
@mogg_kgen_experiment_kernel()
@export
fn my_sub(
    x: Tensor, y: Tensor
) -> Tensor[x.type, x.static_shape, x.static_strides]:
    var out = empty_tensor[x.type](x.shape, x.strides)

    @parameter
    @always_inline
    fn func[width: Int](i: IntList):
        let i2 = rebind[SIMD[x.type, width]](y.simd_load[width](i))
        out.simd_store[width](i, x.simd_load[width](i) - i2)

    apply_per_element[1, func](out.shape)
    return out ^
