# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

from algorithm.functional import _elementwise_impl
from utils._annotations import *
from MOGGIntList import IntList
from MOGGTensor import Tensor


@mogg_tensor_allocator()
@no_inline
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
        alias rank = len(static_shape)

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

    x.enable_fusion()
    y.enable_fusion()
    out.enable_fusion()

    @parameter
    @always_inline
    fn func[width: Int](i: IntList):
        let i2 = rebind[SIMD[x.type, width]](y.simd_load[width](i))
        out.simd_store[width](i, x.simd_load[width](i) + i2)

    out.for_each[1, func]()
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

    out.for_each[1, func]()
    return out ^
