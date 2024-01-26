# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

from NN.Activations import relu
from math import sqrt
from utils._annotations import *
from MOGGIntList import IntList
from MOGGTensor import Tensor


@mogg_tensor_allocator()
@no_inline
@export
fn empty_tensor[
    type: DType,
](shape: IntList) -> Tensor[type, shape.static_values]:
    let ptr = DTypePointer[type].alloc(shape.nelems())
    let ref_cnt = Pointer[Scalar[DType.index]].alloc(1)
    ref_cnt[0] = 0
    return Tensor[type, shape.static_values](ptr, shape, ref_cnt)


@mogg_register("mogg.shape_from_kgen")
@always_inline
@export
fn get_static_shape(shape: IntList) -> StaticIntTuple[shape._safe_len]:
    return shape.stack_alloc_data


@mogg_tensor_allocator()
@no_inline
@export
fn empty_strided_tensor[
    type: DType,
](shape: IntList, strides: IntList) -> Tensor[
    type, shape.static_values, strides.static_values
]:
    let ptr = DTypePointer[type].alloc(shape.nelems())
    let ref_cnt = Pointer[Scalar[DType.index]].alloc(1)
    ref_cnt[0] = 0
    return Tensor[type, shape.static_values, strides.static_values](
        ptr, shape, ref_cnt
    )


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
) -> Tensor[type, static_shape, static_strides, _OWNED_MEMORY=False]:
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
            shape[i] = shape_ptr.load(i)
            strides[i] = stride
            stride *= shape[i]

        unroll[rank, body]()
    else:
        # Start from the back so we can accumulate the strides.
        for i in range(length - 1, -1, -1):
            shape[i] = shape_ptr.load(i)
            strides[i] = stride
            stride *= shape[i]

    return Tensor[type, static_shape, static_strides, _OWNED_MEMORY=False](
        DTypePointer[type](data),
        shape,
        strides,
        Pointer[Scalar[DType.index]](),
    )


@mogg_register_override("mo.add", 1000)
@export
fn my_add_with_fusion(x: Tensor, y: Tensor) -> Tensor[x.type, x.static_shape]:
    var out = empty_tensor[x.type](x.shape)

    x.enable_fusion()
    y.enable_fusion()
    out.enable_fusion()

    @parameter
    @always_inline
    fn func[width: Int, _t: DType](i: IntList) -> SIMD[_t, width]:
        let i2 = rebind[SIMD[x.type, width]](y.simd_load[width](i))
        return rebind[SIMD[_t, width]](x.simd_load[width](i) + i2)

    out.for_each[1, func]()
    return out


@mogg_register_override("mo.sub", 1000)
@export
fn my_sub_without_fusion(
    x: Tensor, y: Tensor
) -> Tensor[x.type, x.static_shape]:
    var out = empty_tensor[x.type](x.shape)

    @parameter
    @always_inline
    fn func[width: Int, _t: DType](i: IntList) -> SIMD[_t, width]:
        let i2 = rebind[SIMD[x.type, width]](y.simd_load[width](i))
        return rebind[SIMD[_t, width]](x.simd_load[width](i) - i2)

    out.for_each[1, func]()
    return out


@mogg_register_override("mo.relu", 1000)
@export
fn unary_op_with_fusion(
    x: Tensor,
) -> Tensor[x.type, x.static_shape]:
    var out = empty_tensor[x.type](x.shape)

    x.enable_fusion()
    out.enable_fusion()

    @parameter
    @always_inline
    fn func[width: Int, _t: DType](i: IntList) -> SIMD[_t, width]:
        return relu(rebind[SIMD[_t, width]](x.simd_load[width](i)))

    out.for_each[1, func]()
    return out


@mogg_register_override("mo.sqrt", 1000)
@export
fn unary_op_without_fusion(
    x: Tensor,
) -> Tensor[x.type, x.static_shape]:
    var out = empty_tensor[x.type](x.shape)

    @parameter
    @always_inline
    fn func[width: Int, _t: DType](i: IntList) -> SIMD[_t, width]:
        return sqrt(rebind[SIMD[_t, width]](x.simd_load[width](i)))

    out.for_each[1, func]()
    return out


@mogg_register_override("mo.transpose", 1000)
@export
fn transpose(x: Tensor, perm: Tensor) -> Tensor[x.type, x.same_rank_param()]:
    # Currently we don't support alias.
    # alias rank = x.static_rank

    var new_shape = IntList[x.same_rank_param()]()
    var new_stride = IntList[x.same_rank_param()]()

    @always_inline
    @parameter
    fn body[i: Int]():
        let index = IntList[DimList(i)](i)
        let dim = int(perm.simd_load[1](index))
        new_shape[i] = x.shape[dim]
        new_stride[i] = x.strides[dim]

    unroll[x.static_rank, body]()

    return Tensor[x.type, x.same_rank_param()](
        x.data, new_shape, new_stride, x.refcount()
    )


@mogg_register_override("copy", 1000)
@export
fn copy(x: Tensor) -> Tensor[x.type, x.static_shape]:
    var out = empty_tensor[x.type](x.shape)

    @parameter
    @always_inline
    fn func[width: Int, _t: DType](i: IntList) -> SIMD[_t, width]:
        return rebind[SIMD[_t, width]](x.simd_load[width](i))

    out.for_each[1, func]()
    return out


# Test we support a nested lambda using values from the parent contexts.
@mogg_register_override("add_like_custom_op_target", 1000)
@export
fn add_like_custom_op_target(
    x: Tensor, y: Tensor
) -> Tensor[x.type, x.static_shape]:
    var out = empty_tensor[x.type](x.shape)

    x.enable_fusion()
    y.enable_fusion()
    out.enable_fusion()

    @parameter
    @always_inline
    fn func[width: Int, _t: DType](i: IntList) -> SIMD[_t, width]:
        let i2 = rebind[SIMD[x.type, width]](y.simd_load[width](i))
        return rebind[SIMD[_t, width]](x.simd_load[width](i) + i2)

    out.for_each[1, func]()
    return out


@mogg_register("recursive_lambda_test_target")
@export
fn recursive_lambda_test_target(x: Tensor) -> Tensor[x.type, x.static_shape]:
    var out = empty_tensor[x.type](x.shape)
    var simd1 = SIMD[x.type, 1](0.5)

    @parameter
    @always_inline
    fn func[width: Int, _t: DType](i: IntList) -> SIMD[_t, width]:
        var simd2 = SIMD[_t, width](4.0)

        @parameter
        @always_inline
        fn inner(val: SIMD[_t, width]) -> SIMD[_t, width]:
            @parameter
            @always_inline
            fn inner2(val2: SIMD[_t, width]) -> SIMD[_t, width]:
                return val2 + rebind[SIMD[_t, width]](simd1) + simd2

            return inner2(val)

        return inner(rebind[SIMD[_t, width]](x.simd_load[width](i)))

    out.for_each[1, func]()
    return out
