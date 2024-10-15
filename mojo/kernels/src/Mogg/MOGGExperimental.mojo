# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

from collections import OptionalReg
from math import ceil, erf, exp, floor, iota, isqrt, log, log1p, sqrt, tanh

from algorithm.reduction import _reduce_generator
from buffer.dimlist import DimList
from memory import UnsafePointer
from MOGGIntList import IntList
from MOGGTensor import Tensor
from nn.activations import relu
from register import *
from runtime.tracing import Trace, TraceLevel

from utils import IndexList, unroll
from utils.numerics import isinf, isnan

alias MAX_BENEFIT = 1000


@mogg_tensor_allocator()
@no_inline
@export
fn empty_tensor[
    type: DType,
](shape: IntList) -> Tensor[type, shape.static_values]:
    var ptr = UnsafePointer[Scalar[type]].alloc(shape.nelems())
    var ref_cnt = UnsafePointer[Scalar[DType.index]].alloc(1)
    ref_cnt[0] = 0
    return Tensor[type, shape.static_values](ptr, shape, ref_cnt)


@mogg_tensor_allocator()
@no_inline
@export
fn empty_strided_tensor[
    type: DType,
](shape: IntList, strides: IntList) -> Tensor[
    type, shape.static_values, strides.static_values
]:
    var ptr = UnsafePointer[Scalar[type]].alloc(shape.nelems())
    var ref_cnt = UnsafePointer[Scalar[DType.index]].alloc(1)
    ref_cnt[0] = 0
    return Tensor[type, shape.static_values, strides.static_values](
        ptr, shape, ref_cnt
    )


@mogg_register("unary_op_without_fusion")
@export
fn unary_op_without_fusion(
    x: Tensor,
) -> Tensor[x.type, x.static_shape]:
    var out = empty_tensor[x.type](x.shape)

    @parameter
    @always_inline
    fn func[width: Int](i: IntList) -> SIMD[out.type, width]:
        return sqrt(rebind[SIMD[out.type, width]](x.simd_load[width](i)))

    out.for_each[func]()
    return out


@mogg_register_override("mo.transpose", MAX_BENEFIT)
@export
fn transpose[
    out_static_strides: DimList,
](
    x: Tensor,
    perm: Tensor,
) -> Tensor[
    x.type, x.same_rank_param(), out_static_strides
]:
    # Currently we don't support alias.
    # alias rank = x.static_rank

    var new_shape = IntList[x.same_rank_param()]()
    var new_stride = IntList[x.same_rank_param()]()

    @always_inline
    @parameter
    fn body[i: Int]():
        var index = IntList[DimList(i)](i)
        var dim = int(perm.simd_load[1](index))
        new_shape[i] = x.shape[dim]
        new_stride[i] = x.strides[dim]

    unroll[body, x.static_rank.value()]()

    return Tensor[x.type, x.same_rank_param(), out_static_strides](
        x.data, new_shape, new_stride, x.refcount()
    )


@mogg_register_override("copy", MAX_BENEFIT)
@export
fn copy(x: Tensor) -> Tensor[x.type, x.static_shape]:
    var out = empty_tensor[x.type](x.shape)

    @parameter
    @always_inline
    fn func[width: Int](i: IntList) -> SIMD[out.type, width]:
        return rebind[SIMD[out.type, width]](x.simd_load[width](i))

    out.for_each[func]()
    return out


# Test we support a nested lambda using values from the parent contexts.
@mogg_register_override("add_like_custom_op_target", MAX_BENEFIT)
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
    fn func[width: Int](i: IntList) -> SIMD[out.type, width]:
        var i2 = rebind[SIMD[x.type, width]](y.simd_load[width](i))
        return rebind[SIMD[out.type, width]](x.simd_load[width](i) + i2)

    out.for_each[func]()
    return out


@mogg_register("recursive_lambda_test_target")
@export
fn recursive_lambda_test_target(x: Tensor) -> Tensor[x.type, x.static_shape]:
    var out = empty_tensor[x.type](x.shape)
    var simd1 = SIMD[out.type, 1](0.5)

    @parameter
    @always_inline
    fn func[width: Int](i: IntList) -> SIMD[out.type, width]:
        var simd2 = SIMD[out.type, width](4.0)

        @parameter
        @always_inline
        fn inner(val: SIMD[out.type, width]) -> SIMD[out.type, width]:
            @parameter
            @always_inline
            fn inner2(val2: SIMD[out.type, width]) -> SIMD[out.type, width]:
                return val2 + simd1 + simd2

            return inner2(val)

        return inner(rebind[SIMD[out.type, width]](x.simd_load[width](i)))

    out.for_each[func]()
    _ = simd1
    return out


fn multiparam_user(input_rank: Int, indices_rank: Int) -> Int:
    return 5


fn dummy_user(x: Tensor):
    pass


@mogg_register("param_expression_shape_test")
@always_inline
@export
fn param_expression_shape_test(
    input1: Tensor,
    input2: Tensor,
) -> Tensor[
    input1.type,
    DimList.create_unknown[
        multiparam_user(input1.static_rank.value(), input2.static_rank.value())
    ](),
]:
    var shape = IntList[
        DimList.create_unknown[
            multiparam_user(
                input1.static_rank.value(), input2.static_rank.value()
            )
        ]()
    ](3, 2, 1, 3, 2)
    var output = empty_tensor[input1.type](shape)

    # Dummy user to make sure mogg has to materialize the toKGEN.
    dummy_user(output)
    return output


@mogg_register_override("view_like_custom_op_target", MAX_BENEFIT)
@export
fn view_like_custom_op_target[
    out_static_strides: DimList,
](x: Tensor, y: Tensor) -> Tensor[
    x.type, x.same_rank_param(), out_static_strides
]:
    var new_shape = IntList[x.same_rank_param()]()
    var new_stride = IntList[x.same_rank_param()]()

    @always_inline
    @parameter
    fn body[i: Int]():
        new_shape[i] = x.shape[i] * y.shape[i]
        new_stride[i] = 0

    unroll[body, x.static_rank.value()]()

    var strides = IntList[x.static_strides](x.strides)

    return Tensor[x.type, x.same_rank_param(), out_static_strides](
        x.data, new_shape, new_stride, x.refcount()
    )


@mogg_register_override("mo.static.broadcast_to", MAX_BENEFIT)
@export
fn broadcast[
    rank: Int,
    out_static_strides: DimList,
](x: Tensor, shape: IndexList[rank]) -> Tensor[
    x.type, DimList.create_unknown[shape.size](), out_static_strides
]:
    var new_shape = IntList[DimList.create_unknown[shape.size]()]()
    var new_stride = IntList[DimList.create_unknown[shape.size]()]()

    var delta = shape.size - x.rank()

    @always_inline
    @__copy_capture(delta)
    @parameter
    fn body[i: Int]():
        new_shape[i] = shape[i]

        if i < delta:
            new_stride[i] = 0
        elif x.shape[i - delta] <= 1:
            new_stride[i] = 0
        else:
            new_stride[i] = x.strides[i - delta]

    unroll[body, shape.size]()

    return Tensor[
        x.type, DimList.create_unknown[shape.size](), out_static_strides
    ](x.data, new_shape, new_stride, x.refcount())


# ===----------------------------------------------------------------------===#
# Unary elementwise op
# ===----------------------------------------------------------------------===#


@mogg_register_override("mo.abs", MAX_BENEFIT)
@export
fn mo_abs(
    x: Tensor,
) -> Tensor[x.type, x.static_shape]:
    var out = empty_tensor[x.type](x.shape)

    x.enable_fusion()
    out.enable_fusion()

    @parameter
    @always_inline
    fn func[width: Int](i: IntList) -> SIMD[out.type, width]:
        return abs(x.simd_load[width](i))

    out.for_each[func]()
    return out


@mogg_register_override("mo.ceil", MAX_BENEFIT)
@export
fn mo_ceil(
    x: Tensor,
) -> Tensor[x.type, x.static_shape]:
    var out = empty_tensor[x.type](x.shape)

    x.enable_fusion()
    out.enable_fusion()

    @parameter
    @always_inline
    fn func[width: Int](i: IntList) -> SIMD[out.type, width]:
        return ceil(x.simd_load[width](i))

    out.for_each[func]()
    return out


@mogg_register_override("mo.erf", MAX_BENEFIT)
@export
fn mo_erf(
    x: Tensor,
) -> Tensor[x.type, x.static_shape]:
    var out = empty_tensor[x.type](x.shape)

    x.enable_fusion()
    out.enable_fusion()

    @parameter
    @always_inline
    fn func[width: Int](i: IntList) -> SIMD[out.type, width]:
        return erf(x.simd_load[width](i))

    out.for_each[func]()
    return out


@mogg_register_override("mo.exp", MAX_BENEFIT)
@export
fn mo_exp(
    x: Tensor,
) -> Tensor[x.type, x.static_shape]:
    var out = empty_tensor[x.type](x.shape)

    x.enable_fusion()
    out.enable_fusion()

    @parameter
    @always_inline
    fn func[width: Int](i: IntList) -> SIMD[out.type, width]:
        return exp(x.simd_load[width](i))

    out.for_each[func]()
    return out


@mogg_register_override("mo.floor", MAX_BENEFIT)
@export
fn mo_floor(
    x: Tensor,
) -> Tensor[x.type, x.static_shape]:
    var out = empty_tensor[x.type](x.shape)

    x.enable_fusion()
    out.enable_fusion()

    @parameter
    @always_inline
    fn func[width: Int](i: IntList) -> SIMD[out.type, width]:
        return floor(x.simd_load[width](i))

    out.for_each[func]()
    return out


@mogg_register_override("mo.is_inf", MAX_BENEFIT)
@export
fn mo_is_inf(
    x: Tensor,
) -> Tensor[DType.bool, x.static_shape]:
    var out = empty_tensor[DType.bool](x.shape)

    x.enable_fusion()
    out.enable_fusion()

    @parameter
    @always_inline
    fn func[width: Int](i: IntList) -> SIMD[out.type, width]:
        return rebind[SIMD[out.type, width]](isinf(x.simd_load[width](i)))

    out.for_each[func]()
    return out


@mogg_register_override("mo.is_nan", MAX_BENEFIT)
@export
fn mo_is_nan(
    x: Tensor,
) -> Tensor[DType.bool, x.static_shape]:
    var out = empty_tensor[DType.bool](x.shape)

    x.enable_fusion()
    out.enable_fusion()

    @parameter
    @always_inline
    fn func[width: Int](i: IntList) -> SIMD[out.type, width]:
        return rebind[SIMD[out.type, width]](isnan(x.simd_load[width](i)))

    out.for_each[func]()
    return out


@mogg_register_override("mo.log", MAX_BENEFIT)
@export
fn mo_log(
    x: Tensor,
) -> Tensor[x.type, x.static_shape]:
    var out = empty_tensor[x.type](x.shape)

    x.enable_fusion()
    out.enable_fusion()

    @parameter
    @always_inline
    fn func[width: Int](i: IntList) -> SIMD[out.type, width]:
        return log(x.simd_load[width](i))

    out.for_each[func]()
    return out


@mogg_register_override("mo.log1p", MAX_BENEFIT)
@export
fn mo_log1p(
    x: Tensor,
) -> Tensor[x.type, x.static_shape]:
    var out = empty_tensor[x.type](x.shape)

    x.enable_fusion()
    out.enable_fusion()

    @parameter
    @always_inline
    fn func[width: Int](i: IntList) -> SIMD[out.type, width]:
        return log1p(x.simd_load[width](i))

    out.for_each[func]()
    return out


@mogg_register_override("mo.relu", MAX_BENEFIT)
@export
fn mo_relu(
    x: Tensor,
) -> Tensor[x.type, x.static_shape]:
    var out = empty_tensor[x.type](x.shape)

    x.enable_fusion()
    out.enable_fusion()

    @parameter
    @always_inline
    fn func[width: Int](i: IntList) -> SIMD[out.type, width]:
        return relu(x.simd_load[width](i))

    out.for_each[func]()
    return out


@mogg_register_override("mo.isqrt", MAX_BENEFIT)
@export
fn mo_isqrt(
    x: Tensor,
) -> Tensor[x.type, x.static_shape]:
    var out = empty_tensor[x.type](x.shape)

    x.enable_fusion()
    out.enable_fusion()

    @parameter
    @always_inline
    fn func[width: Int](i: IntList) -> SIMD[out.type, width]:
        return isqrt(rebind[SIMD[out.type, width]](x.simd_load[width](i)))

    out.for_each[func]()
    return out


@mogg_register_override("mo.sqrt", MAX_BENEFIT)
@export
fn mo_sqrt(
    x: Tensor,
) -> Tensor[x.type, x.static_shape]:
    var out = empty_tensor[x.type](x.shape)

    x.enable_fusion()
    out.enable_fusion()

    @parameter
    @always_inline
    fn func[width: Int](i: IntList) -> SIMD[out.type, width]:
        return sqrt(rebind[SIMD[out.type, width]](x.simd_load[width](i)))

    out.for_each[func]()
    return out


@mogg_register_override("mo.tanh", MAX_BENEFIT)
@export
fn mo_tanh(
    x: Tensor,
) -> Tensor[x.type, x.static_shape]:
    var out = empty_tensor[x.type](x.shape)

    x.enable_fusion()
    out.enable_fusion()

    @parameter
    @always_inline
    fn func[width: Int](i: IntList) -> SIMD[out.type, width]:
        return rebind[SIMD[out.type, width]](tanh(x.simd_load[width](i)))

    out.for_each[func]()
    return out


# ===----------------------------------------------------------------------===#
# Binary elementwise op
# ===----------------------------------------------------------------------===#


@mogg_register_override("mo.add", MAX_BENEFIT)
@export
fn mo_add(x: Tensor, y: Tensor) -> Tensor[x.type, x.static_shape]:
    var out = empty_tensor[x.type](x.shape)

    x.enable_fusion()
    y.enable_fusion()
    out.enable_fusion()

    @parameter
    @always_inline
    fn func[width: Int](i: IntList) -> SIMD[x.type, width]:
        var i2 = rebind[SIMD[x.type, width]](y.simd_load[width](i))
        return x.simd_load[width](i) + i2

    out.for_each[func]()
    return out


@mogg_register_override("mo.and", MAX_BENEFIT)
@export
fn mo_and(x: Tensor, y: Tensor) -> Tensor[DType.bool, x.static_shape]:
    var out = empty_tensor[DType.bool](x.shape)

    x.enable_fusion()
    y.enable_fusion()
    out.enable_fusion()

    @parameter
    @always_inline
    fn func[width: Int](i: IntList) -> SIMD[DType.bool, width]:
        var i2 = rebind[SIMD[x.type, width]](y.simd_load[width](i))
        return rebind[SIMD[out.type, width]](x.simd_load[width](i) & i2)

    out.for_each[func]()
    return out


@mogg_register_override("mo.or", MAX_BENEFIT)
@export
fn mo_or(x: Tensor, y: Tensor) -> Tensor[DType.bool, x.static_shape]:
    var out = empty_tensor[DType.bool](x.shape)

    x.enable_fusion()
    y.enable_fusion()
    out.enable_fusion()

    @parameter
    @always_inline
    fn func[width: Int](i: IntList) -> SIMD[DType.bool, width]:
        var i2 = rebind[SIMD[x.type, width]](y.simd_load[width](i))
        return rebind[SIMD[out.type, width]](x.simd_load[width](i) | i2)

    out.for_each[func]()
    return out


@mogg_register_override("mo.equal", MAX_BENEFIT)
@export
fn mo_equal(x: Tensor, y: Tensor) -> Tensor[DType.bool, x.static_shape]:
    var out = empty_tensor[DType.bool](x.shape)

    x.enable_fusion()
    y.enable_fusion()
    out.enable_fusion()

    @parameter
    @always_inline
    fn func[width: Int](i: IntList) -> SIMD[out.type, width]:
        var i2 = rebind[SIMD[x.type, width]](y.simd_load[width](i))
        return rebind[SIMD[out.type, width]](x.simd_load[width](i) == i2)

    out.for_each[func]()
    return out


@mogg_register_override("mo.greater", MAX_BENEFIT)
@export
fn mo_greater(x: Tensor, y: Tensor) -> Tensor[DType.bool, x.static_shape]:
    var out = empty_tensor[DType.bool](x.shape)

    x.enable_fusion()
    y.enable_fusion()
    out.enable_fusion()

    @parameter
    @always_inline
    fn func[width: Int](i: IntList) -> SIMD[out.type, width]:
        var i1 = x.simd_load[width](i)
        var i2 = rebind[SIMD[x.type, width]](y.simd_load[width](i))
        return rebind[SIMD[out.type, width]](i1 > i2)

    out.for_each[func]()
    return out


@mogg_register_override("mo.max", MAX_BENEFIT)
@export
fn mo_max(x: Tensor, y: Tensor) -> Tensor[x.type, x.static_shape]:
    var out = empty_tensor[x.type](x.shape)

    x.enable_fusion()
    y.enable_fusion()
    out.enable_fusion()

    @parameter
    @always_inline
    fn func[width: Int](i: IntList) -> SIMD[out.type, width]:
        var i1 = rebind[SIMD[out.type, width]](x.simd_load[width](i))
        var i2 = rebind[SIMD[out.type, width]](y.simd_load[width](i))
        return max(i1, i2)

    out.for_each[func]()
    return out


@mogg_register_override("mo.min", MAX_BENEFIT)
@export
fn mo_min(x: Tensor, y: Tensor) -> Tensor[x.type, x.static_shape]:
    var out = empty_tensor[x.type](x.shape)

    x.enable_fusion()
    y.enable_fusion()
    out.enable_fusion()

    @parameter
    @always_inline
    fn func[width: Int](i: IntList) -> SIMD[out.type, width]:
        var i1 = rebind[SIMD[out.type, width]](x.simd_load[width](i))
        var i2 = rebind[SIMD[out.type, width]](y.simd_load[width](i))
        return min(i1, i2)

    out.for_each[func]()
    return out


@mogg_register_override("mo.mul", MAX_BENEFIT)
@export
fn mo_mul(x: Tensor, y: Tensor) -> Tensor[x.type, x.static_shape]:
    var out = empty_tensor[x.type](x.shape)

    x.enable_fusion()
    y.enable_fusion()
    out.enable_fusion()

    @parameter
    @always_inline
    fn func[width: Int](i: IntList) -> SIMD[out.type, width]:
        var i1 = rebind[SIMD[out.type, width]](x.simd_load[width](i))
        var i2 = rebind[SIMD[out.type, width]](y.simd_load[width](i))
        return i1 * i2

    out.for_each[func]()
    return out


@mogg_register_override("mo.not_equal", MAX_BENEFIT)
@export
fn mo_not_equal(x: Tensor, y: Tensor) -> Tensor[DType.bool, x.static_shape]:
    var out = empty_tensor[DType.bool](x.shape)

    x.enable_fusion()
    y.enable_fusion()
    out.enable_fusion()

    @parameter
    @always_inline
    fn func[width: Int](i: IntList) -> SIMD[out.type, width]:
        var i2 = rebind[SIMD[x.type, width]](y.simd_load[width](i))
        return rebind[SIMD[out.type, width]](x.simd_load[width](i) != i2)

    out.for_each[func]()
    return out


@mogg_register_override("mo.sub", MAX_BENEFIT)
@export
fn mo_sub(x: Tensor, y: Tensor) -> Tensor[x.type, x.static_shape]:
    var out = empty_tensor[x.type](x.shape)

    x.enable_fusion()
    y.enable_fusion()
    out.enable_fusion()

    @parameter
    @always_inline
    fn func[width: Int](i: IntList) -> SIMD[out.type, width]:
        var i2 = rebind[SIMD[x.type, width]](y.simd_load[width](i))
        return rebind[SIMD[out.type, width]](x.simd_load[width](i) - i2)

    out.for_each[func]()
    return out


# ===----------------------------------------------------------------------===#
# Tertiary elementwise op
# ===----------------------------------------------------------------------===#


@mogg_register_override("mo.select", MAX_BENEFIT)
@export
fn mo_select(
    cond: Tensor, x: Tensor, y: Tensor
) -> Tensor[x.type, x.static_shape]:
    var out = empty_tensor[x.type](x.shape)

    cond.enable_fusion()
    x.enable_fusion()
    y.enable_fusion()
    out.enable_fusion()

    @parameter
    @always_inline
    fn func[width: Int](i: IntList) -> SIMD[out.type, width]:
        var c = cond.simd_load[width](i)
        var i2 = rebind[SIMD[x.type, width]](y.simd_load[width](i))
        return rebind[SIMD[out.type, width]](
            c.select(x.simd_load[width](i), i2)
        )

    out.for_each[func]()
    return out


@mogg_register_override("mo.range", MAX_BENEFIT)
@export
fn mo_range[
    out_type: DType, out_shape: DimList
](start: Tensor, stop: Tensor, step: Tensor) -> Tensor[out_type, out_shape]:
    var start_ = start.simd_load[1](0)
    var stop_ = rebind[SIMD[start.type, 1]](stop.simd_load[1](0))
    var step_ = rebind[SIMD[start.type, 1]](step.simd_load[1](0))

    var shape: IntList[out_shape]

    @parameter
    if start.type.is_integral():
        shape = IndexList[1](len(range(int(start_), int(stop_), int(step_))))
    else:
        shape = IndexList[1](int(ceil(abs(stop_ - start_) / abs(step_))))

    var out = empty_tensor[out_type](shape)

    @parameter
    @always_inline
    fn func[width: Int](i: IntList) -> SIMD[out.type, width]:
        return rebind[SIMD[out.type, 1]](start_[0]) + (
            iota[out.type, width](i[0]) * rebind[SIMD[out.type, 1]](step_)
        )

    out.for_each[func]()
    _ = step_
    _ = start_
    return out


# ===----------------------------------------------------------------------===#
# reduce op
# ===----------------------------------------------------------------------===#


@always_inline
fn _add(x: SIMD, y: __type_of(x)) -> __type_of(x):
    return x + y


@always_inline
fn _mul(x: SIMD, y: __type_of(x)) -> __type_of(x):
    return x * y


@always_inline
fn _get_reduce_output_shape(
    input: Tensor, axis: Tensor
) -> IntList[DimList.create_unknown[input.static_rank.value()]()]:
    var output = IntList[DimList.create_unknown[input.static_rank.value()]()](
        input.shape
    )
    output[int(axis.simd_load[1](IntList(0)))] = 1
    return output


@always_inline
fn _reduce_wrapper[
    type: DType,
    reduce_op: fn[ty: DType, width: Int] (
        arg1: SIMD[ty, width], arg2: SIMD[ty, width]
    ) -> SIMD[ty, width],
    /,
    single_thread_blocking_override: Bool = False,
    target: StringLiteral = "cpu",
](
    input: Tensor, inout output: Tensor, init: Scalar[type], ax: Int
) raises -> None:
    constrained[
        input.static_rank.__bool__(),
        "reduce kernel does not support dynamic rank inputs",
    ]()

    @parameter
    @always_inline
    fn reduce_impl[
        ty: DType, width: Int
    ](v1: SIMD[ty, width], v2: SIMD[ty, width]) -> SIMD[ty, width]:
        return reduce_op(v1, v2)

    @parameter
    @always_inline
    fn load_input[
        ty: DType, width: Int, rank: Int
    ](coords: IndexList[rank]) -> SIMD[ty, width]:
        return rebind[SIMD[ty, width]](
            input.simd_load[width](
                rebind[IndexList[input.static_rank.value()]](coords)
            )
        )

    @parameter
    @always_inline
    fn store_output[
        ty: DType, width: Int, rank: Int
    ](coords: IndexList[rank], val: SIMD[ty, width]) -> None:
        output.store(
            rebind[IndexList[output.static_rank.value()]](coords),
            rebind[SIMD[output.type, width]](val),
        )

    with Trace[TraceLevel.OP, target="cpu"]("reduce"):
        _reduce_generator[
            load_input,
            store_output,
            reduce_impl,
            target=target,
            single_thread_blocking_override=single_thread_blocking_override,
        ](
            input.shape.to_static_tuple(),
            init,
            ax,
        )


@mogg_register_override("mo.reduce_add", priority=MAX_BENEFIT)
@always_inline
@export
fn reduce_add[
    single_thread_blocking_override: Bool = False,
    target: StringLiteral = "cpu",
](
    input: Tensor,
    axis: Tensor,
) raises -> Tensor[
    input.type,
    DimList.create_unknown[input.static_rank.value()](),
]:
    var ax = int(axis.simd_load[1](0))

    var output_shape = _get_reduce_output_shape(input, axis)
    var output = empty_tensor[input.type](
        IntList[DimList.create_unknown[input.static_rank.value()]()](
            output_shape
        )
    )

    output.enable_fusion()
    input.enable_fusion()

    _reduce_wrapper[
        input.type,
        _add,
        single_thread_blocking_override=single_thread_blocking_override,
        target=target,
    ](input, output, Scalar[input.type](0), ax)

    return output


@mogg_register_override("mo.reduce_max", priority=MAX_BENEFIT)
@always_inline
@export
fn reduce_max[
    single_thread_blocking_override: Bool = False,
    target: StringLiteral = "cpu",
](
    input: Tensor,
    axis: Tensor,
) raises -> Tensor[
    input.type,
    DimList.create_unknown[input.static_rank.value()](),
]:
    var ax = int(axis.simd_load[1](0))

    var output_shape = _get_reduce_output_shape(input, axis)
    var output = empty_tensor[input.type](
        IntList[DimList.create_unknown[input.static_rank.value()]()](
            output_shape
        )
    )

    output.enable_fusion()
    input.enable_fusion()

    _reduce_wrapper[
        input.type,
        max,
        single_thread_blocking_override=single_thread_blocking_override,
        target=target,
    ](input, output, Scalar[input.type].MIN, ax)

    return output


@mogg_register_override("mo.reduce_min", priority=MAX_BENEFIT)
@always_inline
@export
fn reduce_min[
    single_thread_blocking_override: Bool = False,
    target: StringLiteral = "cpu",
](
    input: Tensor,
    axis: Tensor,
) raises -> Tensor[
    input.type,
    DimList.create_unknown[input.static_rank.value()](),
]:
    var ax = int(axis.simd_load[1](0))

    var output_shape = _get_reduce_output_shape(input, axis)
    var output = empty_tensor[input.type](
        IntList[DimList.create_unknown[input.static_rank.value()]()](
            output_shape
        )
    )

    output.enable_fusion()
    input.enable_fusion()

    _reduce_wrapper[
        input.type,
        min,
        single_thread_blocking_override=single_thread_blocking_override,
        target=target,
    ](input, output, Scalar[input.type].MAX, ax)

    return output


@mogg_register_override("mo.reduce_mul", priority=MAX_BENEFIT)
@always_inline
@export
fn reduce_mul[
    single_thread_blocking_override: Bool = False,
    target: StringLiteral = "cpu",
](
    input: Tensor,
    axis: Tensor,
) raises -> Tensor[
    input.type,
    DimList.create_unknown[input.static_rank.value()](),
]:
    var ax = int(axis.simd_load[1](0))

    var output_shape = _get_reduce_output_shape(input, axis)
    var output = empty_tensor[input.type](
        IntList[DimList.create_unknown[input.static_rank.value()]()](
            output_shape
        )
    )
    input.enable_fusion()
    output.enable_fusion()

    _reduce_wrapper[
        input.type,
        _mul,
        single_thread_blocking_override=single_thread_blocking_override,
        target=target,
    ](input, output, Scalar[input.type](1), ax)

    return output


# ===----------------------------------------------------------------------===#
# Test utils
# ===----------------------------------------------------------------------===#


@mogg_register_override("single_thread_blocking_override_test", MAX_BENEFIT)
@export
fn single_thread_blocking_override_test[
    single_thread_blocking_override: Bool
](x: Tensor) -> Tensor[x.type, x.static_shape, x.static_strides]:
    @parameter
    if single_thread_blocking_override:
        print("Running with a single blocking thread")
    else:
        print("Running in async mode")

    return Tensor[x.type, x.static_shape, x.static_strides](
        x.data, x.shape, x.strides, x.refcount()
    )


@mogg_register_override("test_static_shape", MAX_BENEFIT)
@export
fn test_static_shape(
    x: Tensor,
) -> Tensor[x.type, x.static_shape]:
    var out = empty_tensor[x.type](x.shape)
    var out_shape = IntList[x.static_shape](x.shape)

    @always_inline
    @parameter
    fn print_if_static[idx: Int]():
        @parameter
        if out_shape.shape_idx_statically_known[idx]():
            print(idx)

    unroll[print_if_static, x.static_rank.value()]()

    return out


@mogg_register_override("test_static_strides", MAX_BENEFIT)
@export
fn test_static_strides(
    x: Tensor,
) -> Tensor[x.type, x.static_shape]:
    var out = empty_tensor[x.type](x.shape)
    var out_strides = IntList[x.static_strides](x.strides)

    @always_inline
    @parameter
    fn print_if_static[idx: Int]():
        @parameter
        if out_strides.shape_idx_statically_known[idx]():
            print("static_strides[", idx, "] = ", out_strides[idx])
        else:
            print("static_strides[", idx, "] = dyn")

    unroll[print_if_static, x.static_rank.value()]()

    return out


@mogg_register_override("mo.slice", 1000)
@export
fn slice_impl[
    out_static_strides: DimList
](x: Tensor, starts: Tensor, stops: Tensor, steps: Tensor) -> Tensor[
    x.type, x.same_rank_param(), out_static_strides
]:
    # The data does not change however we will be addressing a different
    # offset of the data.
    var new_data = x.data
    var new_shape = IntList[x.same_rank_param()]()
    var new_stride = IntList[x.same_rank_param()]()

    for i in range(x.rank()):
        var start = int(starts.simd_load[1](i))
        var stop = int(stops.simd_load[1](i))
        var step = int(steps.simd_load[1](i))
        var dim_i = x.shape[i]
        debug_assert(step != 0, "step must be nonzero")

        # Normalize the start/stop indices
        if start < 0:
            start = start + dim_i
        if stop < 0:
            stop = stop + dim_i

        # Compute the min/max for clamping start/end
        var idx_min = 0 if step > 0 else -1
        var idx_max = dim_i if step > 0 else dim_i - 1

        # Allow start and stop to truncate like numpy and torch allow.
        if start < idx_min:
            start = idx_min
        elif start > idx_max:
            start = idx_max

        if stop < idx_min:
            stop = idx_min
        elif stop > idx_max:
            stop = idx_max

        var new_offset = start * x.strides[i]
        new_data = new_data.offset(new_offset)

        # Stride == number of elements to the next index in this dimension.
        # So to step we can just increase the stride.
        new_stride[i] = x.strides[i] * step

        # If the steps are positive we traverse from start, if negative from
        # stop.
        new_shape[i] = len(range(start, stop, step))

    return Tensor[x.type, x.same_rank_param(), out_static_strides](
        new_data, new_shape, new_stride, x.refcount()
    )


@mogg_register_override("mo.static.reshape", 1000)
@export
fn reshape_impl[
    out_static_strides: DimList
](x: Tensor, shape: Tensor) -> Tensor[
    x.type, shape.same_rank_param(), out_static_strides
]:
    var stride = IntList[DimList.create_unknown[shape.static_rank.value()]()]()
    var accumulator: Int = 1

    @always_inline
    @parameter
    fn body[i: Int]():
        var idx = shape.static_rank.value() - i - 1
        stride[idx] = accumulator
        accumulator *= shape.shape[idx]

    unroll[body, shape.static_rank.value()]()

    return Tensor[x.type, shape.same_rank_param(), out_static_strides](
        x.data, shape.shape, stride, x.refcount()
    )


@mogg_register("test_my_abs")
@export
fn test_my_abs(
    x: Tensor,
) -> Tensor[x.type]:
    var new_shape = IntList(3, 2)
    var ptr = UnsafePointer[Scalar[x.type]].alloc(new_shape.nelems())
    var ref_cnt = UnsafePointer[Scalar[DType.index]].alloc(1)
    ref_cnt[0] = 0
    var out = Tensor[x.type](ptr, new_shape, ref_cnt)

    @parameter
    @always_inline
    fn func[width: Int](i: IntList) -> SIMD[out.type, width]:
        return abs(x.simd_load[width](i)) + SIMD[out.type, width](1)

    out.for_each[func]()
    return out
