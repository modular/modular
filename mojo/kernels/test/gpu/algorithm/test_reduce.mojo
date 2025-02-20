# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# RUN: %mojo-no-debug-no-assert %s

from algorithm._gpu.reduction import reduce_launch
from buffer import NDBuffer
from gpu.host import DeviceContext
from memory import UnsafePointer
from testing import assert_equal

from utils import IndexList, StaticTuple

alias num_reductions = 2


fn fused_reduce_inner_test[
    reduce_fn: fn[ty: DType, width: Int, reduction_idx: Int] (
        SIMD[ty, width], SIMD[ty, width]
    ) capturing [_] -> SIMD[ty, width],
    rank: Int,
    type: DType,
](
    shape: IndexList[rank],
    init: StaticTuple[Scalar[type], num_reductions],
    expected_vals0: List[Float32],
    expected_vals1: List[Float32],
    ctx: DeviceContext,
) raises:
    var axis = rank - 1
    var out_shape = shape
    out_shape[axis] = 1

    var in_size = shape.flattened_length()
    var out_size = out_shape.flattened_length()

    debug_assert(
        len(expected_vals0) == out_size,
        "expected vals must match output shape",
    )
    debug_assert(
        len(expected_vals1) == out_size,
        "expected vals must match output shape",
    )

    var vec_host = UnsafePointer[Scalar[type]].alloc(in_size)
    var res_host0 = UnsafePointer[Scalar[type]].alloc(out_size)
    var res_host1 = UnsafePointer[Scalar[type]].alloc(out_size)

    for i in range(in_size):
        vec_host[i] = i // shape[axis] + 1

    var vec_device = ctx.enqueue_create_buffer[type](in_size)
    var res_device0 = ctx.enqueue_create_buffer[type](out_size)
    var res_device1 = ctx.enqueue_create_buffer[type](out_size)
    var input_buf_device = NDBuffer[type, rank](vec_device.unsafe_ptr(), shape)
    var output_buf_device0 = NDBuffer[type, rank](
        res_device0.unsafe_ptr(), out_shape
    )
    var output_buf_device1 = NDBuffer[type, rank](
        res_device1.unsafe_ptr(), out_shape
    )

    ctx.enqueue_copy_to_device(vec_device, vec_host)

    @__copy_capture(input_buf_device)
    @parameter
    fn input_fn[
        type: DType,
        width: Int,
        _rank: Int,
    ](coords: IndexList[_rank]) -> SIMD[type, width]:
        return rebind[SIMD[type, width]](
            input_buf_device.load[width=width](rebind[IndexList[rank]](coords))
        )

    @__copy_capture(output_buf_device0, output_buf_device1)
    @parameter
    fn output_fn[
        _type: DType, width: Int, _rank: Int
    ](
        coords: IndexList[_rank],
        val: StaticTuple[SIMD[_type, width], num_reductions],
    ):
        output_buf_device0.__setitem__(
            rebind[IndexList[rank]](coords), rebind[Scalar[type]](val[0])
        )
        output_buf_device1.__setitem__(
            rebind[IndexList[rank]](coords), rebind[Scalar[type]](val[1])
        )

    reduce_launch[num_reductions, input_fn, output_fn, reduce_fn, rank, type](
        shape, axis, init, ctx
    )

    ctx.enqueue_copy_from_device(res_host0, res_device0)
    ctx.enqueue_copy_from_device(res_host1, res_device1)
    ctx.synchronize()

    for i in range(out_shape.flattened_length()):
        assert_equal(String(res_host0[i]), String(expected_vals0[i]))

    for i in range(out_shape.flattened_length()):
        assert_equal(String(res_host1[i]), String(expected_vals1[i]))

    _ = vec_device
    _ = res_device0
    _ = res_device1

    vec_host.free()
    res_host0.free()
    res_host1.free()


fn reduce_inner_test[
    reduce_fn: fn[type: DType, width: Int] (
        SIMD[type, width], SIMD[type, width]
    ) capturing [_] -> SIMD[type, width],
    rank: Int,
    type: DType,
    expected_vals_type: DType,
](
    shape: IndexList[rank],
    init: Scalar[type],
    expected_vals: List[Scalar[expected_vals_type]],
    ctx: DeviceContext,
) raises:
    alias num_reductions = 1

    var axis = rank - 1
    var out_shape = shape
    out_shape[axis] = 1

    var in_size = shape.flattened_length()
    var out_size = shape.flattened_length() // shape[axis]
    debug_assert(
        len(expected_vals) == out_size, "expected vals must match output shape"
    )

    var vec_host = UnsafePointer[Scalar[type]].alloc(in_size)
    var res_host = UnsafePointer[Scalar[type]].alloc(out_size)

    for i in range(in_size):
        vec_host[i] = i // shape[axis] + 1

    var vec_device = ctx.enqueue_create_buffer[type](in_size)
    var res_device = ctx.enqueue_create_buffer[type](out_size)
    var input_buf_device = NDBuffer[type, rank](vec_device.unsafe_ptr(), shape)
    var output_buf_device = NDBuffer[type, rank](
        res_device.unsafe_ptr(), out_shape
    )

    ctx.enqueue_copy_to_device(vec_device, vec_host)

    @always_inline
    @parameter
    fn reduce_wrapper[
        type: DType, width: Int, reduction_idx: Int
    ](lhs: SIMD[type, width], rhs: SIMD[type, width]) -> SIMD[type, width]:
        constrained[reduction_idx < num_reductions, "invalid reduction idx"]()

        return reduce_fn[type, width](lhs, rhs)

    @__copy_capture(input_buf_device)
    @parameter
    fn input_fn[
        type: DType,
        width: Int,
        _rank: Int,
    ](coords: IndexList[_rank]) -> SIMD[type, width]:
        return rebind[SIMD[type, width]](
            input_buf_device.load[width=width](rebind[IndexList[rank]](coords))
        )

    @__copy_capture(output_buf_device)
    @parameter
    fn output_fn[
        _type: DType, width: Int, _rank: Int
    ](
        coords: IndexList[_rank],
        val: StaticTuple[SIMD[_type, width], num_reductions],
    ):
        output_buf_device.__setitem__(
            rebind[IndexList[rank]](coords), rebind[Scalar[type]](val[0])
        )

    reduce_launch[
        num_reductions, input_fn, output_fn, reduce_wrapper, rank, type
    ](shape, axis, init, ctx)

    ctx.synchronize()
    ctx.enqueue_copy_from_device(res_host, res_device)

    for i in range(out_shape.flattened_length()):
        assert_equal(String(res_host[i]), String(expected_vals[i]))

    _ = vec_device
    _ = res_device

    vec_host.free()
    res_host.free()


def main():
    @parameter
    fn reduce_add[
        type: DType,
        width: Int,
    ](x: SIMD[type, width], y: SIMD[type, width]) -> SIMD[type, width]:
        return x + y

    @parameter
    fn reduce_max[
        type: DType,
        width: Int,
    ](x: SIMD[type, width], y: SIMD[type, width]) -> SIMD[type, width]:
        return max(x, y)

    @parameter
    fn fused_reduce_add_max[
        type: DType,
        width: Int,
        reduction_idx: Int,
    ](x: SIMD[type, width], y: SIMD[type, width]) -> SIMD[type, width]:
        constrained[reduction_idx < 2, "reduction idx OOB"]()

        alias func = reduce_max if reduction_idx == 0 else reduce_add
        return func(x, y)

    with DeviceContext() as ctx:
        reduce_inner_test[reduce_add](
            IndexList[3](2, 3, 257),
            Float32(0),
            List[Float32](257.0, 514.0, 771.0, 1028.0, 1285.0, 1542.0),
            ctx,
        )

        reduce_inner_test[reduce_add](
            IndexList[2](5, 257),
            Float32(0),
            List[Float32](257.0, 514.0, 771.0, 1028.0, 1285.0),
            ctx,
        )

        reduce_inner_test[reduce_add](
            IndexList[4](2, 2, 2, 1029),
            Float32(0),
            List[Float32](
                1029.0,
                2058.0,
                3087.0,
                4116.0,
                5145.0,
                6174.0,
                7203.0,
                8232.0,
            ),
            ctx,
        )

        reduce_inner_test[reduce_max](
            IndexList[2](5, 3),
            Float32.MIN,
            List[Float32](1.0, 2.0, 3.0, 4.0, 5.0),
            ctx,
        )

        fused_reduce_inner_test[fused_reduce_add_max, 2, DType.float32](
            IndexList[2](5, 3),
            StaticTuple[Float32, 2](Float32.MIN, 0.0),
            List[Float32](1.0, 2.0, 3.0, 4.0, 5.0),
            List[Float32](3.0, 6.0, 9.0, 12.0, 15.0),
            ctx,
        )

        # bf16 tests
        reduce_inner_test[reduce_max](
            IndexList[2](5, 5),
            BFloat16.MIN,
            List[Float32](1.0, 2.0, 3.0, 4.0, 5.0),
            ctx,
        )

        fused_reduce_inner_test[fused_reduce_add_max, 2, DType.bfloat16](
            IndexList[2](5, 3),
            StaticTuple[BFloat16, 2](BFloat16.MIN, 0.0),
            List[Float32](1.0, 2.0, 3.0, 4.0, 5.0),
            List[Float32](3.0, 6.0, 9.0, 12.0, 15.0),
            ctx,
        )

        # fp16 tests
        reduce_inner_test[reduce_max](
            IndexList[2](5, 5),
            Float16.MIN,
            List[Float32](1.0, 2.0, 3.0, 4.0, 5.0),
            ctx,
        )

        fused_reduce_inner_test[fused_reduce_add_max, 2, DType.float16](
            IndexList[2](5, 3),
            StaticTuple[Float16, 2](Float16.MIN, 0.0),
            List[Float32](1.0, 2.0, 3.0, 4.0, 5.0),
            List[Float32](3.0, 6.0, 9.0, 12.0, 15.0),
            ctx,
        )

        # bool tests
        reduce_inner_test[reduce_max](
            IndexList[2](5, 5),
            Scalar[DType.bool].MIN,
            List[Scalar[DType.bool]](True, False, True, False, True),
            ctx,
        )
