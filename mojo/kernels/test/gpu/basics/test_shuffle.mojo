# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# RUN: %mojo-no-debug %s

import gpu.warp as warp
from gpu import barrier, thread_idx
from gpu.globals import WARP_SIZE
from gpu.host import DeviceContext
from gpu.warp import shuffle_down, shuffle_idx, shuffle_up, shuffle_xor
from memory import UnsafePointer
from testing import assert_equal


fn kernel_wrapper[
    type: DType,
    simd_width: Int,
    kernel_fn: fn (SIMD[type, simd_width]) capturing -> SIMD[type, simd_width],
](device_ptr: UnsafePointer[Scalar[type]]):
    var val = device_ptr.load[width=simd_width](thread_idx.x * simd_width)
    var result = kernel_fn(val)
    barrier()

    device_ptr.store(thread_idx.x * simd_width, result)


fn _kernel_launch_helper[
    type: DType,
    simd_width: Int,
    kernel_fn: fn (SIMD[type, simd_width]) capturing -> SIMD[type, simd_width],
](
    host_ptr: UnsafePointer[Scalar[type]],
    buffer_size: Int,
    block_size: Int,
    ctx: DeviceContext,
) raises:
    var device_ptr = ctx.enqueue_create_buffer[type](buffer_size)
    ctx.enqueue_copy(device_ptr, host_ptr)

    ctx.enqueue_function[kernel_wrapper[type, simd_width, kernel_fn]](
        device_ptr, simd_width, grid_dim=1, block_dim=block_size
    )

    ctx.enqueue_copy(host_ptr, device_ptr)
    ctx.synchronize()
    _ = device_ptr


fn _shuffle_idx_launch_helper[
    type: DType, simd_width: Int
](ctx: DeviceContext) raises:
    alias block_size = WARP_SIZE
    alias buffer_size = block_size * simd_width
    alias constant_add: Scalar[type] = 42
    var host_ptr = UnsafePointer[Scalar[type]].alloc(buffer_size)

    for i in range(buffer_size):
        host_ptr[i] = i + constant_add

    @parameter
    fn do_shuffle(val: SIMD[type, simd_width]) -> SIMD[type, simd_width]:
        alias src_lane = 0
        return shuffle_idx(val, src_lane)

    _kernel_launch_helper[type, simd_width, do_shuffle](
        host_ptr, buffer_size, block_size, ctx
    )

    for i in range(block_size):
        for j in range(simd_width):
            assert_equal(host_ptr[i * simd_width + j], j + constant_add)

    host_ptr.free()


fn test_shuffle_idx_fp32(ctx: DeviceContext) raises:
    _shuffle_idx_launch_helper[DType.float32, 1](ctx)


fn test_shuffle_idx_bf16(ctx: DeviceContext) raises:
    _shuffle_idx_launch_helper[DType.bfloat16, 1](ctx)


fn test_shuffle_idx_bf16_packed(ctx: DeviceContext) raises:
    _shuffle_idx_launch_helper[DType.bfloat16, 2](ctx)


fn test_shuffle_idx_fp16(ctx: DeviceContext) raises:
    _shuffle_idx_launch_helper[DType.float16, 1](ctx)


fn test_shuffle_idx_fp16_packed(ctx: DeviceContext) raises:
    _shuffle_idx_launch_helper[DType.float16, 2](ctx)


fn test_shuffle_idx_int64(ctx: DeviceContext) raises:
    _shuffle_idx_launch_helper[DType.int64, 1](ctx)


fn _shuffle_up_launch_helper[
    type: DType, simd_width: Int
](ctx: DeviceContext) raises:
    alias block_size = WARP_SIZE
    alias buffer_size = block_size * simd_width
    alias constant_add: Scalar[type] = 42
    alias offset = WARP_SIZE // 2

    var host_ptr = UnsafePointer[Scalar[type]].alloc(buffer_size)

    for i in range(buffer_size):
        host_ptr[i] = i + constant_add

    @parameter
    fn do_shuffle(val: SIMD[type, simd_width]) -> SIMD[type, simd_width]:
        return shuffle_up(val, offset)

    _kernel_launch_helper[type, simd_width, do_shuffle](
        host_ptr, buffer_size, block_size, ctx
    )

    for i in range(block_size):
        for j in range(simd_width):
            var idx = i * simd_width + j
            if i < offset:
                assert_equal(
                    host_ptr[idx],
                    idx + constant_add,
                )
            else:
                assert_equal(
                    host_ptr[idx],
                    idx + constant_add - (offset * simd_width),
                )

    host_ptr.free()


fn test_shuffle_up_fp32(ctx: DeviceContext) raises:
    _shuffle_up_launch_helper[DType.float32, 1](ctx)


fn test_shuffle_up_bf16(ctx: DeviceContext) raises:
    _shuffle_up_launch_helper[DType.bfloat16, 1](ctx)


fn test_shuffle_up_bf16_packed(ctx: DeviceContext) raises:
    _shuffle_up_launch_helper[DType.bfloat16, 2](ctx)


fn test_shuffle_up_fp16(ctx: DeviceContext) raises:
    _shuffle_up_launch_helper[DType.float16, 1](ctx)


fn test_shuffle_up_fp16_packed(ctx: DeviceContext) raises:
    _shuffle_up_launch_helper[DType.float16, 2](ctx)


fn test_shuffle_up_int64(ctx: DeviceContext) raises:
    _shuffle_up_launch_helper[DType.int64, 1](ctx)


fn _shuffle_down_launch_helper[
    type: DType, simd_width: Int
](ctx: DeviceContext) raises:
    alias block_size = WARP_SIZE
    alias buffer_size = block_size * simd_width
    alias constant_add: Scalar[type] = 42
    alias offset = WARP_SIZE // 2

    var host_ptr = UnsafePointer[Scalar[type]].alloc(buffer_size)

    for i in range(buffer_size):
        host_ptr[i] = i + constant_add

    @parameter
    fn do_shuffle(val: SIMD[type, simd_width]) -> SIMD[type, simd_width]:
        return shuffle_down(val, offset)

    _kernel_launch_helper[type, simd_width, do_shuffle](
        host_ptr, buffer_size, block_size, ctx
    )

    for i in range(block_size):
        for j in range(simd_width):
            var idx = i * simd_width + j
            if i < offset:
                assert_equal(
                    host_ptr[idx],
                    idx + constant_add + (offset * simd_width),
                )
            else:
                assert_equal(
                    host_ptr[idx],
                    idx + constant_add,
                )

    host_ptr.free()


fn test_shuffle_down_fp32(ctx: DeviceContext) raises:
    _shuffle_down_launch_helper[DType.float32, 1](ctx)


fn test_shuffle_down_bf16(ctx: DeviceContext) raises:
    _shuffle_down_launch_helper[DType.bfloat16, 1](ctx)


fn test_shuffle_down_bf16_packed(ctx: DeviceContext) raises:
    _shuffle_down_launch_helper[DType.bfloat16, 2](ctx)


fn test_shuffle_down_fp16(ctx: DeviceContext) raises:
    _shuffle_down_launch_helper[DType.float16, 1](ctx)


fn test_shuffle_down_fp16_packed(ctx: DeviceContext) raises:
    _shuffle_down_launch_helper[DType.float16, 2](ctx)


fn test_shuffle_down_int64(ctx: DeviceContext) raises:
    _shuffle_down_launch_helper[DType.int64, 1](ctx)


fn _shuffle_xor_launch_helper[
    type: DType, simd_width: Int
](ctx: DeviceContext) raises:
    alias block_size = WARP_SIZE
    alias buffer_size = block_size * simd_width
    alias constant_add: Scalar[type] = 42
    alias offset = WARP_SIZE // 2

    var host_ptr = UnsafePointer[Scalar[type]].alloc(buffer_size)

    for i in range(buffer_size):
        host_ptr[i] = i + constant_add

    @parameter
    fn do_shuffle(val: SIMD[type, simd_width]) -> SIMD[type, simd_width]:
        return shuffle_xor(val, offset)

    _kernel_launch_helper[type, simd_width, do_shuffle](
        host_ptr, buffer_size, block_size, ctx
    )

    for i in range(block_size):
        for j in range(simd_width):
            var xor_mask = (UInt32(i) ^ UInt32(offset)).cast[type]()
            var val = xor_mask * simd_width + j + constant_add
            assert_equal(host_ptr[i * simd_width + j], val)

    host_ptr.free()


fn test_shuffle_xor_fp32(ctx: DeviceContext) raises:
    _shuffle_xor_launch_helper[DType.float32, 1](ctx)


fn test_shuffle_xor_bf16(ctx: DeviceContext) raises:
    _shuffle_xor_launch_helper[DType.bfloat16, 1](ctx)


fn test_shuffle_xor_bf16_packed(ctx: DeviceContext) raises:
    _shuffle_xor_launch_helper[DType.bfloat16, 2](ctx)


fn test_shuffle_xor_fp16(ctx: DeviceContext) raises:
    _shuffle_xor_launch_helper[DType.float16, 1](ctx)


fn test_shuffle_xor_fp16_packed(ctx: DeviceContext) raises:
    _shuffle_xor_launch_helper[DType.float16, 2](ctx)


fn test_shuffle_xor_int64(ctx: DeviceContext) raises:
    _shuffle_xor_launch_helper[DType.int64, 1](ctx)


fn _warp_reduce_launch_helper[
    type: DType,
    simd_width: Int,
](ctx: DeviceContext) raises:
    alias block_size = WARP_SIZE
    alias buffer_size = block_size * simd_width
    alias offset = 1

    var host_ptr = UnsafePointer[Scalar[type]].alloc(buffer_size)
    for i in range(buffer_size):
        host_ptr[i] = 1

    @parameter
    fn reduce_add[
        type: DType,
        width: Int,
    ](x: SIMD[type, width], y: SIMD[type, width]) -> SIMD[type, width]:
        return x + y

    @parameter
    fn do_warp_reduce(val: SIMD[type, simd_width]) -> SIMD[type, simd_width]:
        return warp.reduce[shuffle_down, reduce_add](val)

    _kernel_launch_helper[type, simd_width, do_warp_reduce](
        host_ptr, buffer_size, block_size, ctx
    )

    for i in range(simd_width):
        assert_equal(host_ptr[i], block_size)

    host_ptr.free()


fn test_warp_reduce_fp32(ctx: DeviceContext) raises:
    _warp_reduce_launch_helper[DType.float32, 1](ctx)


fn test_warp_reduce_bf16(ctx: DeviceContext) raises:
    _warp_reduce_launch_helper[DType.bfloat16, 1](ctx)


fn test_warp_reduce_bf16_packed(ctx: DeviceContext) raises:
    _warp_reduce_launch_helper[DType.bfloat16, 2](ctx)


fn test_warp_reduce_fp16(ctx: DeviceContext) raises:
    _warp_reduce_launch_helper[DType.float16, 1](ctx)


fn test_warp_reduce_fp16_packed(ctx: DeviceContext) raises:
    _warp_reduce_launch_helper[DType.float16, 2](ctx)


fn _lane_group_reduce_launch_helper[
    type: DType,
    simd_width: Int,
    num_lanes: Int,
    stride: Int,
](ctx: DeviceContext) raises:
    alias block_size = WARP_SIZE
    alias buffer_size = block_size * simd_width

    var host_ptr = UnsafePointer[Scalar[type]].alloc(buffer_size)
    for i in range(buffer_size):
        host_ptr[i] = i // simd_width

    @parameter
    fn reduce_add[
        type: DType,
        width: Int,
    ](x: SIMD[type, width], y: SIMD[type, width]) -> SIMD[type, width]:
        return x + y

    @parameter
    fn do_lane_group_reduce(
        val: SIMD[type, simd_width]
    ) -> SIMD[type, simd_width]:
        return warp.lane_group_reduce[
            shuffle_down, reduce_add, num_lanes=num_lanes, stride=stride
        ](val)

    _kernel_launch_helper[type, simd_width, do_lane_group_reduce](
        host_ptr, buffer_size, block_size, ctx
    )

    for lane in range(block_size // num_lanes):
        for i in range(simd_width):
            assert_equal(
                host_ptr[lane * simd_width + i],
                (num_lanes // 2) * (2 * lane + (num_lanes - 1) * stride),
            )

    host_ptr.free()


fn test_lane_group_reduce_fp32(ctx: DeviceContext) raises:
    _lane_group_reduce_launch_helper[DType.float32, 1, 4, 8](ctx)


fn test_lane_group_reduce_bf16(ctx: DeviceContext) raises:
    _lane_group_reduce_launch_helper[DType.bfloat16, 1, 4, 8](ctx)


fn test_lane_group_reduce_bf16_packed(ctx: DeviceContext) raises:
    _lane_group_reduce_launch_helper[DType.bfloat16, 2, 4, 8](ctx)


fn test_lane_group_reduce_fp16(ctx: DeviceContext) raises:
    _lane_group_reduce_launch_helper[DType.float16, 1, 4, 8](ctx)


fn test_lane_group_reduce_fp16_packed(ctx: DeviceContext) raises:
    _lane_group_reduce_launch_helper[DType.float16, 2, 4, 8](ctx)


fn main() raises:
    with DeviceContext() as ctx:
        test_shuffle_idx_fp32(ctx)
        test_shuffle_idx_bf16(ctx)
        test_shuffle_idx_bf16_packed(ctx)
        test_shuffle_idx_fp16(ctx)
        test_shuffle_idx_fp16_packed(ctx)
        test_shuffle_idx_int64(ctx)
        test_shuffle_up_fp32(ctx)
        test_shuffle_up_bf16(ctx)
        test_shuffle_up_bf16_packed(ctx)
        test_shuffle_up_fp16(ctx)
        test_shuffle_up_fp16_packed(ctx)
        test_shuffle_up_int64(ctx)
        test_shuffle_down_fp32(ctx)
        test_shuffle_down_bf16(ctx)
        test_shuffle_down_bf16_packed(ctx)
        test_shuffle_down_fp16(ctx)
        test_shuffle_down_fp16_packed(ctx)
        test_shuffle_down_int64(ctx)
        test_shuffle_xor_fp32(ctx)
        test_shuffle_xor_bf16(ctx)
        test_shuffle_xor_bf16_packed(ctx)
        test_shuffle_xor_fp16(ctx)
        test_shuffle_xor_fp16_packed(ctx)
        test_shuffle_xor_int64(ctx)
        test_warp_reduce_fp32(ctx)
        test_warp_reduce_bf16(ctx)
        test_warp_reduce_bf16_packed(ctx)
        test_warp_reduce_fp16(ctx)
        test_warp_reduce_fp16_packed(ctx)
        test_lane_group_reduce_fp32(ctx)
        test_lane_group_reduce_bf16(ctx)
        test_lane_group_reduce_bf16_packed(ctx)
        test_lane_group_reduce_fp16(ctx)
        test_lane_group_reduce_fp16_packed(ctx)
