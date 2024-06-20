# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# REQUIRES: has_cuda_device
# RUN: %mojo-no-debug %s | FileCheck %s

from math import ceildiv, rsqrt
from random import rand
from sys import argv

from buffer import NDBuffer
from gpu import *
from gpu.host.device_context import DeviceContext
from gpu.host.event import time_function
from memory.unsafe import DTypePointer
from nn.mha import (
    _naive_attention_with_transpose,
    flash_attention_kernel,
    flash_attention_kernel_flexible_seqlen,
)

from utils.index import Index

alias type = DType.float32


fn is_benchmark() -> Bool:
    for arg in argv():
        if arg == "--benchmark" or arg == "-benchmark":
            return True
    return False


# CHECK-LABEL: test_flash_attention
fn test(
    seq_len: Int, num_keys: Int, ctx: DeviceContext, is_benchmark: Bool = False
) raises:
    print("test_flash_attention")

    # Query, key, value dimensions.
    alias batch_size = 1
    alias num_heads = 32
    alias depth = 128
    alias scale = Float32(0.125)  # rsqrt[type, 1](Float32(depth))

    # Q, K, V shapes.
    var q_size = batch_size * num_heads * seq_len * depth
    var k_size = batch_size * num_heads * num_keys * depth
    var v_size = k_size
    var o_size = q_size

    # Allocate memory for all variables.
    var q_ptr = DTypePointer[type].alloc(q_size)
    var k_ptr = DTypePointer[type].alloc(k_size)
    var v_ptr = DTypePointer[type].alloc(v_size)
    var mask_ptr = DTypePointer[type].alloc(seq_len * num_keys)
    var output_ptr = DTypePointer[type].alloc(o_size)
    var flash_output_ptr = DTypePointer[type].alloc(o_size)

    # Q, K, V are randomly initalized.
    rand[type](q_ptr, q_size)
    rand[type](k_ptr, k_size)
    rand[type](v_ptr, v_size)
    rand[type](mask_ptr, seq_len * num_keys)

    # Contruct buffers.
    var q = NDBuffer[type, 4](
        q_ptr, Index(batch_size, seq_len, num_heads, depth)
    )
    var k = NDBuffer[type, 4](
        k_ptr, Index(batch_size, num_keys, num_heads, depth)
    )
    var v = NDBuffer[type, 4](
        v_ptr, Index(batch_size, num_keys, num_heads, depth)
    )
    var mask = NDBuffer[type, 2](mask_ptr, Index(seq_len, num_keys))
    var output = NDBuffer[type, 4](
        output_ptr, Index(batch_size, seq_len, num_heads, depth)
    )

    _naive_attention_with_transpose[type](
        rebind[NDBuffer[type, 4]](output),
        rebind[NDBuffer[type, 4]](q),
        rebind[NDBuffer[type, 4]](k),
        rebind[NDBuffer[type, 4]](v),
        rebind[NDBuffer[type, 2]](mask),
        scale,
    )

    # Device pointers
    var q_device_ptr = ctx.create_buffer[type](q_size)
    var k_device_ptr = ctx.create_buffer[type](k_size)
    var v_device_ptr = ctx.create_buffer[type](v_size)
    var mask_device_ptr = ctx.create_buffer[type](seq_len * num_keys)
    var output_device_ptr = ctx.create_buffer[type](o_size)

    # Copy from host to device
    ctx.enqueue_copy_to_device(q_device_ptr, q_ptr)
    ctx.enqueue_copy_to_device(k_device_ptr, k_ptr)
    ctx.enqueue_copy_to_device(v_device_ptr, v_ptr)
    ctx.enqueue_copy_to_device(mask_device_ptr, mask_ptr)

    alias q_tile_num_rows = 32

    if seq_len == num_keys and seq_len % 128 == 0:

        @parameter
        @always_inline
        fn kernel_launch(ctx: DeviceContext) raises:
            var func = ctx.compile_function[
                flash_attention_kernel[
                    BM=32,  # q_tile_num_rows,
                    BN=128,  # kv_tile_num_rows,
                    BK=16,
                    depth=128,
                    num_heads=num_heads,
                    TM=8,
                    TN=4,
                    num_threads=128,  # q_tile_num_rows * kv_tile_num_rows,
                ]
            ]()

            ctx.enqueue_function(
                func,
                q_device_ptr,
                k_device_ptr,
                v_device_ptr,
                mask_device_ptr,
                output_device_ptr,
                scale,
                batch_size,
                seq_len,
                grid_dim=(
                    ceildiv(seq_len, q_tile_num_rows),
                    num_heads,
                    batch_size,
                ),
                block_dim=(128, 1, 1),
            )

        if is_benchmark:
            alias nrun = 1000

            # Warmup
            kernel_launch(ctx)

            var nstime = ctx.execution_time[kernel_launch](nrun) / nrun
            var sectime = nstime / 1000000
            print(nrun, "runs avg", sectime, "ms")

        else:
            kernel_launch(ctx)

    else:
        var func = ctx.compile_function[
            flash_attention_kernel_flexible_seqlen[
                BM=32,  # q_tile_num_rows,
                BN=128,  # kv_tile_num_rows,
                BK=16,
                depth=128,
                num_heads=num_heads,
                TM=8,
                TN=4,
                num_threads=128,  # q_tile_num_rows * kv_tile_num_rows,
            ]
        ]()

        ctx.enqueue_function(
            func,
            q_device_ptr,
            k_device_ptr,
            v_device_ptr,
            mask_device_ptr,
            output_device_ptr,
            scale,
            batch_size,
            seq_len,
            num_keys,
            grid_dim=(
                ceildiv(seq_len, q_tile_num_rows),
                num_heads,
                batch_size,
            ),
            block_dim=(128, 1, 1),
        )

    ctx.synchronize()

    ctx.enqueue_copy_from_device(flash_output_ptr, output_device_ptr)

    var succeed = True
    for h in range(num_heads):
        for s in range(seq_len):
            for d in range(depth):
                var expect = Scalar.load(
                    output_ptr, d + depth * (h + s * num_heads)
                )
                var actual = Scalar.load(
                    flash_output_ptr, d + depth * (h + s * num_heads)
                )
                if abs(expect - actual) > 1e-4 * abs(expect):
                    print(d, expect, actual)
                    succeed = False
                    break

    _ = q_device_ptr
    _ = k_device_ptr
    _ = v_device_ptr
    _ = mask_device_ptr
    _ = output_device_ptr

    q_ptr.free()
    k_ptr.free()
    v_ptr.free()
    mask_ptr.free()
    output_ptr.free()
    flash_output_ptr.free()

    # CHECK: Succeed
    if succeed:
        print("Succeed")


# CHECK-NOT: CUDA_ERROR
def main():
    try:
        with DeviceContext() as ctx:
            # Context encoding demo.
            test(100, 100, ctx)
            test(31, 31, ctx)
            test(
                1024, 1024, ctx, is_benchmark()
            )  # only benchmark a large shape
            # Token generation demo.
            test(1, 1, ctx)
            test(1, 2, ctx)
            test(1, 3, ctx)
            test(1, 4, ctx)
            test(1, 5, ctx)
            test(1, 6, ctx)
            test(1, 7, ctx)
            test(1, 8, ctx)
            test(1, 9, ctx)
            test(1, 10, ctx)
            test(1, 11, ctx)
            test(1, 12, ctx)
            test(1, 13, ctx)
            test(1, 14, ctx)
            test(1, 15, ctx)
            test(1, 20, ctx)
            test(1, 25, ctx)
            test(1, 30, ctx)
            test(1, 50, ctx)
            test(1, 100, ctx)
            test(1, 200, ctx)
    except e:
        print("CUDA_ERROR:", e)
