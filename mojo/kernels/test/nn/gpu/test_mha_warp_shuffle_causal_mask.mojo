# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# RUN: %mojo-no-debug %s

from math import ceildiv, isclose, isqrt
from random import rand
from sys import argv

from buffer import NDBuffer
from buffer.dimlist import Dim, DimList
from gpu import *
from gpu.host import DeviceContext
from memory import UnsafePointer
from nn.mha import flash_attention, mha_gpu_naive
from nn.mha_mask import CausalMask, NullMask
from testing import assert_almost_equal

from utils.index import Index
from utils.numerics import min_or_neg_inf
from nn.mha_warp_shuffle import (
    run_mha_decoding_cpu,
    run_mha_decoding_warp_shuffle,
)
from memory import memset, memset_zero


fn is_benchmark() -> Bool:
    for arg in argv():
        if arg == "--benchmark" or arg == "-benchmark":
            return True
    return False


fn test[
    qkv_type: DType,
    depth: Int,
    num_heads: Int,
    group: Int = 1,
    batch_size: Int = 1,
](
    seq_len: Int,
    num_keys: Int,
    ctx: DeviceContext,
    is_benchmark: Bool = False,
) raises:
    print("test_mha_causal_mask")
    debug_assert(seq_len == 1)
    # Query, key, value dimensions.
    alias scale = Float32(0.125)  # rsqrt[type, 1](Float32(depth))
    alias kv_num_heads = num_heads // group
    # Q, K, V shapes.
    var q_size = batch_size * num_heads * seq_len * depth
    var k_size = batch_size * kv_num_heads * num_keys * depth
    var v_size = k_size
    var o_size = q_size

    # Allocate memory for all variables.
    var q_ptr = UnsafePointer[Scalar[qkv_type]].alloc(q_size)
    var k_ptr = UnsafePointer[Scalar[qkv_type]].alloc(k_size)
    var v_ptr = UnsafePointer[Scalar[qkv_type]].alloc(v_size)
    var output_ptr = UnsafePointer[Scalar[qkv_type]].alloc(o_size)
    var output_ptr_cpu = UnsafePointer[Scalar[qkv_type]].alloc(o_size)

    # Q, K, V are randomly initalized.
    rand[qkv_type](q_ptr, q_size)
    rand[qkv_type](k_ptr, k_size)
    rand[qkv_type](v_ptr, v_size)

    # Device pointers
    var q_device_ptr = ctx.create_buffer[qkv_type](q_size)
    var k_device_ptr = ctx.create_buffer[qkv_type](k_size)
    var v_device_ptr = ctx.create_buffer[qkv_type](v_size)
    var output_device_ptr = ctx.create_buffer[qkv_type](o_size)

    # Copy from host to device
    ctx.enqueue_copy_to_device(q_device_ptr, q_ptr)
    ctx.enqueue_copy_to_device(k_device_ptr, k_ptr)
    ctx.enqueue_copy_to_device(v_device_ptr, v_ptr)

    @parameter
    @always_inline
    fn kernel_launch(ctx: DeviceContext) raises:
        run_mha_decoding_warp_shuffle[
            head_size=depth, num_heads=num_heads, group=group
        ](
            ctx,
            q_device_ptr.ptr,
            k_device_ptr.ptr,
            v_device_ptr.ptr,
            output_device_ptr.ptr,
            scale,
            num_keys,
            CausalMask(),
            batch_size,
        )

    if is_benchmark:
        alias nrun = 50

        # Warmup
        kernel_launch(ctx)

        var nstime = ctx.execution_time[kernel_launch](nrun) / nrun
        var sectime = nstime / 1000000
        print(nrun, "runs avg", sectime, "ms")

    else:
        kernel_launch(ctx)
        # run cpu for verification
        run_mha_decoding_cpu[head_size=depth, num_heads=num_heads, group=group](
            q_ptr,
            k_ptr,
            v_ptr,
            output_ptr_cpu,
            scale,
            num_keys,
            CausalMask(),
            batch_size,
        )

        ctx.enqueue_copy_from_device(output_ptr, output_device_ptr)
        ctx.synchronize()
        var atol = Scalar[qkv_type](1e-5)
        var rtol = Scalar[qkv_type](1e-4)
        if qkv_type == DType.bfloat16:
            atol = 1e-3
            rtol = 8e-3

        for b in range(batch_size):
            for h in range(num_heads):
                for s in range(seq_len):
                    for d in range(depth):
                        var gpu = output_ptr.load(
                            d
                            + depth * (h + s * num_heads)
                            + b * num_heads * depth
                        )
                        var cpu = output_ptr_cpu.load(
                            d
                            + depth * (h + s * num_heads)
                            + b * num_heads * depth
                        )
                        if not isclose(cpu, gpu, atol=atol, rtol=rtol):
                            var rerr = abs((cpu - gpu) / gpu)
                            print("cpu vs gpu", b, h, s, d, cpu, gpu, rerr)
                        assert_almost_equal(cpu, gpu, atol=atol, rtol=rtol)
    _ = q_device_ptr
    _ = k_device_ptr
    _ = v_device_ptr
    _ = output_device_ptr
    q_ptr.free()
    k_ptr.free()
    v_ptr.free()
    output_ptr.free()
    output_ptr_cpu.free()


def main():
    with DeviceContext() as ctx:
        # BF16 token gen
        test[
            DType.bfloat16,
            128,
            32,
            group=4,
            batch_size=128,
        ](1, 512, ctx, is_benchmark())

        test[
            DType.bfloat16,
            128,
            11,
        ](1, 256, ctx, is_benchmark())

        test[
            DType.bfloat16,
            128,
            1,
        ](1, 11, ctx, is_benchmark())

        test[
            DType.bfloat16,
            128,
            2,
        ](1, 523, ctx, is_benchmark())

        test[
            DType.bfloat16,
            128,
            24,
            group=3,
        ](1, 29, ctx, is_benchmark())

        test[
            DType.bfloat16,
            128,
            3,
            group=3,
        ](1, 156, ctx, is_benchmark())

        test[
            DType.bfloat16,
            128,
            3,
            group=3,
        ](1, 208, ctx, is_benchmark())
