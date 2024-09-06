# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# REQUIRES: has_cuda_device
# RUN: %mojo-build %s

from math import ceildiv, isqrt, isclose
from random import rand
from sys import env_get_int

from buffer import NDBuffer
from buffer.dimlist import DimList, Dim
from gpu import *
from gpu.host.event import time_function
from memory import UnsafePointer
from nn.mha import (
    _naive_attention_with_transpose,
    flash_attention,
    mha_gpu_naive,
)
from benchmark import (
    Bench,
    Bencher,
    BenchId,
    BenchMetric,
    ThroughputMeasure,
    keep,
    BenchConfig,
)
from utils.index import Index
from testing import assert_almost_equal

from gpu.host.device_context import DeviceContext
from internal_utils import bench_compile_time


fn run_mha[
    mask_rank: Int,
    qkv_type: DType,
    mask_type: DType,
    depth: Int,
    num_heads: Int,
    group: Int = 1,
](inout m: Bench, seq_len: Int, num_keys: Int, ctx: DeviceContext,) raises:
    constrained[mask_rank in (3, 4), "mha only support rank 3 or 4."]()

    # Query, key, value dimensions.
    alias batch_size = 1
    alias scale = Float32(0.125)  # isqrt[type, 1](Float32(depth))
    alias kv_num_heads = num_heads // group

    # Q, K, V shapes.
    var q_size = batch_size * num_heads * seq_len * depth
    var k_size = batch_size * kv_num_heads * num_keys * depth
    var v_size = k_size
    var o_size = q_size
    var mask_size = (num_heads if mask_rank == 4 else 1) * seq_len * num_keys

    # Allocate memory for all variables.
    var q_ptr = UnsafePointer[Scalar[qkv_type]].alloc(q_size)
    var k_ptr = UnsafePointer[Scalar[qkv_type]].alloc(k_size)
    var v_ptr = UnsafePointer[Scalar[qkv_type]].alloc(v_size)
    var mask_ptr = UnsafePointer[Scalar[mask_type]].alloc(mask_size)
    var output_ptr = UnsafePointer[Scalar[qkv_type]].alloc(o_size)
    var flash_output_ptr = UnsafePointer[Scalar[qkv_type]].alloc(o_size)

    # Q, K, V are randomly initalized.
    rand[qkv_type](q_ptr, q_size)
    rand[qkv_type](k_ptr, k_size)
    rand[qkv_type](v_ptr, v_size)
    rand[mask_type](mask_ptr, mask_size)

    # Contruct buffers.
    var q = NDBuffer[qkv_type, 4](
        q_ptr, Index(batch_size, seq_len, num_heads, depth)
    )
    var k = NDBuffer[qkv_type, 4](
        k_ptr, Index(batch_size, num_keys, kv_num_heads, depth)
    )
    var v = NDBuffer[qkv_type, 4](
        v_ptr, Index(batch_size, num_keys, kv_num_heads, depth)
    )
    var mask = NDBuffer[mask_type, 2](mask_ptr, Index(seq_len, num_keys))
    var output = NDBuffer[qkv_type, 4](
        output_ptr, Index(batch_size, seq_len, num_heads, depth)
    )

    # Device pointers
    var q_device_ptr = ctx.create_buffer[qkv_type](q_size)
    var k_device_ptr = ctx.create_buffer[qkv_type](k_size)
    var v_device_ptr = ctx.create_buffer[qkv_type](v_size)
    var mask_device_ptr = ctx.create_buffer[mask_type](mask_size)
    var output_device_ptr = ctx.create_buffer[qkv_type](o_size)

    # Copy from host to device
    ctx.enqueue_copy_to_device(q_device_ptr, q_ptr)
    ctx.enqueue_copy_to_device(k_device_ptr, k_ptr)
    ctx.enqueue_copy_to_device(v_device_ptr, v_ptr)
    ctx.enqueue_copy_to_device(mask_device_ptr, mask_ptr)

    # Contruct device buffers.
    var q_device = NDBuffer[
        qkv_type, 4, DimList(Dim(), Dim(), num_heads, depth)
    ](q_device_ptr.ptr, Index(batch_size, seq_len, num_heads, depth))
    var k_device = NDBuffer[
        qkv_type, 4, DimList(Dim(), Dim(), kv_num_heads, depth)
    ](k_device_ptr.ptr, Index(batch_size, num_keys, kv_num_heads, depth))
    var v_device = NDBuffer[
        qkv_type, 4, DimList(Dim(), Dim(), kv_num_heads, depth)
    ](v_device_ptr.ptr, Index(batch_size, num_keys, kv_num_heads, depth))
    var mask3d = NDBuffer[mask_type, 3, DimList.create_unknown[3]()](
        mask_device_ptr.ptr, Index(batch_size, seq_len, num_keys)
    )
    var mask4d = NDBuffer[mask_type, 4, DimList.create_unknown[4]()](
        mask_device_ptr.ptr, Index(batch_size, num_heads, seq_len, num_keys)
    )
    var output_device = NDBuffer[
        qkv_type, 4, DimList(Dim(), Dim(), num_heads, depth)
    ](output_device_ptr.ptr, Index(batch_size, seq_len, num_heads, depth))

    alias q_tile_num_rows = 32
    alias k_tile_num_rows = 128

    @parameter
    @always_inline
    @__copy_capture(q_device, k_device, v_device, mask3d, mask4d, output_device)
    fn kernel_launch(ctx: DeviceContext) raises:
        @parameter
        if mask_rank == 3:
            flash_attention(
                output_device, q_device, k_device, v_device, mask3d, scale, ctx
            )
        else:
            flash_attention(
                output_device, q_device, k_device, v_device, mask4d, scale, ctx
            )

    @parameter
    @always_inline
    fn bench_func(inout b: Bencher):
        @parameter
        @always_inline
        fn _kernel_launch(ctx: DeviceContext) raises:
            kernel_launch(ctx)

        b.iter_custom[_kernel_launch](ctx)

    m.bench_function[bench_func](
        BenchId(
            "mha",
            input_id="qkv_type="
            + str(qkv_type)
            + "/num_heads="
            + str(num_heads)
            + "/seq_len="
            + str(seq_len)
            + "/num_keys="
            + str(num_keys),
        ),
        ThroughputMeasure(BenchMetric.elements, 1),
    )

    ctx.synchronize()
    ctx.enqueue_copy_from_device(flash_output_ptr, output_device_ptr)

    var output_ref_device_ptr = ctx.create_buffer[qkv_type](o_size)
    ctx.enqueue_copy_to_device(output_ref_device_ptr, output_ptr)

    mha_gpu_naive[mask_rank](
        q_device_ptr.ptr,
        k_device_ptr.ptr,
        v_device_ptr.ptr,
        mask_device_ptr.ptr,
        output_ref_device_ptr.ptr,
        scale,
        batch_size,
        seq_len,
        num_keys,
        num_heads,
        depth,
        group,
        ctx,
    )

    ctx.enqueue_copy_from_device(output_ptr, output_ref_device_ptr)
    _ = output_ref_device_ptr

    var rtol = Scalar[qkv_type](0.02)

    for h in range(num_heads):
        for s in range(seq_len):
            for d in range(depth):
                var expect = output_ptr.load(d + depth * (h + s * num_heads))
                var actual = flash_output_ptr.load(
                    d + depth * (h + s * num_heads)
                )
                if not isclose(expect, actual, atol=1e-5, rtol=rtol):
                    print(h, s, d, actual, expect)
                assert_almost_equal(expect, actual, atol=1e-5, rtol=rtol)

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


@value
struct MHA_cfg:
    # params
    var mask_rank: Int
    var qkv_type: DType
    var mask_type: DType
    var depth: Int
    var num_heads: Int
    var group: Int
    # vars
    var seq_len: Int
    var num_keys: Int

    @no_inline
    fn __str__(self) -> String:
        return (
            "mask_rank="
            + str(self.mask_rank)
            + "/"
            + "qkv_type="
            + str(self.qkv_type)
            + "/"
            + "mask_type="
            + str(self.mask_type)
            + "/"
            + "depth="
            + str(self.depth)
            + "/"
            + "num_heads="
            + str(self.num_heads)
            + "/"
            + "group="
            + str(self.group)
        )


fn main() raises:
    # TODO: expand to all the params
    alias phony = env_get_int["phony", 1]()
    constrained[phony == 1]()

    alias cfg_list = List[MHA_cfg](
        # Context encoding
        MHA_cfg(
            mask_rank=4,
            qkv_type=DType.bfloat16,
            mask_type=DType.float32,
            depth=128,
            num_heads=32,
            group=1,
            seq_len=64,
            num_keys=64,
        ),
        MHA_cfg(
            mask_rank=4,
            qkv_type=DType.bfloat16,
            mask_type=DType.float32,
            depth=128,
            num_heads=32,
            group=1,
            seq_len=128,
            num_keys=128,
        ),
        MHA_cfg(
            mask_rank=4,
            qkv_type=DType.bfloat16,
            mask_type=DType.float32,
            depth=128,
            num_heads=32,
            group=1,
            seq_len=256,
            num_keys=256,
        ),
        MHA_cfg(
            mask_rank=4,
            qkv_type=DType.bfloat16,
            mask_type=DType.float32,
            depth=128,
            num_heads=32,
            group=1,
            seq_len=600,
            num_keys=600,
        ),
        MHA_cfg(
            mask_rank=4,
            qkv_type=DType.bfloat16,
            mask_type=DType.float32,
            depth=128,
            num_heads=32,
            group=1,
            seq_len=1024,
            num_keys=1024,
        ),
        MHA_cfg(
            mask_rank=4,
            qkv_type=DType.bfloat16,
            mask_type=DType.float32,
            depth=128,
            num_heads=32,
            group=1,
            seq_len=2048,
            num_keys=2048,
        ),
        # context encoding with group query
        MHA_cfg(
            mask_rank=4,
            qkv_type=DType.bfloat16,
            mask_type=DType.float32,
            depth=128,
            num_heads=24,
            group=3,
            seq_len=1024,
            num_keys=1024,
        ),
        # Token gen
        MHA_cfg(
            mask_rank=4,
            qkv_type=DType.bfloat16,
            mask_type=DType.float32,
            depth=128,
            num_heads=32,
            group=1,
            seq_len=1,
            num_keys=32,
        ),
        MHA_cfg(
            mask_rank=4,
            qkv_type=DType.bfloat16,
            mask_type=DType.float32,
            depth=128,
            num_heads=32,
            group=1,
            seq_len=1,
            num_keys=64,
        ),
        MHA_cfg(
            mask_rank=4,
            qkv_type=DType.bfloat16,
            mask_type=DType.float32,
            depth=128,
            num_heads=32,
            group=1,
            seq_len=1,
            num_keys=128,
        ),
        MHA_cfg(
            mask_rank=4,
            qkv_type=DType.bfloat16,
            mask_type=DType.float32,
            depth=128,
            num_heads=32,
            group=1,
            seq_len=1,
            num_keys=256,
        ),
        MHA_cfg(
            mask_rank=4,
            qkv_type=DType.bfloat16,
            mask_type=DType.float32,
            depth=128,
            num_heads=32,
            group=1,
            seq_len=1,
            num_keys=600,
        ),
        MHA_cfg(
            mask_rank=4,
            qkv_type=DType.bfloat16,
            mask_type=DType.float32,
            depth=128,
            num_heads=32,
            group=1,
            seq_len=1,
            num_keys=1024,
        ),
        MHA_cfg(
            mask_rank=4,
            qkv_type=DType.bfloat16,
            mask_type=DType.float32,
            depth=128,
            num_heads=32,
            group=1,
            seq_len=1,
            num_keys=2048,
        ),
        # token gen with group query
        MHA_cfg(
            mask_rank=4,
            qkv_type=DType.bfloat16,
            mask_type=DType.float32,
            depth=128,
            num_heads=24,
            group=3,
            seq_len=1,
            num_keys=1024,
        ),
        # FP32 benchmark
        MHA_cfg(
            mask_rank=3,
            qkv_type=DType.float32,
            mask_type=DType.float32,
            depth=128,
            num_heads=32,
            group=1,
            seq_len=1024,
            num_keys=1024,
        ),
        MHA_cfg(
            mask_rank=3,
            qkv_type=DType.float32,
            mask_type=DType.float32,
            depth=128,
            num_heads=32,
            group=1,
            seq_len=1,
            num_keys=1024,
        ),
    )

    var m = Bench()
    try:
        with DeviceContext() as ctx:

            @parameter
            for i in range(len(cfg_list)):
                alias x = cfg_list[i]
                run_mha[
                    x.mask_rank,
                    x.qkv_type,
                    x.mask_type,
                    x.depth,
                    x.num_heads,
                    x.group,
                ](m, x.seq_len, x.num_keys, ctx)

            @parameter
            for i in range(len(cfg_list)):
                alias x = cfg_list[i]
                bench_compile_time[
                    run_mha[
                        x.mask_rank,
                        x.qkv_type,
                        x.mask_type,
                        x.depth,
                        x.num_heads,
                        x.group,
                    ]
                ](m, "mha" + str(x))

    except e:
        print("CUDA_ERROR:", e)

    m.dump_report()
