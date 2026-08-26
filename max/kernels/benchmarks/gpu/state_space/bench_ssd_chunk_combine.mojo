# ===----------------------------------------------------------------------=== #
# Copyright (c) 2026, Modular Inc. All rights reserved.
# ===----------------------------------------------------------------------=== #
"""Microbenchmark for ``ssd_output_recombination_fwd_gpu`` (SSD stage 4).

Times wall-clock per launch (kernel + sync) and amortized throughput (queue all
launches, sync once -- matches the Triton parity bench methodology) across the
Mamba2-130m prefill profile (B=1, n_chunks=4, n_heads=24, L=256, P=64, N=128)
and a chunk-count sweep.

The recombination adds the inter-chunk (off-diagonal) contribution to the
intra-chunk diagonal output:

    Y_off[l, p] = exp(cumsum(A)[l]) * sum_n C[l, n] * entering_state[p, n]
    Y[l, p]     = Y_diag[l, p] + Y_off[l, p]
"""

from layout import TileTensor, row_major
from max.gpu.host import DeviceContext
from std.math import ceildiv
from std.time import perf_counter_ns

from state_space.ssd_chunk_combine import (
    ssd_output_recombination_fwd_gpu,
    ssd_output_recombination_fwd_gpu_fused,
    ssd_output_recombination_fwd_gpu_static,
)


def time_one[
    dtype: DType,
    batch: Int,
    n_chunks: Int,
    n_heads: Int,
    chunk_len: Int,
    head_dim: Int,
    state_dim: Int,
](ctx: DeviceContext, warmups: Int, iters: Int) raises:
    comptime SCALAR_BLOCK_SIZE = 128

    comptime c_count = batch * n_chunks * n_heads * chunk_len * state_dim
    comptime ent_count = batch * n_chunks * n_heads * head_dim * state_dim
    comptime a_count = batch * n_chunks * n_heads * chunk_len
    comptime y_count = batch * n_chunks * n_heads * chunk_len * head_dim

    var c_device = ctx.enqueue_create_buffer[dtype](c_count)
    var ent_device = ctx.enqueue_create_buffer[dtype](ent_count)
    var a_device = ctx.enqueue_create_buffer[dtype](a_count)
    var yd_device = ctx.enqueue_create_buffer[dtype](y_count)
    var y_device = ctx.enqueue_create_buffer[dtype](y_count)

    var c_tt = TileTensor(
        c_device, row_major(batch, n_chunks, n_heads, chunk_len, state_dim)
    )
    var ent_tt = TileTensor(
        ent_device, row_major(batch, n_chunks, n_heads, head_dim, state_dim)
    )
    var a_tt = TileTensor(
        a_device, row_major(batch, n_chunks, n_heads, chunk_len)
    )
    var yd_tt = TileTensor(
        yd_device, row_major(batch, n_chunks, n_heads, chunk_len, head_dim)
    )
    var y_tt = TileTensor(
        y_device, row_major(batch, n_chunks, n_heads, chunk_len, head_dim)
    )

    var total_threads = batch * n_chunks * n_heads * chunk_len * head_dim
    var grid = ceildiv(total_threads, SCALAR_BLOCK_SIZE)

    var compiled = ctx.compile_function[
        ssd_output_recombination_fwd_gpu[
            dtype,
            c_tt.LayoutType,
            ent_tt.LayoutType,
            a_tt.LayoutType,
            yd_tt.LayoutType,
            y_tt.LayoutType,
        ]
    ]()

    for _ in range(warmups):
        ctx.enqueue_function(
            compiled,
            Int32(batch),
            Int32(n_chunks),
            Int32(n_heads),
            Int32(chunk_len),
            Int32(head_dim),
            Int32(state_dim),
            c_tt,
            ent_tt,
            a_tt,
            yd_tt,
            y_tt,
            grid_dim=(grid,),
            block_dim=(SCALAR_BLOCK_SIZE,),
        )
    ctx.synchronize()

    # Per-launch latency: sync every iter.
    var min_ns: Int = -1
    var sum_ns: Int = 0
    for _ in range(iters):
        var t0 = perf_counter_ns()
        ctx.enqueue_function(
            compiled,
            Int32(batch),
            Int32(n_chunks),
            Int32(n_heads),
            Int32(chunk_len),
            Int32(head_dim),
            Int32(state_dim),
            c_tt,
            ent_tt,
            a_tt,
            yd_tt,
            y_tt,
            grid_dim=(grid,),
            block_dim=(SCALAR_BLOCK_SIZE,),
        )
        ctx.synchronize()
        var t1 = perf_counter_ns()
        var elapsed = Int(t1 - t0)
        sum_ns += elapsed
        if min_ns < 0 or elapsed < min_ns:
            min_ns = elapsed

    # Amortized throughput: queue all launches, sync once.
    var a0 = perf_counter_ns()
    for _ in range(iters):
        ctx.enqueue_function(
            compiled,
            Int32(batch),
            Int32(n_chunks),
            Int32(n_heads),
            Int32(chunk_len),
            Int32(head_dim),
            Int32(state_dim),
            c_tt,
            ent_tt,
            a_tt,
            yd_tt,
            y_tt,
            grid_dim=(grid,),
            block_dim=(SCALAR_BLOCK_SIZE,),
        )
    ctx.synchronize()
    var a1 = perf_counter_ns()
    var amort_ms = Float64(Int(a1 - a0)) / Float64(iters) / 1.0e6

    var min_ms = Float64(min_ns) / 1.0e6
    var mean_ms = Float64(sum_ns) / Float64(iters) / 1.0e6
    print(
        "  nc=",
        n_chunks,
        " nh=",
        n_heads,
        " L=",
        chunk_len,
        " P=",
        head_dim,
        " N=",
        state_dim,
        "  min(sync)=",
        min_ms,
        "ms  mean(sync)=",
        mean_ms,
        "ms  amort=",
        amort_ms,
        "ms",
        sep="",
    )


def time_one_static[
    dtype: DType,
    batch: Int,
    n_chunks: Int,
    n_heads: Int,
    chunk_len: Int,
    head_dim: Int,
    state_dim: Int,
](ctx: DeviceContext, warmups: Int, iters: Int) raises:
    comptime c_count = batch * n_chunks * n_heads * chunk_len * state_dim
    comptime ent_count = batch * n_chunks * n_heads * head_dim * state_dim
    comptime a_count = batch * n_chunks * n_heads * chunk_len
    comptime y_count = batch * n_chunks * n_heads * chunk_len * head_dim

    var c_device = ctx.enqueue_create_buffer[dtype](c_count)
    var ent_device = ctx.enqueue_create_buffer[dtype](ent_count)
    var a_device = ctx.enqueue_create_buffer[dtype](a_count)
    var yd_device = ctx.enqueue_create_buffer[dtype](y_count)
    var y_device = ctx.enqueue_create_buffer[dtype](y_count)

    var c_tt = TileTensor(
        c_device, row_major(batch, n_chunks, n_heads, chunk_len, state_dim)
    )
    var ent_tt = TileTensor(
        ent_device, row_major(batch, n_chunks, n_heads, head_dim, state_dim)
    )
    var a_tt = TileTensor(
        a_device, row_major(batch, n_chunks, n_heads, chunk_len)
    )
    var yd_tt = TileTensor(
        yd_device, row_major(batch, n_chunks, n_heads, chunk_len, head_dim)
    )
    var y_tt = TileTensor(
        y_device, row_major(batch, n_chunks, n_heads, chunk_len, head_dim)
    )

    for _ in range(warmups):
        ssd_output_recombination_fwd_gpu_static[
            dtype,
            chunk_len,
            state_dim,
            head_dim,
            c_tt.LayoutType,
            ent_tt.LayoutType,
            a_tt.LayoutType,
            yd_tt.LayoutType,
            y_tt.LayoutType,
        ](batch, n_chunks, n_heads, c_tt, ent_tt, a_tt, yd_tt, y_tt, ctx)
    ctx.synchronize()

    var a0 = perf_counter_ns()
    for _ in range(iters):
        ssd_output_recombination_fwd_gpu_static[
            dtype,
            chunk_len,
            state_dim,
            head_dim,
            c_tt.LayoutType,
            ent_tt.LayoutType,
            a_tt.LayoutType,
            yd_tt.LayoutType,
            y_tt.LayoutType,
        ](batch, n_chunks, n_heads, c_tt, ent_tt, a_tt, yd_tt, y_tt, ctx)
    ctx.synchronize()
    var a1 = perf_counter_ns()
    var amort_ms = Float64(Int(a1 - a0)) / Float64(iters) / 1.0e6
    print(
        "  [static] nc=",
        n_chunks,
        " nh=",
        n_heads,
        " L=",
        chunk_len,
        " P=",
        head_dim,
        " N=",
        state_dim,
        "  amort=",
        amort_ms,
        "ms",
        sep="",
    )


def time_one_fused[
    dtype: DType,
    batch: Int,
    n_chunks: Int,
    n_heads: Int,
    chunk_len: Int,
    head_dim: Int,
    state_dim: Int,
](ctx: DeviceContext, warmups: Int, iters: Int) raises:
    comptime c_count = batch * n_chunks * n_heads * chunk_len * state_dim
    comptime ent_count = batch * n_chunks * n_heads * head_dim * state_dim
    comptime a_count = batch * n_chunks * n_heads * chunk_len
    comptime y_count = batch * n_chunks * n_heads * chunk_len * head_dim

    var c_device = ctx.enqueue_create_buffer[dtype](c_count)
    var ent_device = ctx.enqueue_create_buffer[dtype](ent_count)
    var a_device = ctx.enqueue_create_buffer[dtype](a_count)
    var yd_device = ctx.enqueue_create_buffer[dtype](y_count)
    var y_device = ctx.enqueue_create_buffer[dtype](y_count)

    var c_tt = TileTensor(
        c_device, row_major(batch, n_chunks, n_heads, chunk_len, state_dim)
    )
    var ent_tt = TileTensor(
        ent_device, row_major(batch, n_chunks, n_heads, head_dim, state_dim)
    )
    var a_tt = TileTensor(
        a_device, row_major(batch, n_chunks, n_heads, chunk_len)
    )
    var yd_tt = TileTensor(
        yd_device, row_major(batch, n_chunks, n_heads, chunk_len, head_dim)
    )
    var y_tt = TileTensor(
        y_device, row_major(batch, n_chunks, n_heads, chunk_len, head_dim)
    )

    for _ in range(warmups):
        ssd_output_recombination_fwd_gpu_fused[
            dtype,
            chunk_len,
            state_dim,
            head_dim,
            c_tt.LayoutType,
            ent_tt.LayoutType,
            a_tt.LayoutType,
            yd_tt.LayoutType,
            y_tt.LayoutType,
        ](batch, n_chunks, n_heads, c_tt, ent_tt, a_tt, yd_tt, y_tt, ctx)
    ctx.synchronize()

    var a0 = perf_counter_ns()
    for _ in range(iters):
        ssd_output_recombination_fwd_gpu_fused[
            dtype,
            chunk_len,
            state_dim,
            head_dim,
            c_tt.LayoutType,
            ent_tt.LayoutType,
            a_tt.LayoutType,
            yd_tt.LayoutType,
            y_tt.LayoutType,
        ](batch, n_chunks, n_heads, c_tt, ent_tt, a_tt, yd_tt, y_tt, ctx)
    ctx.synchronize()
    var a1 = perf_counter_ns()
    var amort_ms = Float64(Int(a1 - a0)) / Float64(iters) / 1.0e6
    print(
        "  [fused] nc=",
        n_chunks,
        " nh=",
        n_heads,
        " L=",
        chunk_len,
        " P=",
        head_dim,
        " N=",
        state_dim,
        "  amort=",
        amort_ms,
        "ms",
        sep="",
    )


def main() raises:
    var ctx = DeviceContext()

    print("Scalar baseline (Mamba2-130m prefill profile + nchunks scaling):")
    time_one[DType.float32, 1, 4, 24, 256, 64, 128](ctx, 20, 100)
    time_one[DType.float32, 1, 8, 24, 256, 64, 128](ctx, 20, 100)
    time_one[DType.float32, 1, 16, 24, 256, 64, 128](ctx, 20, 100)

    print("Static tensor-core path (same profiles):")
    time_one_static[DType.float32, 1, 4, 24, 256, 64, 128](ctx, 20, 100)
    time_one_static[DType.float32, 1, 8, 24, 256, 64, 128](ctx, 20, 100)
    time_one_static[DType.float32, 1, 16, 24, 256, 64, 128](ctx, 20, 100)

    print("Fused single-pass MMA path (same profiles):")
    time_one_fused[DType.float32, 1, 4, 24, 256, 64, 128](ctx, 20, 100)
    time_one_fused[DType.float32, 1, 8, 24, 256, 64, 128](ctx, 20, 100)
    time_one_fused[DType.float32, 1, 16, 24, 256, 64, 128](ctx, 20, 100)
