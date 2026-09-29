# ===----------------------------------------------------------------------=== #
# Copyright (c) 2026, Modular Inc. All rights reserved.
# ===----------------------------------------------------------------------=== #
"""Microbenchmark for ``ssd_chunk_scan_fwd_gpu`` (SSD inter-chunk scan, stage 3).

Times wall-clock per launch (kernel + sync) across a small sweep, ending on
the Mamba2-130m prefill profile (B=1, n_chunks=4, n_heads=24, P=64, N=128).
The inter-chunk recurrence is sequential over chunks; we parallelize over
``(batch, head, p, n)`` exactly as the shipped kernel does.

Set ``iters`` high (e.g. 20000) for the nsys sampling window — a short run
spans <0.2 s and the 10 kHz sampler misses the kernels (see
``.planning/ssd-intra-nsight-profile.md``).
"""

from layout import TileTensor, row_major
from max.gpu.host import DeviceContext
from std.math import ceildiv
from std.time import perf_counter_ns

from state_space.ssd_chunk_scan import ssd_chunk_scan_fwd_gpu


def time_one[
    dtype: DType,
    batch: Int,
    n_chunks: Int,
    n_heads: Int,
    head_dim: Int,
    state_dim: Int,
](ctx: DeviceContext, warmups: Int, iters: Int) raises:
    comptime SCALAR_BLOCK_SIZE = 128

    comptime cs_count = batch * n_chunks * n_heads * head_dim * state_dim
    comptime cd_count = batch * n_chunks * n_heads
    comptime ent_count = batch * n_chunks * n_heads * head_dim * state_dim
    comptime final_count = batch * n_heads * head_dim * state_dim

    var cs_device = ctx.enqueue_create_buffer[dtype](cs_count)
    var cd_device = ctx.enqueue_create_buffer[dtype](cd_count)
    var ent_device = ctx.enqueue_create_buffer[dtype](ent_count)
    var final_device = ctx.enqueue_create_buffer[dtype](final_count)

    var cs_tt = TileTensor(
        cs_device, row_major(batch, n_chunks, n_heads, head_dim, state_dim)
    )
    var cd_tt = TileTensor(cd_device, row_major(batch, n_chunks, n_heads))
    var ent_tt = TileTensor(
        ent_device, row_major(batch, n_chunks, n_heads, head_dim, state_dim)
    )
    var final_tt = TileTensor(
        final_device, row_major(batch, n_heads, head_dim, state_dim)
    )

    var total_threads = batch * n_heads * head_dim * state_dim
    var grid = ceildiv(total_threads, SCALAR_BLOCK_SIZE)

    var compiled = ctx.compile_function[
        ssd_chunk_scan_fwd_gpu[
            dtype,
            cs_tt.LayoutType,
            cd_tt.LayoutType,
            ent_tt.LayoutType,
            final_tt.LayoutType,
        ]
    ]()

    for _ in range(warmups):
        ctx.enqueue_function(
            compiled,
            Int32(batch),
            Int32(n_chunks),
            Int32(n_heads),
            Int32(head_dim),
            Int32(state_dim),
            cs_tt,
            cd_tt,
            ent_tt,
            final_tt,
            grid_dim=(grid,),
            block_dim=(SCALAR_BLOCK_SIZE,),
        )
    ctx.synchronize()

    # Per-launch latency: sync every iter (round-trip).
    var min_ns: Int = -1
    var sum_ns: Int = 0
    for _ in range(iters):
        var t0 = perf_counter_ns()
        ctx.enqueue_function(
            compiled,
            Int32(batch),
            Int32(n_chunks),
            Int32(n_heads),
            Int32(head_dim),
            Int32(state_dim),
            cs_tt,
            cd_tt,
            ent_tt,
            final_tt,
            grid_dim=(grid,),
            block_dim=(SCALAR_BLOCK_SIZE,),
        )
        ctx.synchronize()
        var t1 = perf_counter_ns()
        var elapsed = Int(t1 - t0)
        sum_ns += elapsed
        if min_ns < 0 or elapsed < min_ns:
            min_ns = elapsed

    # Amortized throughput: queue all launches, sync once (matches the Triton
    # parity bench's methodology for an apples-to-apples head-to-head).
    var a0 = perf_counter_ns()
    for _ in range(iters):
        ctx.enqueue_function(
            compiled,
            Int32(batch),
            Int32(n_chunks),
            Int32(n_heads),
            Int32(head_dim),
            Int32(state_dim),
            cs_tt,
            cd_tt,
            ent_tt,
            final_tt,
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


def main() raises:
    var ctx = DeviceContext()

    print("Single-slice (batch=1, n_chunks=4, n_heads=1) sweep:")
    time_one[DType.float32, 1, 4, 1, 64, 128](ctx, 5, 50)
    time_one[DType.float32, 1, 16, 1, 64, 128](ctx, 5, 50)

    print("Multi-slice sweep (Mamba2-130m prefill profile + nchunks scaling):")
    time_one[DType.float32, 1, 4, 24, 64, 128](ctx, 20, 200)
    time_one[DType.float32, 1, 8, 24, 64, 128](ctx, 20, 200)
    time_one[DType.float32, 1, 16, 24, 64, 128](ctx, 20, 200)
