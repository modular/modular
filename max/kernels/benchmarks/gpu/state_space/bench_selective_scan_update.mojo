# ===----------------------------------------------------------------------=== #
# Copyright (c) 2026, Modular Inc. All rights reserved.
# ===----------------------------------------------------------------------=== #
"""Microbenchmark for ``selective_scan_update_gpu`` (Mamba decode step).

The per-token autoregressive recurrence run on every generated token, so this
is the latency-critical path for generation. Cooperative layout: one block per
``(batch, dim)`` row, ``d_state`` threads (padded to a warp) cooperating over
the state dim with a block-reduction for the output sum.

We sweep the two production decode profiles:
  * Mamba1-130m: d_state=16,  dim=1536 (d_inner), n_groups=1
  * Mamba2-130m: d_state=128, dim=1536 (head_dim=64 * n_heads=24), n_groups=1
plus a datacenter-batched point (batch=128) for the throughput regime.

Reports BOTH timings, same as the SSD benches:
  * min/mean(sync) -- sync every launch (round-trip latency), and
  * amort          -- queue all launches, sync once (matches the Triton parity
                      bench's methodology for an apples-to-apples head-to-head).

Set ``iters`` high (e.g. 20000) for the nsys sampling window -- a short run
spans <0.2 s and the 10 kHz sampler misses the kernels (see
``.planning/ssd-intra-nsight-profile.md``).
"""

from layout import TileTensor, row_major
from std.gpu.host import DeviceContext
from std.time import perf_counter_ns

from state_space.selective_scan import (
    Strides1D,
    Strides2D,
    Strides3D,
    selective_scan_update_decode_block_dim,
    selective_scan_update_decode_grid_dim_x,
    selective_scan_update_gpu,
)


def time_one[
    dtype: DType,
    batch: Int,
    dim: Int,
    n_groups: Int,
    d_state: Int,
    block_size: Int = 128,
](ctx: DeviceContext, warmups: Int, iters: Int) raises:
    var group_size = dim // n_groups
    var total_batch_dim = batch * dim
    # 2D grid (x tiles dim, y is batch), matching the op dispatch; the
    # layout-dependent tiling math is shared via the selective_scan launch
    # helpers so this stays in lockstep with the kernel.
    var launch_grid_x = selective_scan_update_decode_grid_dim_x(dim, d_state)
    var launch_block = selective_scan_update_decode_block_dim(d_state)

    # Full decode path: D skip-connection, z gating and dt_bias all present
    # (this is what a real Mamba decode step launches).
    var state_in_dev = ctx.enqueue_create_buffer[dtype](batch * dim * d_state)
    var state_out_dev = ctx.enqueue_create_buffer[dtype](batch * dim * d_state)
    var output_dev = ctx.enqueue_create_buffer[dtype](batch * dim)
    var x_dev = ctx.enqueue_create_buffer[dtype](batch * dim)
    var dt_dev = ctx.enqueue_create_buffer[dtype](batch * dim)
    var A_dev = ctx.enqueue_create_buffer[dtype](dim * d_state)
    var B_dev = ctx.enqueue_create_buffer[dtype](batch * n_groups * d_state)
    var C_dev = ctx.enqueue_create_buffer[dtype](batch * n_groups * d_state)
    var D_dev = ctx.enqueue_create_buffer[dtype](dim)
    var z_dev = ctx.enqueue_create_buffer[dtype](batch * dim)
    var dt_bias_dev = ctx.enqueue_create_buffer[dtype](dim)

    var state_in_tt = TileTensor(state_in_dev, row_major(batch, dim, d_state))
    var state_out_tt = TileTensor(state_out_dev, row_major(batch, dim, d_state))
    var output_tt = TileTensor(output_dev, row_major(batch, dim))
    var x_tt = TileTensor(x_dev, row_major(batch, dim))
    var dt_tt = TileTensor(dt_dev, row_major(batch, dim))
    var A_tt = TileTensor(A_dev, row_major(dim, d_state))
    var B_tt = TileTensor(B_dev, row_major(batch, n_groups, d_state))
    var C_tt = TileTensor(C_dev, row_major(batch, n_groups, d_state))
    var D_tt = TileTensor(D_dev, row_major(dim))
    var z_tt = TileTensor(z_dev, row_major(batch, dim))
    var dt_bias_tt = TileTensor(dt_bias_dev, row_major(dim))

    var state_out_strides = Strides3D(dim * d_state, d_state, 1)
    var output_strides = Strides2D(dim, 1)
    var state_in_strides = Strides3D(dim * d_state, d_state, 1)
    var x_strides = Strides2D(dim, 1)
    var dt_strides = Strides2D(dim, 1)
    var A_strides = Strides2D(d_state, 1)
    var B_strides = Strides3D(n_groups * d_state, d_state, 1)
    var C_strides = Strides3D(n_groups * d_state, d_state, 1)
    var D_strides = Strides1D(1)
    var z_strides = Strides2D(dim, 1)
    var dt_bias_strides = Strides1D(1)

    var compiled = ctx.compile_function[
        selective_scan_update_gpu[
            dtype,
            d_state,
            state_out_tt.LayoutType,
            output_tt.LayoutType,
            state_in_tt.LayoutType,
            x_tt.LayoutType,
            dt_tt.LayoutType,
            A_tt.LayoutType,
            B_tt.LayoutType,
            C_tt.LayoutType,
            D_tt.LayoutType,
            z_tt.LayoutType,
            dt_bias_tt.LayoutType,
        ]
    ]()

    for _ in range(warmups):
        ctx.enqueue_function(
            compiled,
            total_batch_dim,
            batch,
            dim,
            group_size,
            Int8(1),  # delta_softplus
            state_out_tt,
            output_tt,
            state_in_tt,
            x_tt,
            dt_tt,
            A_tt,
            B_tt,
            C_tt,
            D_tt,
            z_tt,
            dt_bias_tt,
            state_out_strides,
            output_strides,
            state_in_strides,
            x_strides,
            dt_strides,
            A_strides,
            B_strides,
            C_strides,
            D_strides,
            z_strides,
            dt_bias_strides,
            grid_dim=(launch_grid_x, batch),
            block_dim=(launch_block,),
        )
    ctx.synchronize()

    # Per-launch latency: sync every iter (round-trip).
    var min_ns: Int = -1
    var sum_ns: Int = 0
    for _ in range(iters):
        var t0 = perf_counter_ns()
        ctx.enqueue_function(
            compiled,
            total_batch_dim,
            batch,
            dim,
            group_size,
            Int8(1),  # delta_softplus
            state_out_tt,
            output_tt,
            state_in_tt,
            x_tt,
            dt_tt,
            A_tt,
            B_tt,
            C_tt,
            D_tt,
            z_tt,
            dt_bias_tt,
            state_out_strides,
            output_strides,
            state_in_strides,
            x_strides,
            dt_strides,
            A_strides,
            B_strides,
            C_strides,
            D_strides,
            z_strides,
            dt_bias_strides,
            grid_dim=(launch_grid_x, batch),
            block_dim=(launch_block,),
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
            total_batch_dim,
            batch,
            dim,
            group_size,
            Int8(1),  # delta_softplus
            state_out_tt,
            output_tt,
            state_in_tt,
            x_tt,
            dt_tt,
            A_tt,
            B_tt,
            C_tt,
            D_tt,
            z_tt,
            dt_bias_tt,
            state_out_strides,
            output_strides,
            state_in_strides,
            x_strides,
            dt_strides,
            A_strides,
            B_strides,
            C_strides,
            D_strides,
            z_strides,
            dt_bias_strides,
            grid_dim=(launch_grid_x, batch),
            block_dim=(launch_block,),
        )
    ctx.synchronize()
    var a1 = perf_counter_ns()
    var amort_ms = Float64(Int(a1 - a0)) / Float64(iters) / 1.0e6

    var min_ms = Float64(min_ns) / 1.0e6
    var mean_ms = Float64(sum_ns) / Float64(iters) / 1.0e6
    print(
        "  B=",
        batch,
        " dim=",
        dim,
        " N=",
        d_state,
        " blk=",
        launch_block,
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

    print("Mamba1-130m decode profile (d_state=16, dim=1536, B=1):")
    time_one[DType.float32, 1, 1536, 1, 16](ctx, 20, 200)

    print("Mamba2-130m decode profile (d_state=128, dim=1536, B=1):")
    time_one[DType.float32, 1, 1536, 1, 128](ctx, 20, 200)

    print("Datacenter-batched decode (B=128):")
    time_one[DType.float32, 128, 1536, 1, 16](ctx, 20, 200)
    time_one[DType.float32, 128, 1536, 1, 128](ctx, 20, 200)
