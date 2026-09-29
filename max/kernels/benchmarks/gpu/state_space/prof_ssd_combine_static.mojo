# ===----------------------------------------------------------------------=== #
# Copyright (c) 2026, Modular Inc. All rights reserved.
# ===----------------------------------------------------------------------=== #
"""Minimal driver: run ONLY the static tensor-core stage-4 path a few times.

Scoped target for ``ncu`` single-metric collection on GB10. The static path is
two launches (batched_matmul + decay/add epilogue), so per-kernel ncu rows show
the matmul vs epilogue split. Mamba2-130m prefill profile.
"""

from layout import TileTensor, row_major
from max.gpu.host import DeviceContext

from state_space.ssd_chunk_combine import (
    ssd_output_recombination_fwd_gpu_static,
)


def main() raises:
    var ctx = DeviceContext()

    comptime dtype = DType.float32
    comptime batch = 1
    comptime n_chunks = 4
    comptime n_heads = 24
    comptime chunk_len = 256
    comptime head_dim = 64
    comptime state_dim = 128

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

    for _ in range(4):
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
