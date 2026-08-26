# ===----------------------------------------------------------------------=== #
# Copyright (c) 2026, Modular Inc. All rights reserved.
# ===----------------------------------------------------------------------=== #
"""Minimal driver: run ONLY ``ssd_chunk_scan_fwd_gpu`` a few times.

Tightly-scoped target for ``ncu`` single-metric roofline collection on GB10
(no ``--set full`` replay — that crashes the box; see
``.planning/ssd-intra-nsight-profile.md``). Every launch is the Mamba2-130m
prefill profile so the per-kernel rows are unambiguous and the launch count is
bounded for ``-c N``.
"""

from layout import TileTensor, row_major
from max.gpu.host import DeviceContext
from std.math import ceildiv

from state_space.ssd_chunk_scan import ssd_chunk_scan_fwd_gpu


def main() raises:
    var ctx = DeviceContext()

    comptime dtype = DType.float32
    comptime batch = 1
    comptime n_chunks = 4
    comptime n_heads = 24
    comptime head_dim = 64
    comptime state_dim = 128
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

    for _ in range(4):
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
