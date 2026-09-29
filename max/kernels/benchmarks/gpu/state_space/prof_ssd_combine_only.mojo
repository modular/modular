# ===----------------------------------------------------------------------=== #
# Copyright (c) 2026, Modular Inc. All rights reserved.
# ===----------------------------------------------------------------------=== #
"""Minimal driver: run ONLY ``ssd_output_recombination_fwd_gpu`` a few times.

Tightly-scoped target for ``ncu`` single-metric roofline collection on GB10
(no ``--set full`` replay -- that crashes the box; see
``.planning/ssd-intra-nsight-profile.md``). Every launch is the Mamba2-130m
prefill profile so the per-kernel rows are unambiguous and the launch count is
bounded for ``-c N``.
"""

from layout import TileTensor, row_major
from max.gpu.host import DeviceContext
from std.math import ceildiv

from state_space.ssd_chunk_combine import ssd_output_recombination_fwd_gpu


def main() raises:
    var ctx = DeviceContext()

    comptime dtype = DType.float32
    comptime batch = 1
    comptime n_chunks = 4
    comptime n_heads = 24
    comptime chunk_len = 256
    comptime head_dim = 64
    comptime state_dim = 128
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

    for _ in range(4):
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
