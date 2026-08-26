# ===----------------------------------------------------------------------=== #
# Copyright (c) 2026, Modular Inc. All rights reserved.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions:
# https://llvm.org/LICENSE.txt
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ===----------------------------------------------------------------------=== #
"""Microbenchmark sweep for ``ssd_chunk_state_fwd_gpu``.

Stage 2 of the chunked SSD prefill: reduce each chunk to its
``[head_dim, state_dim]`` end-state. Sweeps a few dim configs to isolate where
time goes and reports the Mamba2-130m prefill profile used by the intra-chunk
optimization notes (B=1, n_chunks=4, n_heads=24, L=256, P=64, N=128). Times
wall-clock per call (kernel enqueue + sync).
"""

from max.gpu.host import DeviceContext
from std.time import perf_counter_ns
from layout import TileTensor, row_major

from state_space.ssd_chunk_state import (
    ssd_chunk_state_fwd_gpu,
    ssd_chunk_state_fwd_gpu_static,
    ssd_chunk_state_fwd_gpu_fused,
)


def time_one[
    dtype: DType,
    batch: Int,
    n_chunks: Int,
    n_heads: Int,
    chunk_len: Int,
    head_dim: Int,
    state_dim: Int,
](ctx: DeviceContext, warmups: Int, iters: Int, num_p_tiles: Int = 1) raises:
    comptime b_count = batch * n_chunks * n_heads * chunk_len * state_dim
    comptime x_count = batch * n_chunks * n_heads * chunk_len * head_dim
    comptime a_count = batch * n_chunks * n_heads * chunk_len
    comptime state_count = batch * n_chunks * n_heads * head_dim * state_dim

    var B_device = ctx.enqueue_create_buffer[dtype](b_count)
    var X_device = ctx.enqueue_create_buffer[dtype](x_count)
    var A_device = ctx.enqueue_create_buffer[dtype](a_count)
    var state_device = ctx.enqueue_create_buffer[dtype](state_count)

    var B_tt = TileTensor(
        B_device, row_major(batch, n_chunks, n_heads, chunk_len, state_dim)
    )
    var X_tt = TileTensor(
        X_device, row_major(batch, n_chunks, n_heads, chunk_len, head_dim)
    )
    var A_tt = TileTensor(
        A_device, row_major(batch, n_chunks, n_heads, chunk_len)
    )
    var state_tt = TileTensor(
        state_device, row_major(batch, n_chunks, n_heads, head_dim, state_dim)
    )

    var total_slices = batch * n_chunks * n_heads

    var compiled_func = ctx.compile_function[
        ssd_chunk_state_fwd_gpu[
            dtype,
            B_tt.LayoutType,
            X_tt.LayoutType,
            A_tt.LayoutType,
            state_tt.LayoutType,
        ]
    ]()

    for _ in range(warmups):
        ctx.enqueue_function(
            compiled_func,
            Int32(batch),
            Int32(n_chunks),
            Int32(n_heads),
            Int32(chunk_len),
            Int32(state_dim),
            Int32(head_dim),
            B_tt,
            X_tt,
            A_tt,
            state_tt,
            grid_dim=(total_slices, num_p_tiles),
            block_dim=(chunk_len,),
        )
    ctx.synchronize()

    var min_ns: Int = -1
    var sum_ns: Int = 0
    for _ in range(iters):
        var t0 = perf_counter_ns()
        ctx.enqueue_function(
            compiled_func,
            Int32(batch),
            Int32(n_chunks),
            Int32(n_heads),
            Int32(chunk_len),
            Int32(state_dim),
            Int32(head_dim),
            B_tt,
            X_tt,
            A_tt,
            state_tt,
            grid_dim=(total_slices, num_p_tiles),
            block_dim=(chunk_len,),
        )
        ctx.synchronize()
        var t1 = perf_counter_ns()
        var elapsed = Int(t1 - t0)
        sum_ns += elapsed
        if min_ns < 0 or elapsed < min_ns:
            min_ns = elapsed

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
        " p_tiles=",
        num_p_tiles,
        "  min=",
        min_ms,
        "ms  mean=",
        mean_ms,
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
    """Time the static tensor-core path (decay-transpose + batched_matmul)."""
    comptime b_count = batch * n_chunks * n_heads * chunk_len * state_dim
    comptime x_count = batch * n_chunks * n_heads * chunk_len * head_dim
    comptime a_count = batch * n_chunks * n_heads * chunk_len
    comptime state_count = batch * n_chunks * n_heads * head_dim * state_dim

    var B_device = ctx.enqueue_create_buffer[dtype](b_count)
    var X_device = ctx.enqueue_create_buffer[dtype](x_count)
    var A_device = ctx.enqueue_create_buffer[dtype](a_count)
    var state_device = ctx.enqueue_create_buffer[dtype](state_count)

    var B_tt = TileTensor(
        B_device, row_major(batch, n_chunks, n_heads, chunk_len, state_dim)
    )
    var X_tt = TileTensor(
        X_device, row_major(batch, n_chunks, n_heads, chunk_len, head_dim)
    )
    var A_tt = TileTensor(
        A_device, row_major(batch, n_chunks, n_heads, chunk_len)
    )
    var state_tt = TileTensor(
        state_device, row_major(batch, n_chunks, n_heads, head_dim, state_dim)
    )

    for _ in range(warmups):
        ssd_chunk_state_fwd_gpu_static[
            dtype,
            chunk_len,
            state_dim,
            head_dim,
            B_tt.LayoutType,
            X_tt.LayoutType,
            A_tt.LayoutType,
            state_tt.LayoutType,
        ](batch, n_chunks, n_heads, B_tt, X_tt, A_tt, state_tt, ctx)
    ctx.synchronize()

    # Pipelined timing (enqueue all iters, sync once) to match the Triton
    # bench's methodology for a fair throughput comparison.
    var t0 = perf_counter_ns()
    for _ in range(iters):
        ssd_chunk_state_fwd_gpu_static[
            dtype,
            chunk_len,
            state_dim,
            head_dim,
            B_tt.LayoutType,
            X_tt.LayoutType,
            A_tt.LayoutType,
            state_tt.LayoutType,
        ](batch, n_chunks, n_heads, B_tt, X_tt, A_tt, state_tt, ctx)
    ctx.synchronize()
    var t1 = perf_counter_ns()

    print(
        "  [tensor-core] nc=",
        n_chunks,
        " nh=",
        n_heads,
        " L=",
        chunk_len,
        " P=",
        head_dim,
        " N=",
        state_dim,
        "  ",
        Float64(Int(t1 - t0)) / Float64(iters) / 1.0e6,
        "ms/iter (pipelined)",
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
    """Time the fused single-pass tensor-core path (pipelined throughput)."""
    comptime b_count = batch * n_chunks * n_heads * chunk_len * state_dim
    comptime x_count = batch * n_chunks * n_heads * chunk_len * head_dim
    comptime a_count = batch * n_chunks * n_heads * chunk_len
    comptime state_count = batch * n_chunks * n_heads * head_dim * state_dim

    var B_device = ctx.enqueue_create_buffer[dtype](b_count)
    var X_device = ctx.enqueue_create_buffer[dtype](x_count)
    var A_device = ctx.enqueue_create_buffer[dtype](a_count)
    var state_device = ctx.enqueue_create_buffer[dtype](state_count)

    var B_tt = TileTensor(
        B_device, row_major(batch, n_chunks, n_heads, chunk_len, state_dim)
    )
    var X_tt = TileTensor(
        X_device, row_major(batch, n_chunks, n_heads, chunk_len, head_dim)
    )
    var A_tt = TileTensor(
        A_device, row_major(batch, n_chunks, n_heads, chunk_len)
    )
    var state_tt = TileTensor(
        state_device, row_major(batch, n_chunks, n_heads, head_dim, state_dim)
    )

    for _ in range(warmups):
        ssd_chunk_state_fwd_gpu_fused[
            dtype,
            chunk_len,
            state_dim,
            head_dim,
            B_tt.LayoutType,
            X_tt.LayoutType,
            A_tt.LayoutType,
            state_tt.LayoutType,
        ](batch, n_chunks, n_heads, B_tt, X_tt, A_tt, state_tt, ctx)
    ctx.synchronize()

    var t0 = perf_counter_ns()
    for _ in range(iters):
        ssd_chunk_state_fwd_gpu_fused[
            dtype,
            chunk_len,
            state_dim,
            head_dim,
            B_tt.LayoutType,
            X_tt.LayoutType,
            A_tt.LayoutType,
            state_tt.LayoutType,
        ](batch, n_chunks, n_heads, B_tt, X_tt, A_tt, state_tt, ctx)
    ctx.synchronize()
    var t1 = perf_counter_ns()

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
        "  ",
        Float64(Int(t1 - t0)) / Float64(iters) / 1.0e6,
        "ms/iter (pipelined)",
        sep="",
    )


def main() raises:
    var ctx = DeviceContext()

    # Single-slice baselines to expose where time goes.
    print("Single-slice (batch=1, n_chunks=1, n_heads=1) sweep:")
    time_one[DType.float32, 1, 1, 1, 64, 64, 32](ctx, 5, 20)
    time_one[DType.float32, 1, 1, 1, 64, 64, 128](ctx, 5, 20)
    time_one[DType.float32, 1, 1, 1, 128, 64, 128](ctx, 5, 20)
    time_one[DType.float32, 1, 1, 1, 256, 64, 128](ctx, 5, 20)

    # Mamba2-130m prefill profile, swept over p-tile counts (block count =
    # 96 * p_tiles) to find the GB10 occupancy sweet spot.
    print("Multi-slice sweep (Mamba2-130m prefill profile), p-tile sweep:")
    time_one[DType.float32, 1, 4, 24, 256, 64, 128](ctx, 3, 20, 1)
    time_one[DType.float32, 1, 4, 24, 256, 64, 128](ctx, 3, 20, 2)
    time_one[DType.float32, 1, 4, 24, 256, 64, 128](ctx, 3, 20, 4)
    time_one[DType.float32, 1, 4, 24, 256, 64, 128](ctx, 3, 20, 8)
    time_one[DType.float32, 1, 4, 24, 256, 64, 128](ctx, 3, 20, 16)

    # Tensor-core path (decay-transpose + batched_matmul) at the same profile,
    # plus longer prefills, to compare head-to-head with Triton _chunk_state_fwd.
    print("Tensor-core path (static), pipelined throughput:")
    time_one_static[DType.float32, 1, 4, 24, 256, 64, 128](ctx, 10, 200)
    time_one_static[DType.float32, 1, 8, 24, 256, 64, 128](ctx, 10, 200)
    time_one_static[DType.float32, 1, 16, 24, 256, 64, 128](ctx, 10, 200)

    print("Fused single-pass tensor-core path, pipelined throughput:")
    time_one_fused[DType.float32, 1, 4, 24, 256, 64, 128](ctx, 10, 200)
    time_one_fused[DType.float32, 1, 8, 24, 256, 64, 128](ctx, 10, 200)
    time_one_fused[DType.float32, 1, 16, 24, 256, 64, 128](ctx, 10, 200)
