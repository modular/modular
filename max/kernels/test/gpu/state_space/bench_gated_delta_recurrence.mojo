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
"""Benchmark the sequential gated delta recurrence used by Qwen3.5/3.8.

`gated_delta_recurrence_fwd_gpu` runs one CTA per (sequence, value head) and
walks the sequence token by token, so its wall time is linear in sequence
length and its parallelism is batch*heads however long the prompt is. Timed
here with the same methodology as `bench_kda_chunk_parallel.mojo` -- device
events, 5 warmup iterations, 30 timed, median, no L2 flush -- at the shapes
that benchmark uses, so the two can be compared directly.

Shapes also include Qwen's own linear-attention geometry (48 value heads,
16 key heads, key/value head dim 128).
"""

import std.math
from layout import TileTensor, row_major
from max.gpu.host import DeviceContext
from state_space.gated_delta import gated_delta_recurrence_fwd_gpu

comptime K: Int = 128
comptime V: Int = 128
comptime WARMUP: Int = 5
comptime ITERS: Int = 30


def _median(times: List[Float64]) -> Float64:
    var s = List[Float64]()
    for t in times:
        var inserted = False
        for j in range(len(s)):
            if t < s[j]:
                s.insert(j, t)
                inserted = True
                break
        if not inserted:
            s.append(t)
    return s[len(s) // 2]


def _bench(
    ctx: DeviceContext,
    num_value_heads: Int,
    num_key_heads: Int,
    seq_len: Int,
    num_seqs: Int,
) raises:
    comptime work_dtype = DType.float32
    comptime state_dtype = DType.float32

    var total_T = seq_len * num_seqs
    var key_dim = num_key_heads * K
    var value_dim = num_value_heads * V
    var conv_dim = key_dim * 2 + value_dim
    var max_slots = num_seqs
    var pool_size = max_slots * num_value_heads * K * V

    var qkv_d = ctx.enqueue_create_buffer[work_dtype](total_T * conv_dim)
    var decay_d = ctx.enqueue_create_buffer[work_dtype](
        total_T * num_value_heads
    )
    var beta_d = ctx.enqueue_create_buffer[work_dtype](
        total_T * num_value_heads
    )
    var offsets_d = ctx.enqueue_create_buffer[.uint32](num_seqs + 1)
    var pool_d = ctx.enqueue_create_buffer[state_dtype](pool_size)
    var slot_d = ctx.enqueue_create_buffer[.uint32](num_seqs)
    var out_d = ctx.enqueue_create_buffer[work_dtype](total_T * value_dim)

    # Deterministic host fills, same trig seeding as the KDA benchmark.
    var qkv_h = alloc[Scalar[work_dtype]](total_T * conv_dim)
    for i in range(total_T * conv_dim):
        qkv_h[i] = Scalar[work_dtype](std.math.sin(Float32(i + 1) * 0.313))
    ctx.enqueue_copy(qkv_d, qkv_h)

    var gate_h = alloc[Scalar[work_dtype]](total_T * num_value_heads)
    # Decay and beta in the ranges the model's exp(-softplus) / sigmoid give.
    for i in range(total_T * num_value_heads):
        gate_h[i] = Scalar[work_dtype](
            0.75 + 0.2 * std.math.sin(Float32(i + 1) * 0.111)
        )
    ctx.enqueue_copy(decay_d, gate_h)
    for i in range(total_T * num_value_heads):
        gate_h[i] = Scalar[work_dtype](
            0.5 + 0.4 * std.math.cos(Float32(i + 1) * 0.077)
        )
    ctx.enqueue_copy(beta_d, gate_h)

    var pool_h = alloc[Scalar[state_dtype]](pool_size)
    for i in range(pool_size):
        pool_h[i] = Scalar[state_dtype](
            0.1 * std.math.sin(Float32(i + 1) * 0.019)
        )
    ctx.enqueue_copy(pool_d, pool_h)

    var offsets_h = alloc[Scalar[.uint32]](num_seqs + 1)
    for b in range(num_seqs + 1):
        offsets_h[b] = Scalar[.uint32](b * seq_len)
    ctx.enqueue_copy(offsets_d, offsets_h)

    var slot_h = alloc[Scalar[.uint32]](num_seqs)
    for b in range(num_seqs):
        slot_h[b] = Scalar[.uint32](b)
    ctx.enqueue_copy(slot_d, slot_h)
    ctx.synchronize()

    var qkv_tt = TileTensor(qkv_d, row_major(total_T, conv_dim))
    var decay_tt = TileTensor(decay_d, row_major(total_T, num_value_heads))
    var beta_tt = TileTensor(beta_d, row_major(total_T, num_value_heads))
    var offsets_tt = TileTensor(offsets_d, row_major(num_seqs + 1))
    var slot_tt = TileTensor(slot_d, row_major(num_seqs))
    var pool_tt = TileTensor(
        pool_d, row_major(max_slots, num_value_heads, K, V)
    )
    var out_tt = TileTensor(out_d, row_major(total_T, value_dim))

    var num_blocks = num_seqs * num_value_heads

    def launch(
        lctx: DeviceContext,
    ) raises {
        imm out_tt,
        imm pool_tt,
        imm slot_tt,
        imm qkv_tt,
        imm decay_tt,
        imm beta_tt,
        imm offsets_tt,
        imm num_seqs,
        imm num_value_heads,
        imm num_key_heads,
        imm key_dim,
        imm conv_dim,
        imm value_dim,
        imm num_blocks,
    }:
        lctx.enqueue_function[
            gated_delta_recurrence_fwd_gpu[
                work_dtype,
                state_dtype,
                K,
                V,
                out_tt.LayoutType,
                qkv_tt.LayoutType,
                decay_tt.LayoutType,
                beta_tt.LayoutType,
                pool_tt.LayoutType,
                slot_tt.LayoutType,
                offsets_tt.LayoutType,
                out_tt.Engine,
            ]
        ](
            Int32(num_seqs),
            Int32(num_value_heads),
            Int32(num_key_heads),
            Int32(key_dim),
            out_tt,
            pool_tt,
            slot_tt,
            qkv_tt,
            decay_tt,
            beta_tt,
            offsets_tt,
            grid_dim=(num_blocks,),
            block_dim=(V,),
        )

    for _ in range(WARMUP):
        launch(ctx)
    ctx.synchronize()

    var times = List[Float64]()
    for _ in range(ITERS):
        times.append(Float64(ctx.execution_time(launch, 1)) / 1e6)
    ctx.synchronize()

    var ms = _median(times)
    var tok_per_s = Float64(total_T) / (ms / 1000.0)
    print(
        "HV=",
        num_value_heads,
        " H=",
        num_key_heads,
        " T=",
        seq_len,
        " x",
        num_seqs,
        "  median ",
        ms,
        " ms   ",
        tok_per_s / 1e6,
        " Mtok/s (one layer)",
        sep="",
    )

    _ = qkv_d^
    _ = decay_d^
    _ = beta_d^
    _ = offsets_d^
    _ = pool_d^
    _ = slot_d^
    _ = out_d^


def main() raises:
    with DeviceContext() as ctx:
        print("gated_delta_recurrence_fwd_gpu (sequential), median ms")
        # Shapes shared with bench_kda_chunk_parallel.mojo.
        _bench(ctx, 64, 64, 8192, 1)
        _bench(ctx, 96, 96, 8192, 1)
        _bench(ctx, 64, 64, 1024, 8)
        _bench(ctx, 96, 96, 1024, 8)
        # Qwen3.5/3.8 linear-attention geometry.
        _bench(ctx, 48, 16, 8192, 1)
        _bench(ctx, 48, 16, 1024, 8)
