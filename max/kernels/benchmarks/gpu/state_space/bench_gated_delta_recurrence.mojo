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
"""Benchmark the current gated delta recurrence with device events.

The adjacent YAML manifest owns decode and prefill grids. Named CLI arguments
select one batch/window/head configuration and state dtype. The kernel has
fixed head dimensions; window 4 times forward recurrence, not verify-ring.

Each warmup and timed sample starts from the same state, restored outside
the event interval. Samples use one launch, with no L2 flush. Compare source
revisions using the same harness; this does not embed a frozen baseline.
"""

import std.math
from std.benchmark import Batch, Bench, BenchmarkInfo, Report
from std.builtin.sort import sort

from internal_utils import arg_parse
from layout import TileTensor, row_major
from max.gpu.host import DeviceContext
from state_space.gated_delta import gated_delta_recurrence_fwd_gpu

comptime K = 128
comptime V = 128
comptime WARMUP: Int = 5
comptime ITERS: Int = 30


def _median(times: List[Float64]) -> Float64:
    var ordered = times.copy()
    sort(Span(ordered))
    var middle = len(ordered) // 2
    if len(ordered) % 2 == 0:
        return (ordered[middle - 1] + ordered[middle]) / 2
    return ordered[middle]


def _bench[
    state_dtype: DType
](
    ctx: DeviceContext,
    num_value_heads: Int,
    num_key_heads: Int,
    seq_len: Int,
    num_seqs: Int,
) raises:
    comptime work_dtype = DType.float32

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
    var pool_initial_d = ctx.enqueue_create_buffer[state_dtype](pool_size)
    var slot_d = ctx.enqueue_create_buffer[.uint32](num_seqs)
    var out_d = ctx.enqueue_create_buffer[work_dtype](total_T * value_dim)

    # Deterministic host fills, same trig seeding as the KDA benchmark.
    var qkv_h = ctx.enqueue_create_host_buffer[work_dtype](total_T * conv_dim)
    for i in range(total_T * conv_dim):
        qkv_h[i] = Scalar[work_dtype](std.math.sin(Float32(i + 1) * 0.313))
    ctx.enqueue_copy(qkv_d, qkv_h)

    var decay_h = ctx.enqueue_create_host_buffer[work_dtype](
        total_T * num_value_heads
    )
    var beta_h = ctx.enqueue_create_host_buffer[work_dtype](
        total_T * num_value_heads
    )
    # Decay and beta in the ranges the model's exp(-softplus) / sigmoid give.
    for i in range(total_T * num_value_heads):
        decay_h[i] = Scalar[work_dtype](
            0.75 + 0.2 * std.math.sin(Float32(i + 1) * 0.111)
        )
    ctx.enqueue_copy(decay_d, decay_h)
    for i in range(total_T * num_value_heads):
        beta_h[i] = Scalar[work_dtype](
            0.5 + 0.4 * std.math.cos(Float32(i + 1) * 0.077)
        )
    ctx.enqueue_copy(beta_d, beta_h)

    var pool_h = ctx.enqueue_create_host_buffer[state_dtype](pool_size)
    for i in range(pool_size):
        pool_h[i] = Scalar[state_dtype](
            0.1 * std.math.sin(Float32(i + 1) * 0.019)
        )
    ctx.enqueue_copy(pool_initial_d, pool_h)

    var offsets_h = ctx.enqueue_create_host_buffer[.uint32](num_seqs + 1)
    for b in range(num_seqs + 1):
        offsets_h[b] = Scalar[.uint32](b * seq_len)
    ctx.enqueue_copy(offsets_d, offsets_h)

    var slot_h = ctx.enqueue_create_host_buffer[.uint32](num_seqs)
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

    var shape = String(
        t"kernel=gated_delta_recurrence work_dtype={work_dtype} "
        t"state_dtype={state_dtype} value_heads={num_value_heads} "
        t"key_heads={num_key_heads} key_head_dim={K} value_head_dim={V} "
        t"batch={num_seqs} recurrence_window={seq_len} total_tokens={total_T}"
    )
    for _ in range(WARMUP):
        ctx.enqueue_copy(pool_d, pool_initial_d)
        launch(ctx)
    ctx.synchronize()

    var times = List[Float64]()
    for _ in range(ITERS):
        ctx.enqueue_copy(pool_d, pool_initial_d)
        ctx.synchronize()
        times.append(Float64(ctx.execution_time(launch, 1)))
    ctx.synchronize()

    for sample in range(len(times)):
        print(t"sample {shape} index={sample} ns={times[sample]}")
    var median_ns = _median(times)
    print(t"summary {shape} samples={ITERS} median_ns={median_ns}")
    var benchmark = Bench()
    var report = Report()
    for elapsed in times:
        report.runs.append(Batch(Int(elapsed), 1, True))
    benchmark.info_vec.append(BenchmarkInfo(shape, report^))
    benchmark.dump_report()

    _ = qkv_d^
    _ = decay_d^
    _ = beta_d^
    _ = offsets_d^
    _ = pool_d^
    _ = pool_initial_d^
    _ = slot_d^
    _ = out_d^


def main() raises:
    var state_dtype = arg_parse("state_dtype", String(""))
    if state_dtype != "bfloat16" and state_dtype != "float32":
        raise Error("state_dtype must be bfloat16 or float32")
    var batch = arg_parse("batch", 0)
    var window = arg_parse("window", 0)
    var value_heads = arg_parse("value_heads", 0)
    var key_heads = arg_parse("key_heads", 0)
    if (
        batch <= 0
        or window <= 0
        or key_heads <= 0
        or value_heads <= 0
        or value_heads % key_heads != 0
    ):
        raise Error(
            "Positive batch/window and divisible head counts are required"
        )
    print(
        t"config warmup={WARMUP} samples={ITERS} "
        t"launches_per_sample=1 state_reset=outside_timing l2_flush=false"
        t" csv_statistic=mean raw_statistic=median"
    )
    with DeviceContext() as ctx:
        if state_dtype == "bfloat16":
            _bench[.bfloat16](ctx, value_heads, key_heads, window, batch)
        else:
            _bench[.float32](ctx, value_heads, key_heads, window, batch)
