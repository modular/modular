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
"""Benchmarks the production verify-ring recurrence.

The live state must remain bit-exact, every output and ring record must be
written, and unused ring rows, token positions and padding must stay seeded.

The adjacent YAML manifest owns the grids. Named CLI arguments select one
batch/window/head configuration and state dtype, with fixed head dimensions
and ring capacity.
Each sample restores the same state and ring outside device-event timing.
"""

import std.math
from std.benchmark import Batch, Bench, BenchmarkInfo, Report
from std.builtin.sort import sort
from std.memory import bitcast
from std.testing import assert_equal, assert_true

from internal_utils import arg_parse
from layout import TileTensor, row_major
from max.gpu.host import DeviceContext
from state_space.gated_delta import (
    gated_delta_recurrence_verify_ring_gpu,
    gated_delta_ring_record_elements,
)

comptime K = 128
comptime V = 128
comptime RING_LEN = 8
comptime RING_DTYPE = DType.float32
comptime WARMUP = 5
comptime SAMPLES = 30


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
    batch: Int,
    window: Int,
    value_heads: Int,
    key_heads: Int,
) raises:
    comptime work_dtype = DType.float32
    var HV = value_heads
    var HK = key_heads
    var key_dim = HK * K
    var value_dim = HV * V
    var conv_dim = 2 * key_dim + value_dim
    var total_tokens = batch * window
    var state_rows = batch + 2
    var ring_rows = batch + 3
    var record_elements = gated_delta_ring_record_elements[K, V](HV // HK)
    var record_stride = record_elements + 3
    var pool_elements = state_rows * HV * K * V
    var ring_row_elements = HK * RING_LEN * record_stride
    var ring_elements = ring_rows * ring_row_elements

    var qkv_h = ctx.enqueue_create_host_buffer[work_dtype](
        total_tokens * conv_dim
    )
    var decay_h = ctx.enqueue_create_host_buffer[work_dtype](total_tokens * HV)
    var beta_h = ctx.enqueue_create_host_buffer[work_dtype](total_tokens * HV)
    var pool_initial_h = ctx.enqueue_create_host_buffer[state_dtype](
        pool_elements
    )
    var ring_initial_h = ctx.enqueue_create_host_buffer[RING_DTYPE](
        ring_elements
    )
    var offsets_h = ctx.enqueue_create_host_buffer[.uint32](batch + 1)
    var slots_h = ctx.enqueue_create_host_buffer[.uint32](batch)
    var ring_slots_h = ctx.enqueue_create_host_buffer[.uint32](batch)
    for i in range(total_tokens * conv_dim):
        qkv_h[i] = std.math.sin(Float32(i + 1) * 0.313)
    for i in range(total_tokens * HV):
        decay_h[i] = 0.75 + 0.2 * std.math.sin(Float32(i + 1) * 0.111)
        beta_h[i] = 0.5 + 0.4 * std.math.cos(Float32(i + 1) * 0.077)
    for i in range(pool_elements):
        pool_initial_h[i] = Scalar[state_dtype](
            0.1 * std.math.sin(Float32(i + 1) * 0.019)
        )
    for i in range(ring_elements):
        ring_initial_h[i] = -12345.5 - Float32(i % 17) * 0.125
    for b in range(batch):
        offsets_h[b] = UInt32(b * window)
        slots_h[b] = UInt32(batch - b)
        ring_slots_h[b] = UInt32(b + 2)
    offsets_h[batch] = UInt32(total_tokens)

    var qkv_d = ctx.enqueue_create_buffer[work_dtype](total_tokens * conv_dim)
    var decay_d = ctx.enqueue_create_buffer[work_dtype](total_tokens * HV)
    var beta_d = ctx.enqueue_create_buffer[work_dtype](total_tokens * HV)
    var offsets_d = ctx.enqueue_create_buffer[.uint32](batch + 1)
    var slots_d = ctx.enqueue_create_buffer[.uint32](batch)
    var ring_slots_d = ctx.enqueue_create_buffer[.uint32](batch)
    var pool_d = ctx.enqueue_create_buffer[state_dtype](pool_elements)
    var pool_initial_d = ctx.enqueue_create_buffer[state_dtype](pool_elements)
    var ring_d = ctx.enqueue_create_buffer[RING_DTYPE](ring_elements)
    var ring_initial_d = ctx.enqueue_create_buffer[RING_DTYPE](ring_elements)
    var output_d = ctx.enqueue_create_buffer[work_dtype](
        total_tokens * value_dim
    )
    ctx.enqueue_copy(qkv_d, qkv_h)
    ctx.enqueue_copy(decay_d, decay_h)
    ctx.enqueue_copy(beta_d, beta_h)
    ctx.enqueue_copy(offsets_d, offsets_h)
    ctx.enqueue_copy(slots_d, slots_h)
    ctx.enqueue_copy(ring_slots_d, ring_slots_h)
    ctx.enqueue_copy(pool_initial_d, pool_initial_h)
    ctx.enqueue_copy(ring_initial_d, ring_initial_h)
    ctx.synchronize()

    var qkv = TileTensor(qkv_d, row_major(total_tokens, conv_dim))
    var decay = TileTensor(decay_d, row_major(total_tokens, HV))
    var beta = TileTensor(beta_d, row_major(total_tokens, HV))
    var offsets = TileTensor(offsets_d, row_major(batch + 1))
    var slots = TileTensor(slots_d, row_major(batch))
    var ring_slots = TileTensor(ring_slots_d, row_major(batch))
    var pool = TileTensor(pool_d, row_major(state_rows, HV, K, V))
    var ring = TileTensor(
        ring_d, row_major(ring_rows, HK, RING_LEN, record_stride)
    )
    var output = TileTensor(output_d, row_major(total_tokens, value_dim))

    def launch(
        lctx: DeviceContext,
    ) raises {
        imm batch,
        imm HV,
        imm HK,
        imm key_dim,
        imm output,
        imm pool,
        imm slots,
        imm qkv,
        imm decay,
        imm beta,
        imm offsets,
        imm ring,
        imm ring_slots,
    }:
        lctx.enqueue_function[
            gated_delta_recurrence_verify_ring_gpu[
                work_dtype,
                state_dtype,
                RING_DTYPE,
                K,
                V,
                RING_LEN,
                output.LayoutType,
                qkv.LayoutType,
                decay.LayoutType,
                beta.LayoutType,
                pool.LayoutType,
                slots.LayoutType,
                offsets.LayoutType,
                ring.LayoutType,
                ring_slots.LayoutType,
                output.Engine,
            ]
        ](
            Int32(batch),
            Int32(HV),
            Int32(HK),
            Int32(key_dim),
            output,
            pool,
            slots,
            qkv,
            decay,
            beta,
            offsets,
            ring,
            ring_slots,
            grid_dim=(batch * HV,),
            block_dim=(V,),
        )

    def restore(
        lctx: DeviceContext,
    ) raises {imm pool_d, imm pool_initial_d, imm ring_d, imm ring_initial_d,}:
        lctx.enqueue_copy(pool_d, pool_initial_d)
        lctx.enqueue_copy(ring_d, ring_initial_d)

    var output_current_h = ctx.enqueue_create_host_buffer[work_dtype](
        total_tokens * value_dim
    )
    var pool_current_h = ctx.enqueue_create_host_buffer[state_dtype](
        pool_elements
    )
    var ring_current_h = ctx.enqueue_create_host_buffer[RING_DTYPE](
        ring_elements
    )
    restore(ctx)
    output_d.enqueue_fill(std.math.nan[work_dtype]())
    launch(ctx)
    ctx.enqueue_copy(output_current_h, output_d)
    ctx.enqueue_copy(pool_current_h, pool_d)
    ctx.enqueue_copy(ring_current_h, ring_d)
    ctx.synchronize()

    for i in range(total_tokens * value_dim):
        assert_true(not std.math.isnan(output_current_h[i]))
    comptime state_bits: DType = (
        .uint16 if state_dtype == .bfloat16 else .uint32
    )
    for i in range(pool_elements):
        assert_equal(
            bitcast[state_bits](pool_current_h[i]),
            bitcast[state_bits](pool_initial_h[i]),
        )
    for i in range(ring_elements):
        var row = i // ring_row_elements
        var token = (i // record_stride) % RING_LEN
        var element = i % record_stride
        var written = (
            2 <= row < batch + 2
            and token < window
            and element < record_elements
        )
        if written:
            assert_true(not std.math.isnan(ring_current_h[i]))
            assert_true(
                bitcast[.uint32](ring_current_h[i])
                != bitcast[.uint32](ring_initial_h[i])
            )
        else:
            assert_equal(
                bitcast[.uint32](ring_current_h[i]),
                bitcast[.uint32](ring_initial_h[i]),
            )

    var description = String(
        t"kernel=gated_delta_verify work_dtype={work_dtype} "
        t"state_dtype={state_dtype} "
        t"ring_dtype={RING_DTYPE} batch={batch} window={window} value_heads={HV} "
        t"key_heads={HK} key_head_dim={K} value_head_dim={V} "
        t"ring_len={RING_LEN} record_stride={record_stride}"
    )
    print(
        t"validation {description} output=written ring=written "
        t"pool=unchanged ring_padding=unchanged"
    )
    for _ in range(WARMUP):
        restore(ctx)
        launch(ctx)
    ctx.synchronize()

    var times = List[Float64]()
    for _ in range(SAMPLES):
        restore(ctx)
        ctx.synchronize()
        times.append(Float64(ctx.execution_time(launch, 1)))
    for sample in range(SAMPLES):
        print(t"sample {description} index={sample} ns={times[sample]}")
    print(t"summary {description} samples={SAMPLES} median_ns={_median(times)}")
    var benchmark = Bench()
    var report = Report()
    for elapsed in times:
        report.runs.append(Batch(Int(elapsed), 1, True))
    benchmark.info_vec.append(BenchmarkInfo(description, report^))
    benchmark.dump_report()


def main() raises:
    var state_dtype = arg_parse("state_dtype", String(""))
    if state_dtype != "bfloat16" and state_dtype != "float32":
        raise Error("state_dtype must be bfloat16 or float32")
    var batch = arg_parse("batch", 0)
    var window = arg_parse("window", 0)
    var value_heads = arg_parse("value_heads", 0)
    var key_heads = arg_parse("key_heads", 0)
    if value_heads <= 0 or key_heads <= 0 or value_heads % key_heads != 0:
        raise Error("Positive divisible head counts are required")
    if batch <= 0 or window <= 0 or window > RING_LEN:
        raise Error("Batch must be positive and window must be in 1..ring_len")
    print(
        t"config warmup={WARMUP} samples={SAMPLES} launches_per_sample=1"
        t" state_and_ring_reset=outside_timing l2_flush=false"
        t" csv_statistic=mean raw_statistic=median"
    )
    with DeviceContext() as ctx:
        if state_dtype == "bfloat16":
            _bench[.bfloat16](ctx, batch, window, value_heads, key_heads)
        else:
            _bench[.float32](ctx, batch, window, value_heads, key_heads)
