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
"""Compares copy-plus-update against the separate-state update operation."""

from std.benchmark import Batch, Bench, BenchmarkInfo, Report
from std.builtin.sort import sort
from std.memory import bitcast
from std.testing import assert_almost_equal, assert_equal, assert_true

from internal_utils import arg_parse
from layout import TileTensor, row_major
from max.gpu.host import DeviceContext
from state_space.causal_conv1d import (
    causal_conv1d_update_cpu,
    enqueue_causal_conv1d_update_gpu,
)

comptime WARMUP = 5
comptime SAMPLES = 30
comptime ROUNDS = 3


def _median(times: List[Float64]) -> Float64:
    var ordered = times.copy()
    sort(Span(ordered))
    var middle = len(ordered) // 2
    if len(ordered) % 2 == 0:
        return (ordered[middle - 1] + ordered[middle]) / 2
    return ordered[middle]


def _bench[
    dtype: DType
](
    ctx: DeviceContext,
    batch: Int,
    channels: Int,
    seqlen: Int,
    width: Int,
    state_len: Int,
) raises:
    var elements = batch * channels * seqlen
    var state_elements = batch * channels * state_len
    # Keep a valid allocation for the empty-sequence control; its view is empty.
    var input_h = ctx.enqueue_create_host_buffer[dtype](max(1, elements))
    var output_h = ctx.enqueue_create_host_buffer[dtype](max(1, elements))
    var candidate_h = ctx.enqueue_create_host_buffer[dtype](max(1, elements))
    var expected_h = ctx.enqueue_create_host_buffer[dtype](max(1, elements))
    var seed_h = ctx.enqueue_create_host_buffer[dtype](state_elements)
    var state_h = ctx.enqueue_create_host_buffer[dtype](state_elements)
    var candidate_state_h = ctx.enqueue_create_host_buffer[dtype](
        state_elements
    )
    var expected_state_h = ctx.enqueue_create_host_buffer[dtype](state_elements)
    var weight_h = ctx.enqueue_create_host_buffer[dtype](channels * width)
    var bias_h = ctx.enqueue_create_host_buffer[dtype](channels)
    for i in range(max(1, elements)):
        input_h[i] = Scalar[dtype](0.125 + Float32(i % 31) * 0.015625)
        output_h[i] = 0
        candidate_h[i] = 0
        expected_h[i] = 0
    for i in range(state_elements):
        seed_h[i] = Scalar[dtype](0.0625 + Float32(i % 23) * 0.0078125)
        state_h[i] = 0
        candidate_state_h[i] = 0
        expected_state_h[i] = seed_h[i]
    for i in range(channels * width):
        weight_h[i] = Scalar[dtype](0.125 + Float32(i % 7) * 0.015625)
    for i in range(channels):
        bias_h[i] = Scalar[dtype](0.0625 + Float32(i % 5) * 0.015625)

    causal_conv1d_update_cpu[dtype, dtype, dtype, dtype, dtype](
        batch,
        channels,
        seqlen,
        width,
        state_len,
        TileTensor(input_h, row_major(batch, channels, seqlen)),
        TileTensor(expected_state_h, row_major(batch, channels, state_len)),
        TileTensor(weight_h, row_major(channels, width)),
        TileTensor(expected_h, row_major(batch, channels, seqlen)),
        TileTensor(bias_h, row_major(channels)),
        True,
    )

    var input_d = ctx.enqueue_create_buffer[dtype](max(1, elements))
    var output_d = ctx.enqueue_create_buffer[dtype](max(1, elements))
    var candidate_d = ctx.enqueue_create_buffer[dtype](max(1, elements))
    var seed_d = ctx.enqueue_create_buffer[dtype](state_elements)
    var state_d = ctx.enqueue_create_buffer[dtype](state_elements)
    var candidate_state_d = ctx.enqueue_create_buffer[dtype](state_elements)
    var weight_d = ctx.enqueue_create_buffer[dtype](channels * width)
    var bias_d = ctx.enqueue_create_buffer[dtype](channels)
    ctx.enqueue_copy(input_d, input_h)
    ctx.enqueue_copy(output_d, output_h)
    ctx.enqueue_copy(candidate_d, candidate_h)
    ctx.enqueue_copy(seed_d, seed_h)
    ctx.enqueue_copy(state_d, state_h)
    ctx.enqueue_copy(candidate_state_d, candidate_state_h)
    ctx.enqueue_copy(weight_d, weight_h)
    ctx.enqueue_copy(bias_d, bias_h)
    var input_tt = TileTensor(input_d, row_major(batch, channels, seqlen))
    var output_tt = TileTensor(output_d, row_major(batch, channels, seqlen))
    var candidate_tt = TileTensor(
        candidate_d, row_major(batch, channels, seqlen)
    )
    var seed_tt = TileTensor(seed_d, row_major(batch, channels, state_len))
    var state_tt = TileTensor(state_d, row_major(batch, channels, state_len))
    var candidate_state_tt = TileTensor(
        candidate_state_d, row_major(batch, channels, state_len)
    )
    var weight_tt = TileTensor(weight_d, row_major(channels, width))
    var bias_tt = TileTensor(bias_d, row_major(channels))

    def launch_baseline(
        lctx: DeviceContext,
    ) raises {
        imm input_tt,
        imm output_tt,
        imm seed_tt,
        imm state_tt,
        imm weight_tt,
        imm bias_tt,
        imm batch,
        imm seqlen,
        imm channels,
        imm width,
        imm state_len,
    }:
        enqueue_causal_conv1d_update_gpu[separate_state=False](
            lctx,
            batch,
            channels,
            seqlen,
            width,
            state_len,
            input_tt,
            seed_tt,
            state_tt,
            weight_tt,
            output_tt,
            bias_tt,
            True,
        )

    def launch_candidate(
        lctx: DeviceContext,
    ) raises {
        imm input_tt,
        imm candidate_tt,
        imm seed_tt,
        imm candidate_state_tt,
        imm weight_tt,
        imm bias_tt,
        imm batch,
        imm seqlen,
        imm channels,
        imm width,
        imm state_len,
    }:
        enqueue_causal_conv1d_update_gpu[separate_state=True](
            lctx,
            batch,
            channels,
            seqlen,
            width,
            state_len,
            input_tt,
            seed_tt,
            candidate_state_tt,
            weight_tt,
            candidate_tt,
            bias_tt,
            True,
        )

    ctx.synchronize()
    launch_baseline(ctx)
    launch_candidate(ctx)

    ctx.enqueue_copy(output_h, output_d)
    ctx.enqueue_copy(candidate_h, candidate_d)
    ctx.enqueue_copy(candidate_state_h, candidate_state_d)
    ctx.enqueue_copy(state_h, state_d)
    var seed_after_h = ctx.enqueue_create_host_buffer[dtype](state_elements)
    ctx.enqueue_copy(seed_after_h, seed_d)
    ctx.synchronize()
    comptime tolerance = 0.02 if dtype == DType.bfloat16 else 1e-4
    comptime bits_dtype = DType.uint16 if dtype == DType.bfloat16 else DType.uint32
    for i in range(elements):
        assert_almost_equal(output_h[i], expected_h[i], rtol=tolerance)
        assert_equal(
            bitcast[bits_dtype](candidate_h[i]),
            bitcast[bits_dtype](output_h[i]),
        )
    for i in range(state_elements):
        assert_equal(state_h[i], expected_state_h[i])
        assert_equal(seed_after_h[i], seed_h[i])
        assert_equal(
            bitcast[bits_dtype](candidate_state_h[i]),
            bitcast[bits_dtype](state_h[i]),
        )

    var description = (
        String(t"kernel=causal_conv1d_update dtype={dtype} batch={batch} ")
        + String(t"channels={channels} seqlen={seqlen} width={width} ")
        + String(t"state_len={state_len} bias=true activation=silu")
    )
    for _ in range(WARMUP):
        launch_baseline(ctx)
        launch_candidate(ctx)
    ctx.synchronize()
    var benchmark = Bench()
    for round_index in range(ROUNDS):
        var baseline_times = List[Float64]()
        var candidate_times = List[Float64]()
        for sample in range(SAMPLES):
            if (round_index + sample) % 2 == 0:
                baseline_times.append(
                    Float64(ctx.execution_time(launch_baseline, 1))
                )
                candidate_times.append(
                    Float64(ctx.execution_time(launch_candidate, 1))
                )
            else:
                candidate_times.append(
                    Float64(ctx.execution_time(launch_candidate, 1))
                )
                baseline_times.append(
                    Float64(ctx.execution_time(launch_baseline, 1))
                )
        for sample in range(SAMPLES):
            var order = String("AB") if (
                round_index + sample
            ) % 2 == 0 else String("BA")
            print(
                t"paired {description} round={round_index} "
                t"index={sample} order={order} "
                t"baseline_ns={baseline_times[sample]} "
                t"candidate_ns={candidate_times[sample]}"
            )
        print(
            t"summary {description} round={round_index} pairs={SAMPLES} "
            t"baseline_median_ns={_median(baseline_times)} "
            t"candidate_median_ns={_median(candidate_times)}"
        )
        var baseline_report = Report()
        var candidate_report = Report()
        for sample in range(SAMPLES):
            baseline_report.runs.append(
                Batch(Int(baseline_times[sample]), 1, True)
            )
            candidate_report.runs.append(
                Batch(Int(candidate_times[sample]), 1, True)
            )
        benchmark.info_vec.append(
            BenchmarkInfo(
                String(t"baseline {description} round={round_index}"),
                baseline_report^,
            )
        )
        benchmark.info_vec.append(
            BenchmarkInfo(
                String(t"candidate {description} round={round_index}"),
                candidate_report^,
            )
        )
    benchmark.dump_report()
    _ = input_d^
    _ = output_d^
    _ = candidate_d^
    _ = seed_d^
    _ = state_d^
    _ = candidate_state_d^
    _ = weight_d^
    _ = bias_d^


def main() raises:
    var dtype = arg_parse("dtype", String(""))
    if dtype != "bfloat16" and dtype != "float32":
        raise Error("dtype must be bfloat16 or float32")
    var batch = arg_parse("batch", 0)
    var seqlen = arg_parse("seqlen", -1)
    var channels = arg_parse("channels", 0)
    var width = arg_parse("width", 0)
    var state_len = arg_parse("state_len", 0)
    assert_true(batch > 0 and channels > 0 and seqlen >= 0)
    assert_true(1 <= width <= 4 and state_len >= width - 1 and state_len > 0)
    with DeviceContext() as ctx:
        assert_true(ctx.is_compatible(), "A compatible GPU is required")
        print(
            t"config"
            t" warmup={WARMUP} pairs={SAMPLES} rounds={ROUNDS} launches_per_sample=1"
            t" source_state=immutable_seed l2_flush=false"
            t" correctness=cpu_reference+bitwise_arm_parity"
            t" baseline=copy+original_kernel candidate=separate_state"
            t" shared_launcher=true csv_statistic=mean raw_statistic=median"
        )
        if dtype == "bfloat16":
            _bench[.bfloat16](ctx, batch, channels, seqlen, width, state_len)
        else:
            _bench[.float32](ctx, batch, channels, seqlen, width, state_len)
