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

"""Benchmarks one complete Mamba-2 SSD case with restored initial state.

The adjacent YAML declares cohorts. Required runtime options are --ssd-heads,
--ssd-head-dim, --ssd-groups and --sequence_lengths, using --name=value.
Sequence lengths are a quoted comma-separated vector, for example '[1,3,2]'.
--ssd-kernel selects dispatch, serial, unstaged, chunked, paired,
grouped-paired or auto-paired. Paired modes validate GPU outputs and state
before sampling. Raw logs retain indexed pairs and medians; standard benchmark
reports contain means of the same samples, without remeasurement.
"""

from std.benchmark import Batch, Bench, BenchmarkInfo, Report
from std.builtin.sort import sort
from std.memory.alloc import unsafe_alloc
from std.random import rand, seed
from std.time import perf_counter_ns

from internal_utils import arg_parse
from max.gpu.host import DeviceContext
from ssd_scan_test_utils.ssd_scan import (
    SSD_BASELINE,
    SSD_CANDIDATE,
    SSD_RESET,
    SSDRunObserver,
    run_ssd_gpu,
)
from state_space.mamba2_ssd_chunked import (
    GROUPED_CB_AUTO,
    GROUPED_CB_ON,
    use_chunked_ssd,
    use_grouped_cb_ssd,
)


def _median_ns(mut times: List[Int]) -> Float64:
    sort(Span(times))
    var mid = len(times) // 2
    return (Float64(times[mid - 1]) + Float64(times[mid])) / 2


def _record_samples(mut bench: Bench, name: String, samples: List[Int]) raises:
    var report = Report()
    for sample in samples:
        if sample <= 0:
            raise Error("SSD timing sample must be positive")
        report.runs.append(
            Batch(duration=sample, iterations=1, _is_significant=True)
        )
    bench.info_vec.append(BenchmarkInfo(name, report^))


struct SSDSampling(SSDRunObserver):
    """Samples complete GPU operations with state restoration outside timers."""

    comptime enabled = True

    @staticmethod
    def observe[
        Operation: def(DeviceContext, Int) raises -> None,
        //,
        grouped_pair: Bool,
        paired: Bool,
        grouped_mode: Int,
    ](ctx: DeviceContext, operation: Operation) raises:
        var report = Bench()

        def baseline(lctx: DeviceContext) raises {imm operation}:
            operation(lctx, SSD_BASELINE)

        def candidate(lctx: DeviceContext) raises {imm operation}:
            operation(lctx, SSD_CANDIDATE)

        comptime if grouped_pair:
            for _ in range(5):
                operation(ctx, SSD_RESET)
                baseline(ctx)
                operation(ctx, SSD_RESET)
                candidate(ctx)
            ctx.synchronize()

            var baseline_event = List[Int](capacity=30)
            var candidate_event = List[Int](capacity=30)
            for i in range(30):
                var baseline_ns: Int
                var candidate_ns: Int
                operation(ctx, SSD_RESET)
                if i % 2 == 0:
                    baseline_ns = ctx.execution_time(baseline, 1)
                    operation(ctx, SSD_RESET)
                    candidate_ns = ctx.execution_time(candidate, 1)
                else:
                    candidate_ns = ctx.execution_time(candidate, 1)
                    operation(ctx, SSD_RESET)
                    baseline_ns = ctx.execution_time(baseline, 1)
                baseline_event.append(baseline_ns)
                candidate_event.append(candidate_ns)
            for i in range(30):
                print(
                    "SSD_AUTO_EVENT_PAIR," if grouped_mode
                    == GROUPED_CB_AUTO else "SSD_GROUPED_EVENT_PAIR,",
                    i,
                    ",",
                    "AB" if i % 2 == 0 else "BA",
                    ",",
                    baseline_event[i],
                    ",",
                    candidate_event[i],
                    sep="",
                )
            _record_samples(report, "off_event_mean", baseline_event)
            _record_samples(
                report,
                "auto_event_mean" if grouped_mode
                == GROUPED_CB_AUTO else "on_event_mean",
                candidate_event,
            )
            print(
                "SSD_AUTO_EVENT_MEDIAN_NS,baseline_off," if grouped_mode
                == GROUPED_CB_AUTO else "SSD_GROUPED_EVENT_MEDIAN_NS,baseline_false,",
                _median_ns(baseline_event),
                ",candidate_auto," if grouped_mode
                == GROUPED_CB_AUTO else ",candidate_true,",
                _median_ns(candidate_event),
                sep="",
            )

            def wall_baseline() raises {imm} -> Int:
                operation(ctx, SSD_RESET)
                ctx.synchronize()
                var start = perf_counter_ns()
                baseline(ctx)
                ctx.synchronize()
                return perf_counter_ns() - start

            def wall_candidate() raises {imm} -> Int:
                operation(ctx, SSD_RESET)
                ctx.synchronize()
                var start = perf_counter_ns()
                candidate(ctx)
                ctx.synchronize()
                return perf_counter_ns() - start

            var baseline_wall = List[Int](capacity=30)
            var candidate_wall = List[Int](capacity=30)
            for i in range(30):
                var baseline_ns: Int
                var candidate_ns: Int
                if i % 2 == 0:
                    baseline_ns = wall_baseline()
                    candidate_ns = wall_candidate()
                else:
                    candidate_ns = wall_candidate()
                    baseline_ns = wall_baseline()
                baseline_wall.append(baseline_ns)
                candidate_wall.append(candidate_ns)
            for i in range(30):
                print(
                    "SSD_AUTO_WALL_PAIR," if grouped_mode
                    == GROUPED_CB_AUTO else "SSD_GROUPED_WALL_PAIR,",
                    i,
                    ",",
                    "AB" if i % 2 == 0 else "BA",
                    ",",
                    baseline_wall[i],
                    ",",
                    candidate_wall[i],
                    sep="",
                )
            _record_samples(report, "off_wall_mean", baseline_wall)
            _record_samples(
                report,
                "auto_wall_mean" if grouped_mode
                == GROUPED_CB_AUTO else "on_wall_mean",
                candidate_wall,
            )
            print(
                "SSD_AUTO_WALL_MEDIAN_NS,baseline_off," if grouped_mode
                == GROUPED_CB_AUTO else "SSD_GROUPED_WALL_MEDIAN_NS,baseline_false,",
                _median_ns(baseline_wall),
                ",candidate_auto," if grouped_mode
                == GROUPED_CB_AUTO else ",candidate_true,",
                _median_ns(candidate_wall),
                sep="",
            )
        else:
            comptime if paired:
                for _ in range(5):
                    operation(ctx, SSD_RESET)
                    baseline(ctx)
                    operation(ctx, SSD_RESET)
                    candidate(ctx)
                ctx.synchronize()

                var serial_times = List[Int](capacity=30)
                var candidate_times = List[Int](capacity=30)
                for i in range(30):
                    var serial_ns: Int
                    var candidate_ns: Int
                    # Alternate AB/BA and restore the same snapshot before each arm.
                    operation(ctx, SSD_RESET)
                    if i % 2 == 0:
                        serial_ns = ctx.execution_time(baseline, 1)
                        operation(ctx, SSD_RESET)
                        candidate_ns = ctx.execution_time(candidate, 1)
                    else:
                        candidate_ns = ctx.execution_time(candidate, 1)
                        operation(ctx, SSD_RESET)
                        serial_ns = ctx.execution_time(baseline, 1)
                    serial_times.append(serial_ns)
                    candidate_times.append(candidate_ns)
                for i in range(30):
                    print(
                        "SSD_PAIR_SAMPLE,",
                        i,
                        ",",
                        "AB" if i % 2 == 0 else "BA",
                        ",",
                        serial_times[i],
                        ",",
                        candidate_times[i],
                        sep="",
                    )
                _record_samples(report, "serial_event_mean", serial_times)
                _record_samples(report, "chunked_event_mean", candidate_times)
                print(
                    "SSD_PAIR_MEDIAN_NS,serial,",
                    _median_ns(serial_times),
                    ",chunked_candidate,",
                    _median_ns(candidate_times),
                    sep="",
                )
            else:
                for _ in range(5):
                    operation(ctx, SSD_RESET)
                    baseline(ctx)
                ctx.synchronize()

                var times = List[Int](capacity=30)
                for _ in range(30):
                    # Same-stream reset precedes the start event, outside the interval.
                    operation(ctx, SSD_RESET)
                    times.append(ctx.execution_time(baseline, 1))
                for i in range(len(times)):
                    print("SSD_SAMPLE,", i, ",", times[i], sep="")
                _record_samples(report, "event_mean", times)
                print("SSD_MEDIAN_NS,", _median_ns(times), sep="")
        report.dump_report()


def _bench_ssd[
    DSTATE_SPLIT: Int,
    CH_PER_THREAD: Int = 1,
    prefill: Bool = True,
    paired: Bool = False,
    grouped_pair: Bool = False,
    grouped_mode: Int = GROUPED_CB_ON,
](
    nheads: Int,
    head_dim: Int,
    ngroups: Int,
    max_slots: Int,
    seq_lengths: List[Int],
    ctx: DeviceContext,
) raises:
    comptime dtype = DType.bfloat16
    comptime DSTATE = 128
    var batch = len(seq_lengths)
    var total_len = 0
    for i in range(len(seq_lengths)):
        total_len += seq_lengths[i]
    var x_row = nheads * head_dim
    var bc_row = ngroups * DSTATE
    var state_size = nheads * head_dim * DSTATE
    var pool_size = max_slots * state_size
    var y_size = total_len * nheads * head_dim

    var x_h = unsafe_alloc[Scalar[dtype]](total_len * x_row)
    var dt_h = unsafe_alloc[Scalar[dtype]](total_len * nheads)
    var A_h = unsafe_alloc[Scalar[dtype]](nheads)
    var B_h = unsafe_alloc[Scalar[dtype]](total_len * bc_row)
    var C_h = unsafe_alloc[Scalar[dtype]](total_len * bc_row)
    var D_h = unsafe_alloc[Scalar[dtype]](nheads)
    var dt_bias_h = unsafe_alloc[Scalar[dtype]](nheads)
    var is_h = unsafe_alloc[Float32](batch * state_size)
    var his_h = unsafe_alloc[Scalar[.bool]](batch)
    var us_h = unsafe_alloc[Scalar[.bool]](batch)
    var qsl_h = unsafe_alloc[Int32](batch + 1)
    var slot_h = unsafe_alloc[UInt32](batch)
    var pool_h = unsafe_alloc[Float32](pool_size)
    var y_h = unsafe_alloc[Scalar[dtype]](y_size + 64)

    # Match the seeded inputs used for the original before/after measurements.
    seed(1234)
    rand(x_h, total_len * x_row)
    rand(dt_h, total_len * nheads)
    rand(A_h, nheads)
    rand(B_h, total_len * bc_row)
    rand(C_h, total_len * bc_row)
    rand(D_h, nheads)
    rand(dt_bias_h, nheads)
    rand(is_h, batch * state_size)
    for i in range(nheads):
        A_h.unsafe_store(i, Scalar[dtype](-Float32(A_h.unsafe_load(i)) - 0.1))
    for i in range(total_len * nheads):
        dt_h.unsafe_store(i, Scalar[dtype](Float32(dt_h.unsafe_load(i)) - 0.5))
    for i in range(pool_size):
        pool_h.unsafe_store(i, 0)
    var offset = 0
    qsl_h.unsafe_store(0, 0)
    for b in range(batch):
        his_h.unsafe_store(b, True)
        us_h.unsafe_store(b, True)
        slot_h.unsafe_store(b, UInt32(b + 1))
        offset += seq_lengths[b]
        qsl_h.unsafe_store(b + 1, Int32(offset))
        for i in range(state_size):
            pool_h.unsafe_store(
                (b + 1) * state_size + i, is_h.unsafe_load(b * state_size + i)
            )
    print(
        "SSD_SHAPE,nheads=",
        nheads,
        ",head_dim=",
        head_dim,
        ",groups=",
        ngroups,
        ",batch=",
        batch,
        ",tokens=",
        total_len,
        ",split=",
        DSTATE_SPLIT,
        ",channels=",
        CH_PER_THREAD,
        ",prefill=",
        prefill,
        ",lengths=",
        seq_lengths,
        sep="",
    )
    run_ssd_gpu[
        DSTATE,
        DSTATE_SPLIT,
        CH_PER_THREAD,
        prefill,
        compare_chunked=paired,
        guard_output=True,
        grouped_pair=grouped_pair,
        grouped_mode=grouped_mode,
        Observer=SSDSampling,
    ](
        ctx,
        nheads,
        head_dim,
        ngroups,
        max_slots,
        batch,
        total_len,
        x_row,
        bc_row,
        nheads,
        nheads,
        batch,
        0,
        x_h,
        dt_h,
        A_h,
        B_h,
        C_h,
        D_h,
        dt_bias_h,
        qsl_h,
        his_h,
        us_h,
        slot_h,
        y_h,
        pool_h,
    )
    x_h.unsafe_free()
    dt_h.unsafe_free()
    A_h.unsafe_free()
    B_h.unsafe_free()
    C_h.unsafe_free()
    D_h.unsafe_free()
    dt_bias_h.unsafe_free()
    is_h.unsafe_free()
    his_h.unsafe_free()
    us_h.unsafe_free()
    qsl_h.unsafe_free()
    slot_h.unsafe_free()
    pool_h.unsafe_free()
    y_h.unsafe_free()


def _benchmark_ssd_shape(
    ctx: DeviceContext,
    nheads: Int,
    head_dim: Int,
    ngroups: Int,
    seq_lengths: List[Int],
    var kernel: String,
) raises:
    comptime WIDE = ctx.target.is_nvidia_gpu()
    comptime SPLIT = 16 if WIDE else 8
    comptime CHANNELS = 4 if WIDE else 1
    var total_len = 0
    for i in range(len(seq_lengths)):
        total_len += seq_lengths[i]
    var batch = len(seq_lengths)
    if kernel == "dispatch":
        kernel = "serial" if total_len > batch else "unstaged"
        comptime if ctx.target._is_sm_8x_or_newer():
            if use_chunked_ssd(total_len, batch, head_dim, 512):
                kernel = "chunked"
    match kernel:
        case "chunked" | "paired":
            comptime if not (ctx.target._is_sm_8x_or_newer()):
                raise Error("chunked SSD requires NVIDIA SM80 or newer")
            if head_dim % 64 != 0:
                raise Error("chunked SSD requires head_dim divisible by 64")
            if kernel == "paired":
                comptime for mode in range(2):
                    comptime has_prefill = mode == 1
                    if has_prefill == (total_len > batch):
                        _bench_ssd[
                            SPLIT,
                            CH_PER_THREAD=CHANNELS,
                            prefill=has_prefill,
                            paired=True,
                        ](
                            nheads,
                            head_dim,
                            ngroups,
                            batch + 1,
                            seq_lengths,
                            ctx,
                        )
            else:
                _bench_ssd[-1](
                    nheads, head_dim, ngroups, batch + 1, seq_lengths, ctx
                )
        case "auto-paired":
            comptime if not (ctx.target._is_sm_8x_or_newer()):
                raise Error("AUTO SSD requires NVIDIA SM80 or newer")
            if head_dim % 64 != 0:
                raise Error("AUTO SSD requires head_dim divisible by 64")
            print("SSD_AUTO_DEVICE_NAME,", ctx.name(), sep="")
            print(
                "SSD_AUTO_SHAPE_ELIGIBLE,",
                use_grouped_cb_ssd(total_len, batch, nheads, head_dim, ngroups),
                sep="",
            )
            print(
                "SSD_AUTO_HARDWARE_ELIGIBLE,",
                (
                    ctx.target.is_nvidia_gpu["sm_100"]()
                    or ctx.target.is_nvidia_gpu["sm_100a"]()
                )
                and ctx.name() == "NVIDIA B200",
                sep="",
            )
            _bench_ssd[
                -1,
                grouped_pair=True,
                grouped_mode=GROUPED_CB_AUTO,
            ](nheads, head_dim, ngroups, batch + 1, seq_lengths, ctx)
        case "grouped-paired":
            comptime if not (ctx.target._is_sm_8x_or_newer()):
                raise Error("grouped SSD requires NVIDIA SM80 or newer")
            if head_dim % 64 != 0:
                raise Error("grouped SSD requires head_dim divisible by 64")
            _bench_ssd[
                -1,
                grouped_pair=True,
                grouped_mode=GROUPED_CB_ON,
            ](nheads, head_dim, ngroups, batch + 1, seq_lengths, ctx)
        case "serial":
            _bench_ssd[
                SPLIT,
                CH_PER_THREAD=CHANNELS,
                prefill=True,
            ](nheads, head_dim, ngroups, batch + 1, seq_lengths, ctx)
        case "unstaged":
            _bench_ssd[
                SPLIT,
                CH_PER_THREAD=CHANNELS,
                prefill=False,
            ](nheads, head_dim, ngroups, batch + 1, seq_lengths, ctx)
        case _:
            raise Error(
                "--ssd-kernel must be serial, unstaged, chunked, paired,"
                + " grouped-paired, auto-paired, or dispatch"
            )


def main() raises:
    var nheads = arg_parse("ssd-heads", 0)
    var head_dim = arg_parse("ssd-head-dim", 0)
    var ngroups = arg_parse("ssd-groups", 0)
    var lengths_arg = arg_parse("sequence_lengths", String(""))
    var kernel = arg_parse("ssd-kernel", String("dispatch"))
    if nheads <= 0 or head_dim <= 0 or ngroups <= 0:
        raise Error(
            "positive --ssd-heads, --ssd-head-dim and --ssd-groups are required"
        )
    if nheads % ngroups != 0:
        raise Error("groups must divide the head count")
    if not lengths_arg:
        raise Error(
            "--sequence_lengths is required; see bench_mamba2_ssd_scan.yaml"
        )
    var seq_lengths = List[Int]()
    for token in lengths_arg.strip("[]").split(","):
        var length = Int(token.strip())
        if length <= 0:
            raise Error("sequence lengths must be positive")
        seq_lengths.append(length)
    with DeviceContext() as ctx:
        comptime if not (ctx.target.is_nvidia_gpu() or ctx.target.is_amd_gpu()):
            raise Error("SSD benchmarks require a CUDA or HIP GPU")
        print("SSD_BENCHMARK,kernel=", kernel, sep="")
        _benchmark_ssd_shape(
            ctx, nheads, head_dim, ngroups, seq_lengths, kernel^
        )
