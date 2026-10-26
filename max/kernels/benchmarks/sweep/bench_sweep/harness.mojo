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
"""Measurement policy and output for a sweep process.

`SweepHarness` owns the `Bench`, the `PhaseTimer` and the measurement policy:

- `--target_time=<s>` measures each benchmark for about that long instead of
  a fixed iteration count. 0 keeps `Bench`'s stock policy, which stops a GPU
  benchmark after 100 iterations however short they are. Otherwise a timed
  warmup sets a first rate, a calibration batch a quarter of the target long
  ramps the clocks and refines the rate, and the one measured batch is sized
  from that to last the target. Only the measured batch counts, so the clock
  ramp stays out of the mean, and a benchmark costs about 1.25x its target
  instead of `Bench`'s batch ladder of discarded rungs.
- `--l2=cold|flush|warm` sets the cache state each iteration starts from.
  `cold`, the default (`SweepHarness`'s `default_l2` argument moves it, for
  a target whose sweeps run another way), rotates the benchmark's inputs: it
  keeps `copies(n)`
  copies of what a launch reads, and iteration `i` reads copy `i % copies`,
  so a copy is evicted before it is read again. Nothing runs between
  iterations, so `Bench` times batches as it does warm. `flush` overwrites a
  buffer before every iteration instead and times each iteration on its own,
  spending the target time in wall time; it costs a sync per iteration and
  leaves dirty lines behind, so it is the fallback for launches that cannot
  rotate. `warm` reads copy 0 every time.
- `--l2_bytes=auto|<MiB>` is what a cold iteration must evict: rotated copies
  span it, or a flush overwrites it. `auto` is twice the reported L2, and on
  AMD at least 512 MiB, so the Infinity Cache the runtime does not report is
  covered too.
- `--rotate_mb=<MiB>` (4096) caps the memory a config's extra copies may
  take. A config that would need more, such as one whose few touched bytes
  each cost a whole page, is flushed instead.
- `--throttle_check=warn|fail|both|off` reads the device's hardware throttle
  state before the first measurement, where the platform's management tool
  exposes it; `fail` raises if it is throttled and `both` checks again after
  the last measurement. A check runs the tool once, which costs 1-3 s.
- `--json=<path>` writes every benchmark's config fields, timing and the
  phase totals, so analysis joins on fields rather than parsing names. Under
  `bazel test` a relative path lands in the test's undeclared outputs.

`SWEEPINFO` and `SWEEPTIME` lines and the `Bench` report are printed as
before; `sweeptime.py` sums `SWEEPTIME` lines across processes.

A target the library does not know passes its own `SweepBackend` (see
`backend.mojo`); the harness then fills on the host and reports the timing
as the backend describes it.
"""

from std.benchmark import Bench, Bencher, BenchId
from std.benchmark import ThroughputMeasure
from std.math import ceildiv
from std.os import getenv
from std.time import perf_counter_ns

from internal_utils import arg_parse
from max.benchmark import bencher_iter_custom
from max.gpu.host import DeviceBuffer, DeviceContext

from .backend import SweepBackend
from .driver import Config, Footprint
from .timing import PhaseTimer, bench_charged, charge_calibrated_bench

# Flushed iterations each cost a sync, so the count is set from a probe rather
# than ramped; the bounds keep a slow kernel measured and a fast one finite.
comptime _MIN_FLUSHED_ITERS = 10
comptime _MAX_FLUSHED_ITERS = 100_000
comptime _FLUSH_PROBE_ITERS = 3


comptime _HEX_DIGITS: StaticString = "0123456789abcdef"


def _json_string(s: StringSlice, mut writer: Some[Writer]):
    writer.write('"')
    for cp in s.codepoint_slices():
        if cp == '"':
            writer.write('\\"')
        elif cp == "\\":
            writer.write("\\\\")
        elif cp == "\n":
            writer.write("\\n")
        elif cp == "\t":
            writer.write("\\t")
        elif cp.byte_length() == 1 and Int(cp.as_bytes()[0]) < 0x20:
            var c = Int(cp.as_bytes()[0])
            writer.write(
                "\\u00", _HEX_DIGITS[byte=c >> 4], _HEX_DIGITS[byte=c & 15]
            )
        else:
            writer.write(cp)
    writer.write('"')


struct Fields(Copyable, Movable, Writable):
    """Named values describing a benchmark or a sweep, in insertion order.

    Prints as `k=v k=v` for `SWEEPINFO` lines; `write_json` writes a JSON
    object. Values keep their JSON type: numbers and booleans stay bare.
    """

    var keys: List[String]
    var values: List[String]
    var quoted: List[Bool]

    def __init__(out self):
        self.keys = []
        self.values = []
        self.quoted = []

    def __init__(out self, cfg: Config):
        """Starts from every runtime axis of `cfg`."""
        self.keys = cfg.names.copy()
        self.values = []
        self.quoted = List[Bool](length=len(cfg.names), fill=False)
        for v in cfg.values:
            self.values.append(String(v))

    def add(mut self, key: String, value: Int):
        self._add(key, String(value), False)

    def add(mut self, key: String, value: Float64):
        self._add(key, String(value), False)

    def add(mut self, key: String, value: Bool):
        self._add(key, "true" if value else "false", False)

    def add(mut self, key: String, value: String):
        self._add(key, value, True)

    def add(mut self, key: String, value: DType):
        self._add(key, String(value), True)

    def extend(mut self, footprint: Footprint):
        """Adds every requirement of `footprint`."""
        for i in range(len(footprint.names)):
            self.add(footprint.names[i], footprint.values[i])

    def _add(mut self, key: String, value: String, quoted: Bool):
        for i in range(len(self.keys)):
            if self.keys[i] == key:
                self.values[i] = value
                self.quoted[i] = quoted
                return
        self.keys.append(key)
        self.values.append(value)
        self.quoted.append(quoted)

    def write_to(self, mut writer: Some[Writer]):
        for i in range(len(self.keys)):
            if i > 0:
                writer.write(" ")
            writer.write(self.keys[i], "=", self.values[i])

    def write_json(self, mut writer: Some[Writer]):
        writer.write("{")
        for i in range(len(self.keys)):
            if i > 0:
                writer.write(", ")
            _json_string(self.keys[i], writer)
            writer.write(": ")
            if self.quoted[i]:
                _json_string(self.values[i], writer)
            else:
                writer.write(self.values[i])
        writer.write("}")


@fieldwise_init
struct MeasurePolicy(Copyable, Movable):
    """How long each benchmark measures and what runs between iterations."""

    var target_time_s: Float64
    var l2: String
    var l2_bytes: String
    var rotate_mb: Int
    var throttle_check: String

    @staticmethod
    def from_args(default_l2: String = "cold") raises -> Self:
        var policy = Self(
            Float64(String(arg_parse("target_time", "1.0"))),
            String(arg_parse("l2", default_l2)),
            String(arg_parse("l2_bytes", "auto")),
            arg_parse("rotate_mb", 4096),
            String(arg_parse("throttle_check", "warn")),
        )
        if policy.target_time_s < 0:
            raise Error("--target_time must be >= 0")
        if policy.l2 != "cold" and policy.l2 != "flush" and policy.l2 != "warm":
            raise Error("--l2 must be cold, flush or warm, not ", policy.l2)
        if (
            policy.throttle_check != "warn"
            and policy.throttle_check != "fail"
            and policy.throttle_check != "both"
            and policy.throttle_check != "off"
        ):
            raise Error(
                "--throttle_check must be warn, fail, both or off, not ",
                policy.throttle_check,
            )
        return policy^

    def evict_bytes(
        self, ctx: DeviceContext, backend: SweepBackend
    ) raises -> Int:
        """Returns the bytes a cold iteration must evict: `--l2_bytes` when
        set, otherwise the backend's answer for the target."""
        if self.l2_bytes != "auto":
            return Int(self.l2_bytes) * 1024 * 1024
        return backend.evict_bytes(ctx)

    def write_fields(self, mut fields: Fields):
        fields.add("target_time_s", self.target_time_s)
        fields.add("l2", self.l2)
        fields.add("l2_bytes", self.l2_bytes)
        fields.add("rotate_mb", self.rotate_mb)
        fields.add("throttle_check", self.throttle_check)


@fieldwise_init
struct Rotation(ImplicitlyCopyable, Movable):
    """How many copies of its inputs a cold launch rotates through.

    A value, so fixture planning can capture it while the harness itself is
    in use.
    """

    var evict_bytes: Int
    var cold: Bool
    var max_bytes: Int

    def copies(self, touched_bytes: Int, copy_bytes: Int = 0) -> Int:
        """Returns the copies a launch that reads `touched_bytes` needs to
        span the cache under `--l2=cold`, 1 otherwise, or 0 when the copies
        past the first, `copy_bytes` each, would exceed `--rotate_mb`.

        Count only what one launch reads from memory: a sparse kernel touches
        a fraction of its cache. `copy_bytes` is what one more copy
        allocates, whole pages and per-copy inputs included. Pass 0 copies to
        `SweepHarness.bench` to flush instead; plan for at least 1.
        """
        if not self.cold:
            return 1
        var n = max(1, ceildiv(self.evict_bytes, max(touched_bytes, 1)))
        if (n - 1) * copy_bytes > self.max_bytes:
            return 0
        return n


def _time_flushed[
    LaunchFn: def(DeviceContext) raises -> None
](
    ctx: DeviceContext,
    launch: LaunchFn,
    flush: DeviceBuffer[DType.uint8],
    num_iters: Int,
) raises -> Int:
    # One timer per iteration keeps the flush out of the measurement. The
    # flush is still running when the launch is enqueued, so the timer does not
    # see the host's launch latency either.
    var total = 0
    for i in range(num_iters):
        ctx.enqueue_memset(flush, UInt8(i & 0xFF))
        total += ctx.execution_time(launch, 1)
    return total


struct SweepHarness(Movable):
    """The `Bench`, `PhaseTimer` and measurement policy of a sweep process.

    Create it first thing in `main`, so its timer covers the whole process,
    and call `start` once the device context exists and `stop` before it
    closes. Laps between them are the benchmark's own.

    A plugin-backed target passes its own `SweepBackend`; every other target
    keeps the `auto` one the library builds from `ctx`'s portable APIs.
    """

    var m: Bench
    var timer: PhaseTimer
    var policy: MeasurePolicy
    var backend: SweepBackend
    var tag: String
    var json_path: String
    var evict_bytes: Int
    var flush: Optional[DeviceBuffer[DType.uint8]]
    var info: List[Fields]
    var records: List[String]

    def __init__(
        out self,
        default_tag: String,
        backend: SweepBackend = SweepBackend.auto(),
        default_l2: String = "cold",
    ) raises:
        self.timer = PhaseTimer()
        self.policy = MeasurePolicy.from_args(default_l2)
        self.m = Bench()
        self.backend = backend.copy()
        self.tag = String(arg_parse("tag", default_tag))
        self.json_path = String(arg_parse("json", ""))
        var outputs = getenv("TEST_UNDECLARED_OUTPUTS_DIR", "")
        if (
            self.json_path.byte_length() > 0
            and not self.json_path.startswith("/")
            and outputs.byte_length() > 0
        ):
            self.json_path = String(outputs, "/", self.json_path)
        self.evict_bytes = 0
        self.flush = None
        self.info = []
        self.records = []

    def start(mut self, ctx: DeviceContext) raises:
        """Sizes the cache to evict, allocates a flush buffer under
        `--l2=flush`, and checks throttling.

        Charged to `harness_setup`.
        """
        self.timer.reset()
        self.evict_bytes = self.policy.evict_bytes(ctx, self.backend)
        if self.policy.l2 == "flush":
            self.flush = ctx.enqueue_create_buffer[DType.uint8](
                self.evict_bytes
            )
        var fields = Fields()
        fields.add("tag", self.tag)
        fields.add("device", ctx.name())
        fields.add("api", ctx.api())
        self.policy.write_fields(fields)
        fields.add("evict_bytes", self.evict_bytes)
        fields.add(
            "fill", "device" if self.backend.device_fill(ctx) else "host"
        )
        fields.add(
            "timing",
            "device_events" if self.backend.device_events(
                ctx
            ) else "host_clock",
        )
        self._check_throttle(ctx, "throttle_before", fields)
        self.note(fields^)
        ctx.synchronize()
        self.timer.lap("harness_setup")

    def stop(mut self, ctx: DeviceContext) raises:
        """Frees the flush buffer and, under `--throttle_check=both`, checks
        throttling again, so a sweep that heated the device says so.

        Charged to `harness_teardown`.
        """
        self.timer.reset()
        if self.policy.throttle_check == "both":
            var fields = Fields()
            self._check_throttle(ctx, "throttle_after", fields)
            self.note(fields^)
        self.flush = None
        self.timer.lap("harness_teardown")

    def _check_throttle(
        mut self, ctx: DeviceContext, key: String, mut fields: Fields
    ) raises:
        if self.policy.throttle_check == "off":
            return
        var state = self.backend.throttle_state(ctx)
        fields.add(key, state)
        if state.startswith("active"):
            print("WARNING: device is hardware-throttled (", key, "): ", state)
            if (
                self.policy.throttle_check == "fail"
                and key == "throttle_before"
            ):
                raise Error("device is hardware-throttled: ", state)

    def note(mut self, var fields: Fields):
        """Prints a `SWEEPINFO` line and keeps it for the JSON file."""
        print("SWEEPINFO ", fields, sep="")
        self.info.append(fields^)

    def rotation(self) -> Rotation:
        """Returns the rotation policy; valid after `start`."""
        return Rotation(
            self.evict_bytes,
            self.policy.l2 == "cold",
            self.policy.rotate_mb * 1024 * 1024,
        )

    def first_launch[
        LaunchFn: def(DeviceContext) raises -> None
    ](mut self, ctx: DeviceContext, launch: LaunchFn) raises:
        """Runs `launch` once and charges it, module load included, to
        `first_launch`."""
        self.timer.reset()
        launch(ctx)
        ctx.synchronize()
        self.timer.lap("first_launch")

    def first_launch[
        LaunchFn: def(DeviceContext, Int) raises -> None
    ](mut self, ctx: DeviceContext, launch: LaunchFn) raises:
        """Runs copy 0 of a rotating `launch` once; see the other overload."""
        self.timer.reset()
        launch(ctx, 0)
        ctx.synchronize()
        self.timer.lap("first_launch")

    def bench[
        LaunchFn: def(DeviceContext, Int) raises -> None
    ](
        mut self,
        ctx: DeviceContext,
        launch: LaunchFn,
        bench_id: BenchId,
        var fields: Fields,
        copies: Int,
        measures: List[ThroughputMeasure] = {},
    ) raises:
        """Measures a rotating `launch`, which reads copy `i % copies` of
        its inputs at iteration `i`, and records it.

        `copies` should come from `rotation().copies()`. Under `--l2=flush`,
        or with 0 copies, every iteration reads copy 0 behind a flush.
        """
        fields.add("copies", copies)

        @inline(.always)
        def launch_copy(launch_ctx: DeviceContext, i: Int) raises {imm}:
            launch(launch_ctx, i % copies)

        @inline(.always)
        def launch_first(launch_ctx: DeviceContext) raises {imm}:
            launch(launch_ctx, 0)

        if self.policy.l2 == "flush" or copies == 0:
            self._ensure_flush(ctx)
            self._bench_flushed(ctx, launch_first, bench_id, fields^, measures)
            return

        if self.policy.target_time_s > 0:
            self._bench_target_time(
                ctx, launch_copy, bench_id, fields^, measures
            )
            return

        @inline(.always)
        def bench_func(mut b: Bencher) raises {imm}:
            bencher_iter_custom(b, launch_copy, ctx)

        bench_charged(self.m, self.timer, bench_func, bench_id, measures)
        self._record(fields, measures)

    def bench[
        LaunchFn: def(DeviceContext) raises -> None
    ](
        mut self,
        ctx: DeviceContext,
        launch: LaunchFn,
        bench_id: BenchId,
        var fields: Fields,
        measures: List[ThroughputMeasure] = {},
    ) raises:
        """Measures `launch`, which cannot rotate its inputs, and records it.

        `fields` should identify the benchmark completely: the runtime axes,
        the comptime arms and anything else its name encodes. Under
        `--l2=cold` it falls back to a flush, since it has no copies to
        rotate.
        """
        if self.policy.l2 == "warm":
            if self.policy.target_time_s > 0:

                @inline(.always)
                def launch_warm(launch_ctx: DeviceContext, i: Int) raises {imm}:
                    _ = i
                    launch(launch_ctx)

                self._bench_target_time(
                    ctx, launch_warm, bench_id, fields^, measures
                )
                return

            @inline(.always)
            def bench_warm(mut b: Bencher) raises {imm}:
                bencher_iter_custom(b, launch, ctx)

            bench_charged(self.m, self.timer, bench_warm, bench_id, measures)
            self._record(fields, measures)
            return
        self._ensure_flush(ctx)
        self._bench_flushed(ctx, launch, bench_id, fields^, measures)

    def _bench_target_time[
        LaunchFn: def(DeviceContext, Int) raises -> None
    ](
        mut self,
        ctx: DeviceContext,
        launch: LaunchFn,
        bench_id: BenchId,
        var fields: Fields,
        measures: List[ThroughputMeasure],
    ) raises:
        """Measures `launch` for `--target_time` through one calibration batch.

        `Bench` would grow batches toward the target with each rung capped at
        10x the last, so a benchmark measured over n iterations first pays
        rungs of about n/1000, n/100 and n/10, each behind its own timer sync,
        and discards them all. Instead the warmup sets a first rate, the
        calibration batch lasts a quarter of the target -- ramping the clocks,
        so the measured batch starts at the sustained clocks the target
        exists to measure -- and the measured batch is sized from its rate to
        last the target. Only the measured batch counts; counting the
        calibration instead would put the clock ramp inside the mean.
        """
        comptime _CALIBRATION_SHARE = 0.25
        var target_ns = Int(self.policy.target_time_s * 1e9)
        var num_warmup = self.m.config.num_warmup_iters

        var t0 = perf_counter_ns()
        var warmup_ns = ctx.execution_time_iter(launch, num_warmup)
        # Both batch sizes go through a measured rate, never a fixed iteration
        # count: 100 iterations of a slow kernel is whole targets of
        # calibration, and of a fast one too little to ramp anything.
        var warmup_rate = Float64(warmup_ns) / Float64(max(num_warmup, 1))
        var cal_iters = min(
            max(
                Int(
                    Float64(target_ns)
                    * _CALIBRATION_SHARE
                    / max(warmup_rate, 1.0)
                ),
                1,
            ),
            1_000_000_000,
        )
        var cal_ns = ctx.execution_time_iter(launch, cal_iters)
        var cal_rate = Float64(cal_ns) / Float64(cal_iters)
        var iters = min(
            max(Int(Float64(target_ns) / max(cal_rate, 1.0)), 1),
            1_000_000_000,
        )

        @inline(.always)
        def bench_func(mut b: Bencher) raises {imm}:
            bencher_iter_custom(b, launch, ctx)

        self.m.bench_function(
            bench_func, bench_id, measures, fixed_iterations=iters
        )
        charge_calibrated_bench(
            self.timer,
            self.m,
            perf_counter_ns() - t0,
            warmup_ns,
            cal_ns,
        )
        self.timer.reset()
        self._record(fields, measures, calibrate_ns=cal_ns)

    def _ensure_flush(mut self, ctx: DeviceContext) raises:
        """Allocates the flush buffer on first use."""
        if not self.flush:
            self.flush = ctx.enqueue_create_buffer[DType.uint8](
                self.evict_bytes
            )

    def _bench_flushed[
        LaunchFn: def(DeviceContext) raises -> None
    ](
        mut self,
        ctx: DeviceContext,
        launch: LaunchFn,
        bench_id: BenchId,
        var fields: Fields,
        measures: List[ThroughputMeasure],
    ) raises:
        fields.add("flushed", True)
        var flush = self.flush.copy()

        @inline(.always)
        def bench_func(mut b: Bencher) raises {imm}:
            b.elapsed = _time_flushed(ctx, launch, flush.value(), b.num_iters)

        var saved = self.m.config.copy()
        if self.policy.target_time_s > 0:
            var iters = self._flushed_iters(ctx, launch, flush.value())
            self.m.config.min_runtime_secs = 0
            self.m.config.max_runtime_secs = 1e9
            self.m.config.max_batch_size = iters
            self.m.config.max_iters = iters
        try:
            bench_charged(self.m, self.timer, bench_func, bench_id, measures)
        finally:
            self.m.config = saved^
        self._record(fields, measures)

    def _record(
        mut self,
        fields: Fields,
        measures: List[ThroughputMeasure],
        calibrate_ns: Int = 0,
    ):
        """Appends the JSON record of the benchmark just run."""
        ref info = self.m.info_vec[len(self.m.info_vec) - 1]
        ref res = info.result
        var record = String()
        record.write('{"name": ')
        _json_string(info.name, record)
        record.write(', "fields": ')
        fields.write_json(record)
        record.write(
            ', "met_ms": ',
            res.mean("ms"),
            ', "iters": ',
            res.iters(),
            ', "measured_s": ',
            res.duration("s"),
            ', "warmup_s": ',
            Float64(res.warmup_duration) * 1e-9,
            ', "calibrate_s": ',
            Float64(calibrate_ns) * 1e-9,
            ', "measures": {',
        )
        for i in range(len(measures)):
            if i > 0:
                record.write(", ")
            _json_string(measures[i].metric.name, record)
            record.write(": ", measures[i].value)
        record.write("}}")
        self.records.append(record^)

    def _flushed_iters[
        LaunchFn: def(DeviceContext) raises -> None
    ](
        mut self,
        ctx: DeviceContext,
        launch: LaunchFn,
        flush: DeviceBuffer[DType.uint8],
    ) raises -> Int:
        """Returns the flushed iterations that fill the target wall time.

        `Bench` would size batches by the kernel time it is given, which
        under a flush is a small part of each iteration's wall time. The
        probe is charged to `flush_probe`.
        """
        self.timer.reset()
        var t0 = perf_counter_ns()
        _ = _time_flushed(ctx, launch, flush, _FLUSH_PROBE_ITERS)
        var per_iter = Float64(perf_counter_ns() - t0) / _FLUSH_PROBE_ITERS
        self.timer.lap("flush_probe")
        var iters = Int(self.policy.target_time_s * 1e9 / max(per_iter, 1.0))
        return min(max(iters, _MIN_FLUSHED_ITERS), _MAX_FLUSHED_ITERS)

    def finish(mut self) raises:
        """Prints the `Bench` report and the phase totals, and writes JSON."""
        self.timer.reset()
        self.m.dump_report()
        self.timer.lap("report")
        self.timer.finish(self.tag)
        if self.json_path.byte_length() == 0:
            return
        var out = String('{"tag": ')
        _json_string(self.tag, out)
        out.write(',\n "info": [')
        for i in range(len(self.info)):
            out.write(",\n  " if i > 0 else "\n  ")
            self.info[i].write_json(out)
        out.write('],\n "phases": {')
        for i in range(len(self.timer.names)):
            out.write(",\n  " if i > 0 else "\n  ")
            _json_string(self.timer.names[i], out)
            out.write(
                ': {"count": ',
                self.timer.counts[i],
                ', "seconds": ',
                Float64(self.timer.totals_ns[i]) * 1e-9,
                "}",
            )
        out.write('},\n "benchmarks": [')
        for i in range(len(self.records)):
            out.write(",\n  " if i > 0 else "\n  ")
            out.write(self.records[i])
        out.write("]}\n")
        with open(self.json_path, "w") as f:
            f.write(out)
