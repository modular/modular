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
"""Phase accounting: where the wall time of a sweep process goes."""

from std.benchmark import Bench, Bencher, BenchId, ThroughputMeasure
from std.time import perf_counter_ns


def seconds_since_process_start() -> Float64:
    """Returns wall seconds since the kernel started this process.

    Covers what happens before `main` runs: loading the binary and its shared
    libraries and the Mojo runtime's own init. Resolution is one clock tick
    (10 ms on Linux), which is fine for a quantity measured in hundreds of ms.
    Returns 0 where `/proc` is unavailable, such as macOS.
    """
    try:
        var stat = open("/proc/self/stat", "r").read()
        # Field 2 (comm) may contain spaces; everything after the last ')' is
        # fields 3.. separated by single spaces. starttime is field 22.
        var tail = stat.split(")")
        var fields = tail[len(tail) - 1].split(" ")
        var start_ticks = Float64(Int(fields[20]))
        var uptime = Float64(open("/proc/uptime", "r").read().split(" ")[0])
        return uptime - start_ticks / 100.0
    except:
        return 0.0


struct PhaseTimer(Movable):
    """Accumulates wall time per named phase, in first-seen order.

    `lap` charges everything since the previous `lap` (or `reset`) to a phase,
    so consecutive laps tile the timeline with no gaps. `reset` drops the time
    since the last lap, for spans another call already charged.
    """

    var names: List[String]
    var totals_ns: List[Int]
    var counts: List[Int]
    var _last: Int
    var _start: Int
    var _pre_main_ns: Int

    def __init__(out self):
        """Starts the timeline now and records the time spent before `main`."""
        self.names = []
        self.totals_ns = []
        self.counts = []
        self._last = perf_counter_ns()
        self._start = self._last
        self._pre_main_ns = Int(seconds_since_process_start() * 1e9)

    def reset(mut self):
        self._last = perf_counter_ns()

    def add(mut self, name: String, ns: Int):
        for i in range(len(self.names)):
            if self.names[i] == name:
                self.totals_ns[i] += ns
                self.counts[i] += 1
                return
        self.names.append(name)
        self.totals_ns.append(ns)
        self.counts.append(1)

    def lap(mut self, name: String):
        var now = perf_counter_ns()
        self.add(name, now - self._last)
        self._last = now

    def total_s(self, name: String) -> Float64:
        for i in range(len(self.names)):
            if self.names[i] == name:
                return Float64(self.totals_ns[i]) * 1e-9
        return 0.0

    def report(self, tag: String):
        """Prints one machine-readable `SWEEPTIME` line per phase."""
        for i in range(len(self.names)):
            print(
                "SWEEPTIME,",
                tag,
                ",",
                self.names[i],
                ",",
                self.counts[i],
                ",",
                Float64(self.totals_ns[i]) * 1e-9,
                sep="",
            )

    def finish(mut self, tag: String):
        """Charges `pre_main` and `process_total`, then prints the report."""
        self.add("pre_main", self._pre_main_ns)
        self.add(
            "process_total",
            perf_counter_ns() - self._start + self._pre_main_ns,
        )
        self.report(tag)


def charge_last_bench(mut timer: PhaseTimer, m: Bench, wall_ns: Int):
    """Splits the wall time of the last `bench_function` into its parts.

    `bench_measured` is the device time of the iterations the report averages;
    `bench_warmup` is the timed warmup batch; `bench_overhead` is the rest:
    work the timer does not see, such as the flushed path's memsets and
    per-iteration syncs, and host work. No batch is discarded here -- a
    target-time benchmark charges its own warmup and calibration batches
    through `charge_calibrated_bench` instead.
    """
    ref res = m.info_vec[len(m.info_vec) - 1].result
    var measured_ns = Int(res.duration("ns"))
    var warmup_ns = res.warmup_duration
    timer.add("bench_measured", measured_ns)
    timer.add("bench_warmup", warmup_ns)
    timer.add("bench_overhead", wall_ns - measured_ns - warmup_ns)


def bench_charged[
    FuncType: def(mut Bencher) raises -> None,
](
    mut m: Bench,
    mut timer: PhaseTimer,
    func: FuncType,
    bench_id: BenchId,
    measures: List[ThroughputMeasure] = {},
) raises:
    """Runs `m.bench_function` and charges its wall time to the bench phases.

    Leaves the timer's lap mark at the end of the call, so a following `lap`
    does not charge the benchmark twice.
    """
    var t0 = perf_counter_ns()
    m.bench_function(func, bench_id, measures)
    charge_last_bench(timer, m, perf_counter_ns() - t0)
    timer.reset()


def charge_calibrated_bench(
    mut timer: PhaseTimer,
    m: Bench,
    wall_ns: Int,
    warmup_ns: Int,
    calibrate_ns: Int,
):
    """Splits the wall time of a calibration-measured `bench_function` call.

    The harness timed the warmup and the calibration batch itself, so their
    device time comes in as arguments rather than out of the report, whose
    `fixed_iterations` run has no warmup of its own. `bench_measured` is the
    device time of the fixed batch the report averages; `bench_overhead` is
    the rest: the event syncs and host work.
    """
    ref res = m.info_vec[len(m.info_vec) - 1].result
    var measured_ns = Int(res.duration("ns"))
    timer.add("bench_measured", measured_ns)
    timer.add("bench_warmup", warmup_ns)
    timer.add("bench_calibrate", calibrate_ns)
    timer.add(
        "bench_overhead", wall_ns - measured_ns - warmup_ns - calibrate_ns
    )
