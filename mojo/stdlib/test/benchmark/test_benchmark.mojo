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

from std.time import sleep, time_function

from std.benchmark import Batch, Report, Unit, clobber_memory, keep, run
from std.benchmark.bencher import BenchMetric, Format, ThroughputMeasure
from std.testing import TestSuite, assert_equal, assert_true


def test_stopping_criteria() raises:
    # Stop when min_runtime_secs has elapsed and either max_runtime_secs or max_iters
    # is reached

    @always_inline
    @parameter
    fn time_me():
        sleep(0.002)
        clobber_memory()
        return

    var lb = 0.02  # 20ms
    var ub = 0.1  # 100ms

    # stop after ub (max_runtime_secs)
    var max_iters_1 = 1000_000_000

    @__copy_capture(lb, ub)
    @parameter
    fn timer() raises:
        var report = run[func4=time_me](
            max_iters=max_iters_1, min_runtime_secs=lb, max_runtime_secs=ub
        )
        assert_true(report.mean() > 0)
        assert_true(report.iters() != max_iters_1)

    var t1 = time_function[timer]()
    assert_true(Float64(t1) / 1e9 >= ub)

    # stop after lb (min_runtime_secs)
    var ub_big = 1  # 1s
    var max_iters_2 = 1

    @__copy_capture(ub_big, lb)
    @parameter
    fn timer2() raises:
        var report = run[func4=time_me](
            max_iters=max_iters_2,
            min_runtime_secs=lb,
            max_runtime_secs=Float64(ub_big),
        )
        assert_true(report.mean() > 0)
        assert_true(report.iters() >= max_iters_2)

    var t2 = time_function[timer2]()

    assert_true(
        Float64(t2) / 1e9 >= lb and Float64(t2) / 1e9 <= Float64(ub_big)
    )

    # stop on or before max_iters
    var max_iters_3 = 3

    @__copy_capture(ub_big)
    @parameter
    fn timer3() raises:
        var report = run[func4=time_me](
            max_iters=max_iters_3,
            min_runtime_secs=0,
            max_runtime_secs=Float64(ub_big),
        )
        assert_true(report.mean() > 0)
        assert_true(report.iters() <= max_iters_3)

    var t3 = time_function[timer3]()

    assert_true(Float64(t3) / 1e9 <= Float64(ub_big))


struct SomeStruct(TrivialRegisterPassable):
    var x: Int
    var y: Int

    @always_inline
    fn __init__(out self):
        self.x = 5
        self.y = 4


struct SomeTrivialStruct(TrivialRegisterPassable):
    var x: Int
    var y: Int

    @always_inline
    fn __init__(out self):
        self.x = 3
        self.y = 5


# There is nothing to test here other than the code executes and does not crash.
def test_keep() raises:
    keep(False)
    keep(33)

    var val = SIMD[DType.int, 4](1, 2, 3, 4)
    keep(val)

    var ptr = UnsafePointer(to=val)
    keep(ptr)

    var s0 = SomeStruct()
    keep(s0)

    var s1 = SomeTrivialStruct()
    keep(s1)


fn sleeper():
    sleep(0.001)


def test_non_capturing() raises:
    var report = run[func2=sleeper](min_runtime_secs=0.1, max_runtime_secs=0.3)
    assert_true(report.mean() > 0.001)


def test_change_units() raises:
    var report = run[func2=sleeper](min_runtime_secs=0.1, max_runtime_secs=0.3)
    assert_true(report.mean("ms") > 1.0)
    assert_true(report.mean("us") > 1_000)
    assert_true(report.mean("ns") > 1_000_000.0)


def test_report() raises:
    var report = run[func2=sleeper](min_runtime_secs=0.1, max_runtime_secs=0.3)

    var report_string = report.as_string()
    assert_true("Benchmark Report (s)" in report_string)
    assert_true("Mean: " in report_string)
    assert_true("Total: " in report_string)
    assert_true("Iters: " in report_string)
    assert_true("Warmup Total: " in report_string)
    assert_true("Fastest Mean: " in report_string)
    assert_true("Slowest Mean: " in report_string)


def test_bench_metric_write_repr_to() raises:
    var s = String()
    BenchMetric.elements.write_repr_to(s)
    assert_true(s.startswith("BenchMetric("))
    assert_true("code=0" in s)
    assert_true("name=" in s)
    assert_true("unit=" in s)


def test_format_write_repr_to() raises:
    var s = String()
    Format.csv.write_repr_to(s)
    assert_equal(s, "Format('csv')")

    s = String()
    Format.table.write_repr_to(s)
    assert_equal(s, "Format('table')")


def test_throughput_measure_write_repr_to() raises:
    var m = ThroughputMeasure(BenchMetric.elements, 1024)
    var s = String()
    m.write_repr_to(s)
    assert_true(s.startswith("ThroughputMeasure("))
    assert_true("metric=" in s)
    assert_true("value=1024" in s)


fn _make_report(*durations_ns: Int) -> Report:
    """Build a Report with one significant batch per duration (1 iteration each).
    """
    var runs = List[Batch]()
    for d in durations_ns:
        runs.append(Batch(duration=d, iterations=1, _is_significant=True))
    return Report(warmup_duration=0, runs=runs^)


def test_percentile_empty_report() raises:
    """Returns 0 for a report with no runs."""
    var report = Report()
    assert_equal(report.percentile(50, Unit.ns), 0.0)


def test_percentile_single_batch() raises:
    """Any percentile on a single batch returns that batch's mean."""
    var report = _make_report(42)
    assert_equal(report.percentile(0, Unit.ns), 42.0)
    assert_equal(report.percentile(50, Unit.ns), 42.0)
    assert_equal(report.percentile(100, Unit.ns), 42.0)


def test_percentile_odd_count() raises:
    """Median (p50) of [1,2,3,4,5] ns is the middle value (3 ns)."""
    var report = _make_report(3, 1, 5, 2, 4)  # unsorted to verify sort
    assert_equal(report.percentile(0, Unit.ns), 1.0)
    assert_equal(report.percentile(50, Unit.ns), 3.0)
    assert_equal(report.percentile(100, Unit.ns), 5.0)


def test_percentile_even_count() raises:
    """Median (p50) of [1,2,3,4] ns is the average of the two middle values (2.5 ns).
    """
    var report = _make_report(4, 1, 3, 2)
    assert_equal(report.percentile(0, Unit.ns), 1.0)
    assert_equal(report.percentile(50, Unit.ns), 2.5)
    assert_equal(report.percentile(100, Unit.ns), 4.0)


def test_percentile_p25_p75() raises:
    """Quartiles (p25, p75) of [1,2,3,4,5] ns use linear interpolation."""
    var report = _make_report(1, 2, 3, 4, 5)
    # index = 0.25 * 4 = 1.0 → values[1] = 2, frac = 0 → 2.0
    assert_equal(report.percentile(25, Unit.ns), 2.0)
    # index = 0.75 * 4 = 3.0 → values[3] = 4, frac = 0 → 4.0
    assert_equal(report.percentile(75, Unit.ns), 4.0)


def test_percentile_clamps_bounds() raises:
    """Clamps: p<=0 returns min, p>=100 returns max."""
    var report = _make_report(10, 20, 30)
    assert_equal(report.percentile(-1, Unit.ns), 10.0)
    assert_equal(report.percentile(101, Unit.ns), 30.0)


def test_percentile_ignores_non_significant_batches() raises:
    """Non-significant batches are excluded from percentile computation."""
    var runs: List[Batch] = [
        Batch(duration=1, iterations=1, _is_significant=True),
        Batch(duration=1000, iterations=1, _is_significant=False),
        Batch(duration=3, iterations=1, _is_significant=True),
        Batch(duration=2, iterations=1, _is_significant=True),
    ]
    var report = Report(warmup_duration=0, runs=runs^)
    assert_equal(report.percentile(100, Unit.ns), 3.0)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
