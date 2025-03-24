# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
"""Implements the benchmark module for runtime benchmarking.

You can import these APIs from the `benchmark` package. For example:

```mojo
import benchmark
from time import sleep
```

You can pass any `fn` as a parameter into `benchmark.run[...]()`, it will return
a `Report` where you can get the mean, duration, max, and more:

```mojo
fn sleeper():
    sleep(.01)

var report = benchmark.run[sleeper]()
print(report.mean())
```

```output
0.012256487394957985
```

You can print a full report:

```mojo
report.print()
```

```output
---------------------
Benchmark Report (s)
---------------------
Mean: 0.012265747899159664
Total: 1.459624
Iters: 119
Warmup Total: 0.025020000000000001
Fastest Mean: 0.0121578
Slowest Mean: 0.012321428571428572

```

Or all the batch runs:

```mojo
report.print_full()
```

```output
---------------------
Benchmark Report (s)
---------------------
Mean: 0.012368649122807017
Total: 1.410026
Iters: 114
Warmup Total: 0.023341000000000001
Fastest Mean: 0.012295586956521738
Slowest Mean: 0.012508099999999999

Batch: 1
Iterations: 20
Mean: 0.012508099999999999
Duration: 0.250162

Batch: 2
Iterations: 46
Mean: 0.012295586956521738
Duration: 0.56559700000000002

Batch: 3
Iterations: 48
Mean: 0.012380562499999999
Duration: 0.59426699999999999
```

If you want to use a different time unit you can bring in the Unit and pass
it in as an argument:

```mojo
from benchmark import Unit

report.print(Unit.ms)
```

```output
---------------------
Benchmark Report (ms)
---------------------
Mean: 0.012312411764705882
Total: 1.465177
Iters: 119
Warmup Total: 0.025010999999999999
Fastest Mean: 0.012015649999999999
Slowest Mean: 0.012421204081632654
```

The unit's are just aliases for `StringLiteral`, so you can for example:

```mojo
print(report.mean("ms"))
```

```output
12.199145299145298
```

Benchmark.run takes four arguments to change the behaviour, to set warmup
iterations to 5:

```mojo
r = benchmark.run[sleeper](5)
```

```output
0.012004808080808081
```

To set 1 warmup iteration, 2 max iterations, a min total time of 3 sec, and a
max total time of 4 s:

```mojo
r = benchmark.run[sleeper](1, 2, 3, 4)
```

Note that the min total time will take precedence over max iterations
"""

from collections import List
from time import time_function

from utils.numerics import max_finite, min_finite


# ===-----------------------------------------------------------------------===#
# Batch
# ===-----------------------------------------------------------------------===#
@value
@register_passable("trivial")
struct Batch(CollectionElement):
    """
    A batch of benchmarks, the benchmark.run() function works out how many
    iterations to run in each batch based the how long the previous iterations
    took.
    """

    var duration: Int
    """Total duration of batch stored as nanoseconds."""
    var iterations: Int
    """Total iterations in the batch."""
    var _is_significant: Bool
    """This batch contributes to the reporting of this benchmark."""

    fn __init__(out self, *, other: Self):
        """Explicitly construct a deep copy of the provided value.

        Args:
            other: The value to copy.
        """
        self = other

    fn _mark_as_significant(mut self):
        self._is_significant = True

    fn mean(self, unit: String = Unit.s) -> Float64:
        """
        Returns the average duration of the batch.

        Args:
            unit: The time unit to display for example: ns, ms, s (default `s`).

        Returns:
            The average duration of the batch.
        """
        return self.duration / self.iterations / Unit._divisor(unit)


# ===-----------------------------------------------------------------------===#
# Unit
# ===-----------------------------------------------------------------------===#
struct Unit:
    """Time Unit used by Benchmark Report."""

    alias ns = "ns"
    """Nanoseconds"""
    alias ms = "ms"
    """Milliseconds"""
    alias s = "s"
    """Seconds"""

    @staticmethod
    fn _divisor(unit: String) -> Int:
        if unit == Unit.ns:
            return 1
        elif unit == Unit.ms:
            return 1_000_000
        else:
            return 1_000_000_000


# ===-----------------------------------------------------------------------===#
# Report
# ===-----------------------------------------------------------------------===#
@value
struct Report(CollectionElement):
    """
    Contains the average execution time, iterations, min and max of each batch.
    """

    var warmup_duration: Int
    """The total duration it took to warmup."""
    var runs: List[Batch]
    """A `List` of benchmark runs."""

    fn __init__(out self):
        """
        Default initializer for the Report.

        Sets all values to 0
        """
        self.warmup_duration = 0
        self.runs = List[Batch]()

    fn __init__(out self, *, other: Self):
        """Explicitly construct a deep copy of the provided value.

        Args:
            other: The value to copy.
        """
        self = other

    fn __copyinit__(out self, existing: Self):
        """
        Creates a shallow copy (it doesn't copy the data).

        Args:
            existing: The `Report` to copy.
        """
        self.warmup_duration = existing.warmup_duration
        self.runs = existing.runs

    fn iters(self) -> Int:
        """
        The total benchmark iterations.

        Returns:
            The total benchmark iterations.
        """
        var iters = 0
        for i in range(len(self.runs)):
            if self.runs[i]._is_significant:
                iters += self.runs[i].iterations
        return iters

    fn duration(self, unit: String = Unit.s) -> Float64:
        """
        The total duration it took to run all benchmarks.

        Args:
            unit: The time unit to display for example: ns, ms, s (default `s`).

        Returns:
            The total duration it took to run all benchmarks.
        """
        var duration = 0
        for i in range(len(self.runs)):
            if self.runs[i]._is_significant:
                duration += self.runs[i].duration
        return duration / Unit._divisor(unit)

    fn mean(self, unit: String = Unit.s) -> Float64:
        """
        The average duration of all benchmark runs.

        Args:
            unit: The time unit to display for example: ns, ms, s (default `s`).

        Returns:
            The average duration of all benchmark runs.
        """
        return self.duration(unit) / self.iters()

    fn min(self, unit: String = Unit.s) -> Float64:
        """
        The batch of benchmarks that was the fastest to run.

        Args:
            unit: The time unit to display for example: ns, ms, s (default `s`).

        Returns:
            The fastest duration out of all batches.
        """
        if len(self.runs) == 0:
            return 0
        var min = max_finite[DType.float64]()
        for i in range(len(self.runs)):
            if self.runs[i]._is_significant and self.runs[i].mean(unit) < min:
                min = self.runs[i].mean(unit)
        return min

    fn max(self, unit: String = Unit.s) -> Float64:
        """
        The batch of benchmarks that was the slowest to run.

        Args:
            unit: The time unit to display for example: ns, ms, s (default `s`).

        Returns:
            The slowest duration out of all batches.
        """
        if len(self.runs) == 0:
            return 0
        var result = min_finite[DType.float64]()
        for i in range(len(self.runs)):
            if (
                self.runs[i]._is_significant
                and self.runs[i].mean(unit) > result
            ):
                result = self.runs[i].mean(unit)
        return result

    fn print(self, unit: String = Unit.s):
        """
        Prints out the shortened version of the report.

        Args:
            unit: The time unit to display for example: ns, ms, s (default `s`).
        """
        var divisor = Unit._divisor(unit)
        print(String("-") * 80)
        print("Benchmark Report (", end="")
        print(unit, end="")
        print(")")
        print(String("-") * 80)
        print("Mean:", self.mean(unit))
        print("Total:", self.duration(unit))
        print("Iters:", self.iters())
        print("Warmup Total:", self.warmup_duration / divisor)
        print("Fastest Mean:", self.min(unit))
        print("Slowest Mean:", self.max(unit))
        print()

    fn print_full(self, unit: String = Unit.s):
        """
        Prints out the full version of the report with each batch of benchmark
        runs.

        Args:
            unit: The time unit to display for example: ns, ms, s (default `s`).
        """

        var divisor = Unit._divisor(unit)
        self.print(unit)

        for i in range(len(self.runs)):
            print("Batch:", i + 1)
            print("Iterations:", self.runs[i].iterations)
            print(
                "Mean:",
                self.runs[i].mean(unit),
            )
            print("Duration:", self.runs[i].duration / divisor)
            print()


# ===-----------------------------------------------------------------------===#
# RunOptions
# ===-----------------------------------------------------------------------===#


@value
@register_passable("trivial")
struct _RunOptions[timing_fn: fn (num_iters: Int) raises capturing [_] -> Int]:
    var max_batch_size: Int
    var max_iters: Int
    var min_runtime_secs: Float64
    var max_runtime_secs: Float64
    var min_warmuptime_secs: Float64

    fn __init__(
        out self,
        max_batch_size: Int = 0,
        max_iters: Int = 1_000_000_000,
        min_runtime_secs: Float64 = 2,
        max_runtime_secs: Float64 = 60,
        min_warmuptime_secs: Float64 = 1,
    ):
        self.max_batch_size = max_batch_size
        self.max_iters = max_iters
        self.min_runtime_secs = min_runtime_secs
        self.max_runtime_secs = max_runtime_secs
        self.min_warmuptime_secs = min_warmuptime_secs


# ===-----------------------------------------------------------------------===#
# run
# ===-----------------------------------------------------------------------===#


@always_inline
fn run[
    func: fn () raises -> None
](
    max_iters: Int = 1_000_000_000,
    min_runtime_secs: Float64 = 2,
    max_runtime_secs: Float64 = 60,
    max_batch_size: Int = 0,
) raises -> Report:
    """Benchmarks the function passed in as a parameter.

    Benchmarking continues until 'min_time_ns' has elapsed and either
    `max_time_ns` OR `max_iters` is achieved.

    Parameters:
        func: The function to benchmark.

    Args:
        max_iters: Max number of iterations to run (default `1_000_000_000`).
        min_runtime_secs: Upper bound on benchmarking time in secs (default `2`).
        max_runtime_secs: Lower bound on benchmarking time in secs (default `60`).
        max_batch_size: The maximum number of iterations to perform per time
            measurement.

    Returns:
        Average execution time of func in ns.
    """

    @parameter
    @always_inline
    fn benchmark_fn(num_iters: Int) raises -> Int:
        @parameter
        @always_inline
        fn iter_fn() raises:
            for _ in range(num_iters):
                func()

        return time_function[iter_fn]()

    return _run_impl(
        _RunOptions[benchmark_fn](
            max_batch_size=max_batch_size,
            max_iters=max_iters,
            min_runtime_secs=min_runtime_secs,
            max_runtime_secs=max_runtime_secs,
        )
    )


@always_inline
fn run[
    func: fn () -> None
](
    max_iters: Int = 1_000_000_000,
    min_runtime_secs: Float64 = 2,
    max_runtime_secs: Float64 = 60,
    max_batch_size: Int = 0,
) raises -> Report:
    """Benchmarks the function passed in as a parameter.

    Benchmarking continues until 'min_time_ns' has elapsed and either
    `max_time_ns` OR `max_iters` is achieved.

    Parameters:
        func: The function to benchmark.

    Args:
        max_iters: Max number of iterations to run (default `1_000_000_000`).
        min_runtime_secs: Upper bound on benchmarking time in secs (default `2`).
        max_runtime_secs: Lower bound on benchmarking time in secs (default `60`).
        max_batch_size: The maximum number of iterations to perform per time
            measurement.

    Returns:
        Average execution time of func in ns.
    """

    @parameter
    fn raising_func() raises:
        func()

    return run[raising_func](
        max_iters, min_runtime_secs, max_runtime_secs, max_batch_size
    )


@always_inline
fn run[
    func: fn () raises capturing [_] -> None
](
    max_iters: Int = 1_000_000_000,
    min_runtime_secs: Float64 = 2,
    max_runtime_secs: Float64 = 60,
    max_batch_size: Int = 0,
) raises -> Report:
    """Benchmarks the function passed in as a parameter.

    Benchmarking continues until 'min_time_ns' has elapsed and either
    `max_time_ns` OR `max_iters` is achieved.

    Parameters:
        func: The function to benchmark.

    Args:
        max_iters: Max number of iterations to run (default `1_000_000_000`).
        min_runtime_secs: Upper bound on benchmarking time in secs (default `2`).
        max_runtime_secs: Lower bound on benchmarking time in secs (default `60`).
        max_batch_size: The maximum number of iterations to perform per time
            measurement.

    Returns:
        Average execution time of func in ns.
    """

    @parameter
    @always_inline
    fn benchmark_fn(num_iters: Int) raises -> Int:
        @parameter
        @always_inline
        fn iter_fn() raises:
            for _ in range(num_iters):
                func()

        return time_function[iter_fn]()

    return _run_impl(
        _RunOptions[benchmark_fn](
            max_batch_size=max_batch_size,
            max_iters=max_iters,
            min_runtime_secs=min_runtime_secs,
            max_runtime_secs=max_runtime_secs,
        )
    )


@always_inline
fn run[
    func: fn () capturing [_] -> None
](
    max_iters: Int = 1_000_000_000,
    min_runtime_secs: Float64 = 2,
    max_runtime_secs: Float64 = 60,
    max_batch_size: Int = 0,
) raises -> Report:
    """Benchmarks the function passed in as a parameter.

    Benchmarking continues until 'min_time_ns' has elapsed and either
    `max_time_ns` OR `max_iters` is achieved.

    Parameters:
        func: The function to benchmark.

    Args:
        max_iters: Max number of iterations to run (default `1_000_000_000`).
        min_runtime_secs: Upper bound on benchmarking time in secs (default `2`).
        max_runtime_secs: Lower bound on benchmarking time in secs (default `60`).
        max_batch_size: The maximum number of iterations to perform per time
            measurement.

    Returns:
        Average execution time of func in ns.
    """

    @parameter
    fn raising_func() raises:
        func()

    return run[raising_func](
        max_iters, min_runtime_secs, max_runtime_secs, max_batch_size
    )


@always_inline
fn _run_impl(opts: _RunOptions) raises -> Report:
    var report = Report()

    var prev_dur: Int = 0
    var prev_iters: Int = 0
    var min_warmup_time_ns = Int(opts.min_warmuptime_secs * 1_000_000_000)
    if min_warmup_time_ns > 0:
        # Make sure to warm up the function and use one iteration to compute
        # the previous duration.
        var time_elapsed: Int = 0
        while time_elapsed < min_warmup_time_ns:
            prev_dur = opts.timing_fn(1)
            # If the function is too fast, we need to make sure we don't have a
            # duration of 0 which will cause an endless loop.
            if prev_dur == 0:
                prev_dur = 1_000
            time_elapsed += prev_dur
        prev_iters = 1
        report.warmup_duration = prev_dur
    else:
        report.warmup_duration = 0

    var total_iters: Int = 0
    var time_elapsed: Int = 0
    var min_time_ns = Int(opts.min_runtime_secs * 1_000_000_000)
    var max_time_ns = Int(opts.max_runtime_secs * 1_000_000_000)

    while time_elapsed < min_time_ns:
        if time_elapsed > max_time_ns or total_iters > opts.max_iters:
            break

        var n = Float64(opts.max_batch_size)
        if opts.max_batch_size == 0:
            # We now count the next batchSize. A user might run the benchmark
            # with no warmup phase, so we need to make sure the divisor is not
            # zero.
            # Compute the next batch size.
            if prev_dur > 0:
                n = 1.2 * min_time_ns * prev_iters / Float64(prev_dur)
            # We should not grow too fast, so we cap it to only 10x the growth
            # from the prior iteration. Fast growth can happen when the function
            # is too fast.
            n = min(n, 10 * prev_iters)
            # We have to increase the batchSize each time. So, we make sure we
            # advance the number of iterations regardless of the prior logic.
            n = max(n, prev_iters + 1)
            # The batch size should not be larger than 1.0e9.
            n = min(n, 1.0e9)

        prev_dur = opts.timing_fn(Int(n))
        prev_iters = Int(n)
        report.runs.append(Batch(prev_dur, prev_iters, False))
        total_iters += prev_iters
        time_elapsed += prev_dur

    for i in range(len(report.runs)):
        if _is_significant_measurement(
            i, report.runs[i], len(report.runs), opts
        ):
            report.runs[i]._mark_as_significant()
    return report


fn _is_significant_measurement(
    idx: Int, batch: Batch, num_batches: Int, opts: _RunOptions
) -> Bool:
    # The measurement number of iteration is the same as the requested
    # maxBatchSize and the measurement duration execeeded the requested min
    # runtime.
    if (
        opts.max_batch_size
        and batch.iterations >= opts.max_batch_size
        and Float64(batch.duration) >= opts.min_runtime_secs
    ):
        return True

    # This measument occured in the last 10% of the run.
    if Float64(idx + 1) >= 0.9 * num_batches:
        return True

    # Otherwise the result is not statically significant.
    return False
