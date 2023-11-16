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

let report = benchmark.run[sleeper]()
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
Warmup Mean: 0.01251
Warmup Total: 0.025020000000000001
Warmup Iters: 2
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
Warmup Mean: 0.0116705
Warmup Total: 0.023341000000000001
Warmup Iters: 2
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
Warmup Mean: 0.012505499999999999
Warmup Total: 0.025010999999999999
Warmup Iters: 2
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

from math import max, min
from time import time_function

from memory.unsafe import DTypePointer, Pointer
from utils.vector import DynamicVector


# ===----------------------------------------------------------------------===#
# Batch
# ===----------------------------------------------------------------------===#
@value
@register_passable("trivial")
struct Batch:
    """
    A batch of benchmarks, the benchmark.run() function works out how many
    iterations to run in each batch based the how long the previous iterations
    took.
    """

    var duration: Int
    """Total duration of batch stored as nanoseconds."""
    var iterations: Int
    """Total iterations in the batch."""

    fn mean(self, unit: String = Unit.s) -> Float64:
        """
        Returns the average duration of the batch.

        Args:
            unit: The time unit to display for example: ns, ms, s (default `s`).

        Returns:
            The average duration of the batch.
        """
        return self.duration / self.iterations / _divisor(unit)


# ===----------------------------------------------------------------------===#
# Unit
# ===----------------------------------------------------------------------===#
struct Unit:
    """Time Unit used by Benchmark Report."""

    alias ns = "ns"
    """Nanoseconds"""
    alias ms = "ms"
    """Milliseconds"""
    alias s = "s"
    """Seconds"""


fn _divisor(unit: String) -> Int:
    if unit == Unit.ns:
        return 1
    elif unit == Unit.ms:
        return 1_000_000
    else:
        return 1_000_000_000


# ===----------------------------------------------------------------------===#
# Report
# ===----------------------------------------------------------------------===#
struct Report:
    """
    Contains the average execution time, iterations, min and max of each batch.
    """

    var warmup_iters: Int
    """The total warmup iterations."""
    var warmup_duration: Int
    """The total duration it took to warmup."""
    var runs: DynamicVector[Batch]
    """A `DynamicVector` of benchmark runs."""

    fn __init__(inout self):
        """
        Default initializer for the Report.

        Sets all values to 0
        """
        self.warmup_iters = 0
        self.warmup_duration = 0
        self.runs = DynamicVector[Batch]()

    fn __del__(owned self):
        """Delets the Report object."""
        self.runs._del_old()

    fn __copyinit__(inout self, existing: Self):
        """
        Creates a shallow copy (it doesn't copy the data).

        Args:
            existing: The `Report` to copy.
        """
        self.warmup_iters = existing.warmup_iters
        self.warmup_duration = existing.warmup_duration
        self.runs = existing.runs.deepcopy()

    fn iters(self) -> Int:
        """
        The total benchmark iterations.

        Returns:
            The total benchmark iterations.
        """
        var iters = 0
        for i in range(len(self.runs)):
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
            duration += self.runs[i].duration
        return duration / _divisor(unit)

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
        var min = self.runs[0].mean(unit)
        for i in range(1, len(self.runs)):
            if self.runs[i].mean(unit) < min:
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
        var result: Float64 = 0.0
        for i in range(len(self.runs)):
            if self.runs[i].mean(unit) > result:
                result = self.runs[i].mean(unit)
        return result

    fn print(self, unit: String = Unit.s):
        """
        Prints out the shortened version of the report.

        Args:
            unit: The time unit to display for example: ns, ms, s (default `s`).
        """
        let divisor = _divisor(unit)
        print("---------------------")
        print_no_newline("Benchmark Report (")
        print_no_newline(unit)
        print(")")
        print("---------------------")
        print("Mean:", self.mean(unit))
        print("Total:", self.duration(unit))
        print("Iters:", self.iters())
        print(
            "Warmup Mean:",
            self.warmup_duration / self.warmup_iters / divisor,
        )
        print("Warmup Total:", self.warmup_duration / divisor)
        print("Warmup Iters:", self.warmup_iters)
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

        let divisor = _divisor(unit)
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


# ===----------------------------------------------------------------------===#
# RunOptions
# ===----------------------------------------------------------------------===#


@value
@register_passable("trivial")
struct _RunOptions[timing_fn: fn (num_iters: Int) capturing -> Int]:
    var num_warmup: Int
    var max_iters: Int
    var min_time_secs: Float64
    var max_time_secs: Float64

    fn __init__() -> Self:
        return Self {
            num_warmup: 2,
            max_iters: 100_000,
            min_time_secs: 0.5,
            max_time_secs: 1,
        }

    fn __init__(
        num_warmup: Int,
        max_iters: Int,
        min_time_secs: Float64,
        max_time_secs: Float64,
    ) -> Self:
        return Self {
            num_warmup: num_warmup,
            max_iters: max_iters,
            min_time_secs: min_time_secs,
            max_time_secs: max_time_secs,
        }


# ===----------------------------------------------------------------------===#
# run
# ===----------------------------------------------------------------------===#


@always_inline
fn run[
    func: fn () -> None
](
    num_warmup: Int = 2,
    max_iters: Int = 100_000,
    min_time_secs: Float64 = 0.5,
    max_time_secs: Float64 = 1,
) -> Report:
    """Benchmarks the function passed in as a parameter.

    Benchmarking continues until 'min_time_ns' has elapsed and either
    `max_time_ns` OR `max_iters` is achieved.

    Parameters:
        func: The function to benchmark.

    Args:
        num_warmup: Number of warmup iterations to run before starting
            benchmarking (default 2).
        max_iters: Max number of iterations to run (default `100_000`).
        min_time_secs: Upper bound on benchmarking time in secs (default `0.5`).
        max_time_secs: Lower bound on benchmarking time in secs (default `1`).

    Returns:
        Average execution time of func in ns.
    """

    @parameter
    @always_inline
    fn benchmark_fn(num_iters: Int) -> Int:
        @parameter
        @always_inline
        fn iter_fn():
            for _ in range(num_iters):
                func()

        return time_function[iter_fn]()

    return _run_impl(
        _RunOptions[benchmark_fn](
            num_warmup, max_iters, min_time_secs, max_time_secs
        )
    )


@always_inline
fn run[
    func: fn () capturing -> None
](
    num_warmup: Int = 2,
    max_iters: Int = 100_000,
    min_time_secs: Float64 = 0.5,
    max_time_secs: Float64 = 1,
) -> Report:
    """Benchmarks the function passed in as a parameter.

    Benchmarking continues until 'min_time_ns' has elapsed and either
    `max_time_ns` OR `max_iters` is achieved.

    Parameters:
        func: The function to benchmark.

    Args:
        num_warmup: Number of warmup iterations to run before starting
            benchmarking (default 2).
        max_iters: Max number of iterations to run (default `100_000`).
        min_time_secs: Upper bound on benchmarking time in secs (default `0.5`).
        max_time_secs: Lower bound on benchmarking time in secs (default `1`).

    Returns:
        Average execution time of func in ns.
    """

    @parameter
    @always_inline
    fn benchmark_fn(num_iters: Int) -> Int:
        @parameter
        @always_inline
        fn iter_fn():
            for _ in range(num_iters):
                func()

        return time_function[iter_fn]()

    return _run_impl(
        _RunOptions[benchmark_fn](
            num_warmup, max_iters, min_time_secs, max_time_secs
        )
    )


@always_inline
fn _run_impl(opts: _RunOptions) -> Report:
    var report = Report()

    var prev_iters = opts.num_warmup
    var prev_dur = opts.timing_fn(opts.num_warmup) if opts.num_warmup > 0 else 0
    report.warmup_iters = prev_iters
    report.warmup_duration = prev_dur
    var total_iters: Int = 0
    var time_elapsed: Int = 0
    let min_time_ns = (opts.min_time_secs * 1_000_000_000).to_int()
    let max_time_ns = (opts.max_time_secs * 1_000_000_000).to_int()

    while time_elapsed < max_time_ns:
        if total_iters > opts.max_iters and time_elapsed > min_time_ns:
            break
        prev_dur = max(1, prev_dur)  # avoid dividing by 0
        # Order of operations matters.
        # For very fast benchmarks, prev_iterations ~= prev_duration.
        # If you divide first, you get 0 or 1,
        # which can hide an order of magnitude in execution time.
        # So multiply first, then divide.
        var n = min_time_ns * prev_iters // prev_dur
        n += n // 5
        # Don't grow too fast in case we had timing errors previously.
        n = min(n, 10 * prev_iters)
        # Be sure to run at least one more than last time.
        n = max(n, prev_iters + 1)
        # Don't run more than 1e9 times.
        # (This also keeps n in int range on 32 bit platforms.)
        n = min(n, 1_000_000_000)

        prev_dur = opts.timing_fn(n)
        prev_iters = n
        report.runs.push_back(Batch(prev_dur, prev_iters))
        total_iters += prev_iters
        time_elapsed += prev_dur

    return report


# ===----------------------------------------------------------------------===#
# clobber_memory
# ===----------------------------------------------------------------------===#


@always_inline
fn clobber_memory():
    """Forces all pending memory writes to be flushed to memory.

    This ensures that the compiler does not optimize away memory writes if it
    deems them to be not neccessary. In effect, this operation acts a barrier
    to memory reads and writes.
    """

    # This opereration corresponds to  atomic_signal_fence(memory_order_acq_rel)
    # in C++.
    __mlir_op.`pop.fence`[
        _type=None,
        syncscope = "singlethread".value,
        ordering = __mlir_attr.`#pop<atomic_ordering acq_rel>`,
    ]()


# ===----------------------------------------------------------------------===#
# keep
# ===----------------------------------------------------------------------===#


@always_inline
fn keep(val: Bool):
    """Provides a hint to the compiler to not optimize the variable use away.

    This is useful in benchmarking to avoid the compiler not deleting the
    code to be benchmarked because the variable is not used in a side-effecting
    manner.

    Args:
      val: The value to not optimize away.
    """
    keep(UInt8(val))


@always_inline
fn keep(val: Int):
    """Provides a hint to the compiler to not optimize the variable use away.

    This is useful in benchmarking to avoid the compiler not deleting the
    code to be benchmarked because the variable is not used in a side-effecting
    manner.

    Args:
      val: The value to not optimize away.
    """
    keep(SIMD[DType.index, 1](val))


@always_inline
fn keep[type: DType, simd_width: Int](val: SIMD[type, simd_width]):
    """Provides a hint to the compiler to not optimize the variable use away.

    This is useful in benchmarking to avoid the compiler not deleting the
    code to be benchmarked because the variable is not used in a side-effecting
    manner.

    Parameters:
      type: The `dtype` of the input and output SIMD vector.
      simd_width: The width of the input and output SIMD vector.

    Args:
      val: The value to not optimize away.
    """
    var tmp = val
    let tmp_ptr = Pointer.address_of(tmp)

    @parameter
    if sizeof[type]() <= sizeof[Pointer[SIMD[type, simd_width]].pointer_type]():
        __mlir_op.`pop.inline_asm`[
            _type=None,
            assembly = "".value,
            constraints = "+m,r,~{memory}".value,
            hasSideEffects = __mlir_attr.unit,
        ](tmp_ptr, val)
    else:
        __mlir_op.`pop.inline_asm`[
            _type=None,
            assembly = "".value,
            constraints = "+m,~{memory}".value,
            hasSideEffects = __mlir_attr.unit,
        ](tmp_ptr, tmp_ptr)


@always_inline
fn keep[type: DType](val: DTypePointer[type]):
    """Provides a hint to the compiler to not optimize the variable use away.

    This is useful in benchmarking to avoid the compiler not deleting the
    code to be benchmarked because the variable is not used in a side-effecting
    manner.

    Parameters:
      type: The type of the input.

    Args:
      val: The value to not optimize away.
    """
    keep(Pointer(val.address))


@always_inline
fn keep[type: AnyRegType](val: Pointer[type]):
    """Provides a hint to the compiler to not optimize the variable use away.

    This is useful in benchmarking to avoid the compiler not deleting the
    code to be benchmarked because the variable is not used in a side-effecting
    manner.

    Parameters:
      type: The type of the input.

    Args:
      val: The value to not optimize away.
    """
    var tmp = val
    let tmp_ptr = Pointer.address_of(tmp)
    __mlir_op.`pop.inline_asm`[
        _type=None,
        assembly = "".value,
        constraints = "r,~{memory}".value,
        hasSideEffects = __mlir_attr.unit,
    ](tmp_ptr)


@always_inline
fn keep[type: AnyRegType](inout val: type):
    """Provides a hint to the compiler to not optimize the variable use away.

    This is useful in benchmarking to avoid the compiler not deleting the
    code to be benchmarked because the variable is not used in a side-effecting
    manner.

    Parameters:
      type: The type of the input.

    Args:
      val: The value to not optimize away.
    """
    var tmp = val
    let tmp_ptr = Pointer.address_of(tmp)
    __mlir_op.`pop.inline_asm`[
        _type=None,
        assembly = "".value,
        constraints = "r,~{memory}".value,
        hasSideEffects = __mlir_attr.unit,
    ](tmp_ptr)
