# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
"""
Implements the benchmark module for runtime benchmarking.

You can import these APIs from the `benchmark` package. For example:

### Import

```mojo
import benchmark
from time import sleep
```

### Usage

You can pass any `fn` as a parameter into `benchmark.run[...]()`, it will return a
`Report` where you can get the mean, duration, max, and more:

```mojo
fn sleeper():
    sleep(.01)

let report = benchmark.run[sleeper]()
print(report.mean())
```

```output
0.012451871794871795
```

You can print a full report:

```mojo
report.print()
```

```output
---------------------
Benchmark Report (s)
---------------------
Mean: 0.012314264957264957
Total: 1.440769
Iters: 117
Warmup Mean: 0.0119335
Warmup Total: 0.023866999999999999
Warmup Iters: 2
Fastest Mean: 0.012227958333333334
Slowest Mean: 0.012442699999999999

```

Or all the batch runs:

```mojo
report.print_full()
```

```output
---------------------
Benchmark Report (ms)
---------------------
Mean: 12.397538461538462
Total: 1450.5119999999999
Iters: 117
Warmup Mean: 11.715
Warmup Total: 23.43
Warmup Iters: 2
Fastest Mean: 12.0754375
Slowest Mean: 12.760265306122449

Batch: 1
Iterations: 20
Mean: 12.2819
Duration: 245.63800000000001

Batch: 2
Iterations: 48
Mean: 12.0754375
Duration: 579.62099999999998

Batch: 3
Iterations: 49
Mean: 12.760265306122449
Duration: 625.25300000000004

```

If you want to use a different time unit you can bring in the Unit and pass
it in as a parameter:

```mojo
from benchmark import Unit

report.print[Unit.ms]()
```

```output
---------------------
Benchmark Report (ms)
---------------------
Mean: 12.474991228070174
Total: 1422.1489999999999
Iters: 114
Warmup Mean: 11.976000000000001
Warmup Total: 23.952000000000002
Warmup Iters: 2
Fastest Mean: 12.297478260869564
Slowest Mean: 12.6313125

```

The unit's are just aliases for `StringLiteral`, so you can for example:

```mojo
print(report.mean["ms"]())
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

    fn mean[unit: StringLiteral = "s"](self) -> Float64:
        """
        Returns the average duration of the batch.

        Parameters:
            unit: The time unit to display for example: ns, ms, s (default `s`).

        Returns:
            The average duration of the batch.
        """
        return self.duration / self.iterations / _divisor[unit]()


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


fn _divisor[unit: StringLiteral = "s"]() -> Int:
    constrained[
        unit == "s" or unit == "ms" or unit == "ns",
        "Unit must be `s`, `ns`, or `ms`",
    ]()
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

    fn __copyinit__(inout self, existing: Self):
        """
        Creates a shallow copy (it doesn't copy the data).

        Args:
            existing: The `Report` to copy.
        """
        self.warmup_iters = existing.warmup_iters
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
            iters += self.runs[i].iterations
        return iters

    fn duration[unit: StringLiteral = Unit.s](self) -> Float64:
        """
        The total duration it took to run all benchmarks.

        Parameters:
            unit: The time unit to display for example: ns, ms, s (default `s`).

        Returns:
            The total duration it took to run all benchmarks.
        """
        var duration = 0
        for i in range(len(self.runs)):
            duration += self.runs[i].duration
        return duration / _divisor[unit]()

    fn mean[unit: StringLiteral = Unit.s](self) -> Float64:
        """
        The average duration of all benchmark runs.

        Parameters:
            unit: The time unit to display for example: ns, ms, s (default `s`).

        Returns:
            The average duration of all benchmark runs.
        """

        return self.duration[unit]() / self.iters()

    fn min[unit: StringLiteral = Unit.s](self) -> Float64:
        """
        The batch of benchmarks that was the fastest to run.

        Parameters:
            unit: The time unit to display for example: ns, ms, s (default `s`).

        Returns:
            The fastest duration out of all batches.
        """
        if len(self.runs) == 0:
            return 0
        var min = self.runs[0].mean[unit]()
        for i in range(1, len(self.runs)):
            if self.runs[i].mean[unit]() < min:
                min = self.runs[i].mean[unit]()
        return min

    fn max[unit: StringLiteral = Unit.s](self) -> Float64:
        """
        The batch of benchmarks that was the slowest to run.

        Parameters:
            unit: The time unit to display for example: ns, ms, s (default `s`).

        Returns:
            The slowest duration out of all batches.
        """
        var result: Float64 = 0.0
        for i in range(len(self.runs)):
            if self.runs[i].mean[unit]() > result:
                result = self.runs[i].mean[unit]()
        return result

    fn print[unit: StringLiteral = "s"](self):
        """
        Prints out the shortened version of the report.

        Parameters:
            unit: The time unit to display for example: ns, ms, s (default `s`).
        """
        alias divisor = _divisor[unit]()
        print("---------------------")
        print_no_newline("Benchmark Report (")
        print_no_newline(unit)
        print(")")
        print("---------------------")
        print("Mean:", self.mean[unit]())
        print("Total:", self.duration[unit]())
        print("Iters:", self.iters())
        print(
            "Warmup Mean:",
            self.warmup_duration / self.warmup_iters / divisor,
        )
        print("Warmup Total:", self.warmup_duration / divisor)
        print("Warmup Iters:", self.warmup_iters)
        print("Fastest Mean:", self.min[unit]())
        print("Slowest Mean:", self.max[unit]())
        print()

    fn print_full[unit: StringLiteral = "ms"](self):
        """
        Prints out the full version of the report with each batch of benchmark
        runs.

        Parameters:
            unit: The time unit to display for example: ns, ms, s (default `s`).
        """

        alias divisor = _divisor[unit]()
        self.print[unit]()

        for i in range(len(self.runs)):
            print("Batch:", i + 1)
            print("Iterations:", self.runs[i].iterations)
            print(
                "Mean:",
                self.runs[i].mean[unit](),
            )
            print("Duration:", self.runs[i].duration / divisor)
            print()


# ===----------------------------------------------------------------------===#
# benchmark
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
    fn benchmark_fn():
        func()

    return _run_impl[benchmark_fn](
        num_warmup, max_iters, min_time_secs, max_time_secs
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
    return _run_impl[func](num_warmup, max_iters, min_time_secs, max_time_secs)


@always_inline
fn _run_impl[
    func: fn () capturing -> None
](
    num_warmup: Int = 2,
    max_iters: Int = 100_000,
    min_time_secs: Float64 = 0.5,
    max_time_secs: Float64 = 1,
) -> Report:
    var report = Report()

    # run for specified number of warmup iterations
    @parameter
    @always_inline
    fn warmup_fn():
        for _ in range(num_warmup):
            func()

    var prev_iters = num_warmup
    var prev_dur = time_function[warmup_fn]() if num_warmup > 0 else 0
    report.warmup_iters = prev_iters
    report.warmup_duration = prev_dur
    var total_iters: Int = 0
    var time_elapsed: Int = 0
    let min_time_ns = (min_time_secs * 1_000_000_000).to_int()
    let max_time_ns = (max_time_secs * 1_000_000_000).to_int()

    while time_elapsed < max_time_ns:
        if total_iters > max_iters and time_elapsed > min_time_ns:
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

        @parameter
        @always_inline
        fn benchmark_fn():
            for _ in range(n):
                func()

        prev_dur = time_function[benchmark_fn]()
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
fn keep[type: AnyType](val: Pointer[type]):
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
fn keep[type: AnyType](inout val: type):
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
