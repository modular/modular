# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
"""Implements the Benchmark class for runtime benchmarking.

You can import these APIs from the `benchmark` package. For example:

```mojo
from benchmark import Benchmark
```
"""

from math import max, min
from time import time_function

from memory.unsafe import DTypePointer, Pointer

# ===----------------------------------------------------------------------===#
# Benchmark
# ===----------------------------------------------------------------------===#


@value
struct Benchmark:
    """A benchmark harness.

    The class allows to benchmark a given function (passed as a parameter) and
    configure various benchmarking parameters, such as number of warmup
    iterations, maximum number of iterations, minimum and maximum elapsed time.
    """

    var num_warmup: Int
    """The number of warmup iterations to perform before the main benchmark
    loop."""
    var max_iters: Int
    """The maximum number of iterations to perform during the main benchmark
    loop."""
    var min_time_ns: Int
    """The minimum time (in ns) to spend within the main benchmark loop."""
    var max_time_ns: Int
    """The maximum time (in ns) to spend within the main benchmark loop."""

    fn __init__(
        inout self,
        num_warmup: Int = 2,
        max_iters: Int = 100_000,
        min_time_ns: Int = 500_000_000,  # 500ms
        max_time_ns: Int = 1000_000_000,  # 1s
    ):
        """Constructs a new benchmark object.

        Given a function the benchmark object will benchmark it until
        min_time_ns has elapsed and either max_time_ns OR max_iters is hit.

        Args:
            num_warmup: Number of warmup iterations to run before starting
              benchmarking (default 2).
            max_iters: Max number of iterations to run (default 100_000).
            min_time_ns: Upper bound on benchmarking time in ns (default 500ms).
            max_time_ns: Lower bound on benchmarking time in ns (default 1s).
        """
        self.num_warmup = num_warmup
        self.max_iters = max_iters
        self.min_time_ns = min_time_ns
        self.max_time_ns = max_time_ns

    @always_inline
    fn run[func: fn () capturing -> None](self) -> Int:
        """Benchmarks the given function.

        Benchmarking continues until min_time_ns has elapsed and either
        `max_time_ns` or `max_iters` is achieved.

        Parameters:
            func: The function to benchmark.

        Returns:
            Average execution time of func in ns.
        """

        # run for specified number of warmup iterations
        @parameter
        @always_inline
        fn warmup_fn():
            for _ in range(self.num_warmup):
                func()

        var prev_iters = self.num_warmup
        var prev_dur = time_function[warmup_fn]() if self.num_warmup > 0 else 0
        var total_iters: Int = 0
        var time_elapsed: Int = 0

        while time_elapsed < self.max_time_ns:
            if total_iters > self.max_iters and time_elapsed > self.min_time_ns:
                break
            prev_dur = max(1, prev_dur)  # avoid dividing by 0
            # Order of operations matters.
            # For very fast benchmarks, prev_iterations ~= prev_duration.
            # If you divide first, you get 0 or 1,
            # which can hide an order of magnitude in execution time.
            # So multiply first, then divide.
            var n = self.min_time_ns * prev_iters // prev_dur
            # Run 1.2x more iterations than we think we need.
            n += n // 5
            # Don't grow too fast in case we had timing errors previously.
            n = min(n, 10 * prev_iters)
            # Be sure to run at least one more than last time.
            n = max(n, prev_iters + 1)
            # Don't run more than 1e9 times.
            # (This also keeps n in int range on 32 bit platforms.)
            n = min(n, 1000_000_000)

            @parameter
            @always_inline
            fn benchmark_fn():
                for __ in range(n):
                    func()

            prev_dur = time_function[benchmark_fn]()
            prev_iters = n
            total_iters += prev_iters
            time_elapsed += prev_dur
        return time_elapsed // total_iters


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
