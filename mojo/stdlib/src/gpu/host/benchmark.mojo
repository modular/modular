# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
"""Implements benchmark utilties."""

from benchmark.benchmark import Report, _RunOptions, _run_impl
from .event import time_function

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
fn run[
    func: fn () raises capturing -> None
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
            try:
                for _ in range(num_iters):
                    func()
            except e:
                print(e)

        return time_function[iter_fn]()

    return _run_impl(
        _RunOptions[benchmark_fn](
            num_warmup, max_iters, min_time_secs, max_time_secs
        )
    )
