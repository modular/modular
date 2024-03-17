# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
"""Implements benchmark utilties."""

from gpu.host import synchronize
from gpu.host.event import Event, time_function
from gpu.host.stream import Stream

from .benchmark import Report, _run_impl, _RunOptions

# ===----------------------------------------------------------------------===#
# run
# ===----------------------------------------------------------------------===#


@always_inline
fn run[
    func: fn (Stream) -> None
](
    num_warmup: Int = 2,
    max_iters: Int = 1_000_000_000,
    min_runtime_secs: Float64 = 2,
    max_runtime_secs: Float64 = 60,
    max_batch_size: Int = 0,
) -> Report:
    """Benchmarks the function passed in as a parameter.

    Benchmarking continues until 'min_time_ns' has elapsed and either
    `max_time_ns` OR `max_iters` is achieved.

    Parameters:
        func: The function to benchmark.

    Args:
        num_warmup: Number of warmup iterations to run before starting
            benchmarking (default 2).
        max_iters: Max number of iterations to run (default `1_000_000_000`).
        min_runtime_secs: Upper bound on benchmarking time in secs (default `2`).
        max_runtime_secs: Lower bound on benchmarking time in secs (default `60`).
        max_batch_size: The maximum number of iterations to perform per time
            measurement.

    Returns:
        Average execution time of func in ns.
    """

    var stream: Stream

    try:
        stream = Stream()
    except e:
        print(e)

    @parameter
    @always_inline
    fn benchmark_fn(num_iters: Int) -> Int:
        @parameter
        @always_inline
        fn iter_fn(stream: Stream):
            for _ in range(num_iters):
                func(stream)

        return time_function[iter_fn](stream)

    var stats = _run_impl(
        _RunOptions[benchmark_fn](
            max_batch_size=max_batch_size,
            num_warmup=num_warmup,
            max_iters=max_iters,
            min_runtime_secs=min_runtime_secs,
            max_runtime_secs=max_runtime_secs,
        )
    )
    _ = (
        stream ^
    )  # without this stream is deleted before last use of benchmark_fn
    return stats


@always_inline
fn run[
    func: fn (Stream) capturing -> None
](
    num_warmup: Int = 2,
    max_iters: Int = 1_000_000_000,
    min_runtime_secs: Float64 = 2,
    max_runtime_secs: Float64 = 60,
    max_batch_size: Int = 0,
) -> Report:
    """Benchmarks the function passed in as a parameter.

    Benchmarking continues until 'min_time_ns' has elapsed and either
    `max_time_ns` OR `max_iters` is achieved.

    Parameters:
        func: The function to benchmark.

    Args:
        num_warmup: Number of warmup iterations to run before starting
            benchmarking (default 2).
        max_iters: Max number of iterations to run (default `1_000_000_000`).
        min_runtime_secs: Upper bound on benchmarking time in secs (default `2`).
        max_runtime_secs: Lower bound on benchmarking time in secs (default `60`).
        max_batch_size: The maximum number of iterations to perform per time
            measurement.

    Returns:
        Average execution time of func in ns.
    """

    var stream: Stream

    try:
        stream = Stream()
    except e:
        print(e)

    @parameter
    @always_inline
    fn benchmark_fn(num_iters: Int) -> Int:
        @parameter
        @always_inline
        fn iter_fn(stream: Stream):
            for _ in range(num_iters):
                func(stream)

        return time_function[iter_fn](stream)

    var stats = _run_impl(
        _RunOptions[benchmark_fn](
            max_batch_size=max_batch_size,
            num_warmup=num_warmup,
            max_iters=max_iters,
            min_runtime_secs=min_runtime_secs,
            max_runtime_secs=max_runtime_secs,
        )
    )

    _ = (
        stream ^
    )  # without this stream is deleted before last use of benchmark_fn
    return stats


@always_inline
fn run[
    func: fn (Stream) raises capturing -> None
](
    num_warmup: Int = 2,
    max_iters: Int = 1_000_000_000,
    min_runtime_secs: Float64 = 2,
    max_runtime_secs: Float64 = 60,
    max_batch_size: Int = 0,
) -> Report:
    """Benchmarks the function passed in as a parameter.

    Benchmarking continues until 'min_time_ns' has elapsed and either
    `max_time_ns` OR `max_iters` is achieved.

    Parameters:
        func: The function to benchmark.

    Args:
        num_warmup: Number of warmup iterations to run before starting
            benchmarking (default 2).
        max_iters: Max number of iterations to run (default `1_000_000_000`).
        min_runtime_secs: Upper bound on benchmarking time in secs (default `2`).
        max_runtime_secs: Lower bound on benchmarking time in secs (default `60`).
        max_batch_size: The maximum number of iterations to perform per time
            measurement.

    Returns:
        Average execution time of func in ns.
    """
    var stream: Stream
    try:
        stream = Stream()
    except e:
        print(e)

    @parameter
    @always_inline
    fn benchmark_fn(num_iters: Int) -> Int:
        @parameter
        @always_inline
        fn iter_fn(stream: Stream):
            try:
                for _ in range(num_iters):
                    func(stream)
            except e:
                print(e)

        return time_function[iter_fn](stream)

    var stats = _run_impl(
        _RunOptions[benchmark_fn](
            max_batch_size=max_batch_size,
            num_warmup=num_warmup,
            max_iters=max_iters,
            min_runtime_secs=min_runtime_secs,
            max_runtime_secs=max_runtime_secs,
        )
    )

    _ = (
        stream ^
    )  # without this stream is deleted before last use of benchmark_fn
    return stats


@always_inline
fn run[
    func: fn () capturing -> None
](
    num_warmup: Int = 2,
    max_iters: Int = 1_000_000_000,
    min_runtime_secs: Float64 = 2,
    max_runtime_secs: Float64 = 60,
    max_batch_size: Int = 0,
) -> Report:
    """Benchmarks the function passed in as a parameter.

    Benchmarking continues until 'min_time_ns' has elapsed and either
    `max_time_ns` OR `max_iters` is achieved.

    Parameters:
        func: The function to benchmark.

    Args:
        num_warmup: Number of warmup iterations to run before starting
            benchmarking (default 2).
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
    fn benchmark_fn(num_iters: Int) -> Int:
        @parameter
        @always_inline
        fn iter_fn():
            try:
                for _ in range(num_iters):
                    func()
                synchronize()
            except e:
                print(e)

        return time_function[iter_fn]()

    return _run_impl(
        _RunOptions[benchmark_fn](
            max_batch_size=max_batch_size,
            num_warmup=num_warmup,
            max_iters=max_iters,
            min_runtime_secs=min_runtime_secs,
            max_runtime_secs=max_runtime_secs,
        )
    )


# ===----------------------------------------------------------------------===#
# time_async_cuda_kernel
# ===----------------------------------------------------------------------===#


fn time_async_cuda_kernel[
    func: fn (Stream) raises capturing -> None
](num_iters: Int) raises -> Int:
    """Runs a CUDA kernel for an input number of iterations and returns total
    time taken.

    Parameters:
        func: The kernel to benchmark.

    Args:
        num_iters: The number of iterations to run the Kernel.

    Returns:
        The total elapsed time.
    """

    var ret: Int = 0
    try:
        var stream = Stream()
        var start = Event()
        var end = Event()
        start.record(stream)
        for _ in range(num_iters):
            func(stream)
        end.record(stream)
        end.sync()
        ret = int(start.elapsed(end) * 1e6)
    except e:
        abort(e)
    return ret
