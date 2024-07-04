# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

import time


trait Benchmarkable:
    fn global_pre_run(self):
        """Function that runs once during the start of the entire Benchmark trace.
        """
        ...

    fn pre_run(self):
        """Function that runs before the Target Function during every benchmark iteration.
        """
        ...

    fn run(self):
        """The target Function that is to be benchmarked."""
        ...

    fn post_run(self):
        """Function that runs after the Target Function during every benchmark iteration.
        """
        ...

    fn global_post_run(self):
        """Function that runs once during the end of the entire Benchmark trace.
        """
        ...


@always_inline
fn run[
    T: Benchmarkable
](benchmark_obj: T, name: String, num_iters: Int = 10) -> None:
    benchmark_obj.global_pre_run()

    var total_time = 0.0
    for _ in range(num_iters):
        benchmark_obj.pre_run()
        var start_time = time.perf_counter_ns()
        benchmark_obj.run()
        var end_time = time.perf_counter_ns()
        benchmark_obj.post_run()
        total_time += end_time - start_time

    var average_execution_time = total_time / num_iters / 1e3

    benchmark_obj.global_post_run()
    print(name, average_execution_time)
