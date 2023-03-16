# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
"""The module implements 'Benchmark' class for runtime benchmarking."""

from Int import Int
from Math import min, max
from Range import range
from Time import now


struct Benchmark:
    """A benchmark harness.

    The class allows to benchmark a given function (passed as a parameter) and
    configure various benchmarking parameters, such as number of warmup
    iterations, maximum number of iterations, minimum and maximum elapsed time.
    """

    var num_warmup: Int
    var max_iters: Int
    var min_time_ns: Int
    var max_time_ns: Int

    fn __clone__(self&) -> Self:
        return Self {
            num_warmup: self.num_warmup,
            max_iters: self.max_iters,
            min_time_ns: self.min_time_ns,
            max_time_ns: self.max_time_ns,
        }

    fn __init__(
        num_warmup: Int = 10,
        max_iters: Int = 100_000,
        min_time_ns: Int = 500_000_000,  # 500ms
        max_time_ns: Int = 1000_000_000,  # 1s
    ) -> Benchmark:
        """Constructs a new benchmark object.

        Given a function the benchmark object will benchmark it until
        min_tims_ns has elapsed and either max_time_ns OR max_iters is hit.

        Args:
            num_warmup (Int): Number of warmup iterations to run before
              starting benchmarking.
            max_iters (Int): Max number of iterations to run.
            min_time_ns (Int): Upper bound on benchmarking time in ns.
            max_time_ns (Int): Lower bound on benchmarking time in ns.

        Returns:
            Benchmark: A new constructed benchmark object.
        """
        return Benchmark {
            num_warmup: num_warmup,
            max_iters: max_iters,
            min_time_ns: min_time_ns,
            max_time_ns: max_time_ns,
        }

    @always_inline
    fn run[
        func: __mlir_type.`!kgen.signature<() -> !lit.none>`,
    ](self) -> Int:
        """Benchmark the given function.

        Benchmarking continues until min_tims_ns has elapsed and either
        max_time_ns OR max_iters is hit.

        Parameters:
            func (!kgen.signature<() -> !lit.none>): The function to benchmark.

        Returns:
            Int: Average execution time of func in ns.
        """

        # run for specified number of warmup iterations
        var tic = now()
        for _ in range(self.num_warmup):
            func()
        var toc = now()

        var prev_iters = self.num_warmup
        var prev_dur = toc - tic if self.num_warmup > 0 else Int(0)
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

            tic = now()
            for __ in range(n):  # TODO(#8365)
                func()
            toc = now()

            prev_dur = toc - tic
            prev_iters = n
            total_iters += prev_iters
            time_elapsed += prev_dur
        return time_elapsed // total_iters
