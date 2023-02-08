# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

from Time import now
from Int import Int
from Range import range


struct Benchmark:
    alias _default_num_warmup = 10
    alias _default_max_iters = 100_000
    alias _default_min_time_ns = 500_000_000  # 500ms
    alias _default_max_time_ns = 1000_000_000  # 1s

    var num_warmup: Int
    var max_iters: Int
    var min_time_ns: Int
    var max_time_ns: Int

    fn __new__() -> Benchmark:
        return Benchmark(
            _default_num_warmup,
            _default_max_iters,
            _default_min_time_ns,
            _default_max_time_ns,
        )

    fn __new__(
        num_warmup: Int, max_iters: Int, min_time_ns: Int, max_time_ns: Int
    ) -> Benchmark:
        """Constructs a benchmark object that given a function will benchmark it
        until min_tims_ns has elapsed and either max_time_ns OR max_iters is hit.


        Args:
            num_warmup(Int): number of warmup iterations to run before starting benchmarking.
            max_iters(Int): max number of iterations to run.
            min_time_ns(Int): upper bound on benchmarking time in ns.
            max_time_ns(Int): lower bound on benchmarking time in ns.
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
        """benchmark the given function until min_tims_ns has elapsed and either
        max_time_ns OR max_iters is hit.


        Args:
            func: the function to benchmark.

        Returns:
            Int: average execution time of func in ns.
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
            prev_dur = Int.max(1, prev_dur)  # avoid dividing by 0
            # Order of operations matters.
            # For very fast benchmarks, prev_iterations ~= prev_duration.
            # If you divide first, you get 0 or 1,
            # which can hide an order of magnitude in execution time.
            # So multiply first, then divide.
            var n = self.min_time_ns * prev_iters // prev_dur
            # Run 1.2x more iterations than we think we need.
            n += n // 5
            # Don't grow too fast in case we had timing errors previously.
            n = Int.min(n, 10 * prev_iters)
            # Be sure to run at least one more than last time.
            n = Int.max(n, prev_iters + 1)
            # Don't run more than 1e9 times.
            # (This also keeps n in int range on 32 bit platforms.)
            n = Int.min(n, 1000_000_000)

            tic = now()
            for __ in range(n):  # TODO(#8365)
                func()
            toc = now()

            prev_dur = toc - tic
            prev_iters = n
            total_iters += prev_iters
            time_elapsed += prev_dur
        return time_elapsed // total_iters
