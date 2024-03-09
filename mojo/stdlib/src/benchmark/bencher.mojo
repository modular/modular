# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

import time
from collections import Optional
from math._numerics import FlushDenormals
from pathlib import Path
from sys.arg import argv

from .benchmark import *
from .benchmark import _run_impl, _RunOptions
from stdlib.builtin.file import FileHandle


@value
struct BenchConfig(CollectionElement):
    """Defines a benchmark configuration struct to control
    execution times and frequency.
    """

    var out_file: Optional[Path]
    """Output file to write results to."""
    var min_runtime_secs: Float64
    """Upper bound on benchmarking time in secs."""
    var max_runtime_secs: Float64
    """Lower bound on benchmarking time in secs."""
    var max_batch_size: Int
    """The maximum number of iterations to perform per time measurement."""
    var max_iters: Int
    """Max number of iterations to run."""
    var warmup_iters: Int
    """Number of warmup iterations to run before starting benchmarking."""
    var num_repetitions: Int
    """Number of times the benchmark has to be repeated."""
    var flush_denormals: Bool
    """Whether or not the denormal values are flushed."""

    fn __init__(
        inout self,
        /,
        out_file: Optional[Path] = None,
        min_runtime_secs: Float64 = 0.1,
        max_runtime_secs: Float64 = 1,
        warmup_iters: Int = 2,
        max_batch_size: Int = 0,
        max_iters: Int = 1_000_000_000,
        num_repetitions: Int = 1,
        flush_denormals: Bool = True,
    ) raises:
        """Constructs and initializes Benchmark config object with default and inputed values.

        Args:
            out_file: Output file to write results to.
            min_runtime_secs: Upper bound on benchmarking time in secs (default `0.1`).
            max_runtime_secs: Lower bound on benchmarking time in secs (default `1`).
            warmup_iters: Number of warmup iterations to run before starting benchmarking (default 2).
            max_batch_size: The maximum number of iterations to perform per time measurement.
            max_iters: Max number of iterations to run (default `1_000_000_000`).
            num_repetitions: Number of times the benchmark has to be repeated.
            flush_denormals: Whether or not the denormal values are flushed.
        """

        self.min_runtime_secs = min_runtime_secs
        self.max_runtime_secs = max_runtime_secs
        self.max_batch_size = max_batch_size
        self.max_iters = max_iters
        self.warmup_iters = warmup_iters
        self.out_file = out_file
        self.num_repetitions = num_repetitions
        self.flush_denormals = flush_denormals

        @parameter
        fn argparse() raises:
            """Parse cmd line args to define benchmark configuration."""

            var args = argv()
            for i in range(len(args)):
                if args[i] == "-o":
                    self.out_file = Path(args[i + 1])
                    i += 2
                elif args[i] == "-r":
                    self.num_repetitions = int(args[i + 1])

        argparse()


@value
struct BenchId:
    """Defines a benchmark Id struct to identify and represent a particular benchmark execution.
    """

    var func_name: String
    """The target function name."""
    var input_id: Optional[String]
    """The target function input id phrase."""

    fn __init__(inout self, func_name: String, input_id: String):
        """Constructs a Benchmark Id object from input function name and Id phrase.

        Args:
            func_name: The target function name.
            input_id: The target function input id phrase.
        """

        self.func_name = func_name
        self.input_id = input_id

    fn __init__(inout self, func_name: String):
        """Constructs a Benchmark Id object from input function name.

        Args:
            func_name: The target function name.
        """

        self.func_name = func_name
        self.input_id = None


@value
struct BenchmarkInfo(CollectionElement, Stringable):
    """Defines a Benchmark Info struct to record execution Statistics."""

    var name: String
    """The name of the benchmark."""
    var result: Report
    """The output report after executing a benchmark."""
    var elems: Optional[
        Int
    ]  # TODO: wrap in a throughput type that can be flops or bytes
    """Optional arg used to represent a specific metric like throughput."""

    fn __init__(inout self, name: String, result: Report, elems: Optional[Int]):
        """Constructs a Benchmark Info object to return Benchmark report and Stats.

        Args:
            name: The name of the benchmark.
            result: The output report after executing a benchmark.
            elems: Optional arg used to represent a specific metric like throughput.
        """

        self.name = name
        self.result = result
        self.elems = elems

    fn _throughput(self) -> Float64:
        """Computes Benchmark Throughput value.

        Returns:
            The throughput values as a floating point 64.
        """

        return self.elems.value() * 1e-9 / self.result.mean(unit=Unit.s)

    fn __str__(self) -> String:
        """Formats Benchmark Statistical Info.

        Returns:
            A string representing benchmark statistics.
        """

        var elems = "," + str(self._throughput()) if self.elems else ""
        return (
            self.name
            + ","
            + self.result.mean(unit=Unit.ms)
            + ","
            + str(self.result.iters())
            + elems
        )


@value
struct Mode:
    """Defines a Benchmark Mode to distinguish between test runs and actual benchmarks.
    """

    var value: Int
    """Represents the mode type."""
    alias Benchmark = Mode(0)
    alias Test = Mode(1)

    fn __eq__(self, other: Self) -> Bool:
        """Check if its Benchmark mode or test mode.

        Args:
            other: The mode to be compared against.

        Returns:
            If its a test mode or benchmark mode.
        """

        return self.value == other.value


@value
struct Bench:
    """Defines the main Benchmark struct which executes a Benchmark and print result.
    """

    var config: BenchConfig
    """Constructs a Benchmark object based on specific configuration and mode."""
    var mode: Mode
    """Benchmark mode object representing benchmark or test mode."""
    var info_vec: List[BenchmarkInfo]
    """A list containing the bencmark info."""

    fn __init__(
        inout self,
        config: Optional[BenchConfig] = None,
        mode: Mode = Mode.Benchmark,
    ) raises:
        """Constructs a Benchmark object based on specific configuration and mode.

        Args:
            config: Benchmark configuration object to control length and frequency of benchmarks.
            mode: Benchmark mode object representing benchmark or test mode.
        """

        self.config = config.value() if config else BenchConfig()
        self.mode = mode
        self.info_vec = List[BenchmarkInfo]()

        @parameter
        fn argparse():
            """Parse cmd line args to define benchmark configuration."""

            var args = argv()
            for i in range(len(args)):
                if args[i] == "-t":
                    self.mode = Mode.Test

        argparse()

    fn bench_with_input[
        T: AnyType,
        bench_fn: fn (inout Bencher, T) capturing -> None,
    ](
        inout self,
        bench_id: BenchId,
        input: T,
        throughput_elems: Optional[Int] = None,
    ) raises:
        """Benchmarks an input function with input args of type AnyType.

        Parameters:
            T: Benchmark function input type.
            bench_fn: The function to be benchmarked.

        Args:
            bench_id: The benchmark Id object used for identification.
            input: Represents the target function's input arguments.
            throughput_elems: Optional argument representing algorithmic throughput.
        """

        @parameter
        fn input_closure(inout b: Bencher):
            """Executes benchmark for a target function.

            Args:
                b: The bencher object to facilitate benchmark execution.
            """

            bench_fn(b, input)

        self.bench_function[input_closure](bench_id, throughput_elems)

    fn bench_with_input[
        T: AnyRegType,
        bench_fn: fn (inout Bencher, T) capturing -> None,
    ](
        inout self,
        bench_id: BenchId,
        input: T,
        throughput_elems: Optional[Int] = None,
    ) raises:
        """Benchmarks an input function with input args of type AnyRegType.

        Parameters:
            T: Benchmark function input type.
            bench_fn: The function to be benchmarked.

        Args:
            bench_id: The benchmark Id object used for identification.
            input: Represents the target function's input arguments.
            throughput_elems: Optional argument representing algorithmic throughput.
        """

        @parameter
        fn input_closure(inout b: Bencher):
            """Executes benchmark for a target function.

            Args:
                b: The bencher object to facilitate benchmark execution.
            """

            bench_fn(b, input)

        self.bench_function[input_closure](bench_id, throughput_elems)

    fn bench_function[
        bench_fn: fn (inout Bencher) capturing -> None
    ](
        inout self, bench_id: BenchId, throughput_elems: Optional[Int] = None
    ) raises:
        """Benchmarks or Tests an input function.

        Parameters:
            bench_fn: The function to be benchmarked.

        Args:
            bench_id: The benchmark Id object used for identification.
            throughput_elems: Optional argument representing algorithmic throughput.
        """

        if self.mode == Mode.Benchmark:
            for _ in range(self.config.num_repetitions):
                self._bench[bench_fn](bench_id, throughput_elems)
        elif self.mode == Mode.Test:
            self._test[bench_fn]()

    # TODO (#31795): overload should not be needed
    fn bench_function[
        bench_fn: fn (inout Bencher) raises capturing -> None
    ](
        inout self, bench_id: BenchId, throughput_elems: Optional[Int] = None
    ) raises:
        """Benchmarks or Tests an input function.

        Parameters:
            bench_fn: The function to be benchmarked.

        Args:
            bench_id: The benchmark Id object used for identification.
            throughput_elems: Optional argument representing algorithmic throughput.
        """

        @parameter
        fn abort_on_err(inout b: Bencher):
            """Aborts benchmark in case of an error.

            Args:
                b: The bencher object to facilitate benchmark execution.
            """

            # TODO (#31795): if we don't catch the exception here we have to overload
            # almost every function in stdlib benchmark and stdlib time
            try:
                bench_fn(b)
            except e:
                abort(e)

        self.bench_function[abort_on_err](bench_id, throughput_elems)

    fn _test[bench_fn: fn (inout Bencher) capturing -> None](inout self) raises:
        """Tests an input function by executing it only once.

        Parameters:
            bench_fn: The function to be benchmarked.
        """

        var b = Bencher(1)
        bench_fn(b)

    fn _bench[
        user_bench_fn: fn (inout Bencher) capturing -> None
    ](inout self, bench_id: BenchId, throughput_elems: Optional[Int]) raises:
        """Benchmarks an input function.

        Parameters:
            user_bench_fn: The function to be benchmarked.

        Args:
            bench_id: The benchmark Id object used for identification.
            throughput_elems: Optional argument representing algorithmic throughput.
        """

        @parameter
        fn bench_fn(inout b: Bencher):
            """Executes benchmark for a target function.

            Args:
                b: The bencher object to facilitate benchmark execution.
            """

            if self.config.flush_denormals:
                with FlushDenormals():
                    user_bench_fn(b)
            else:
                user_bench_fn(b)

        # warmup
        var b = Bencher(num_iters=self.config.warmup_iters)
        bench_fn(b)

        @parameter
        @always_inline
        fn benchmark_fn(num_iters: Int) -> Int:
            """Executes benchmark for a target function.

            Args:
                num_iters: The number of iterations to run a target function.
            """

            var b = Bencher(num_iters)
            bench_fn(b)
            return b.elapsed

        var full_name = bench_id.func_name + "/" + bench_id.input_id.value() if bench_id.input_id else bench_id.func_name
        print_no_newline("Running", full_name, "...")

        var res = _run_impl(
            _RunOptions[benchmark_fn](
                max_batch_size=self.config.max_batch_size,
                num_warmup=self.config.warmup_iters,
                max_iters=self.config.max_iters,
                min_runtime_secs=self.config.min_runtime_secs,
                max_runtime_secs=self.config.max_runtime_secs,
            )
        )
        print("done.")

        self.info_vec.push_back(
            BenchmarkInfo(
                full_name,
                res,
                throughput_elems,
            )
        )

    fn dump_report(self) raises:
        """Prints out the report from a Benchmark execution."""

        var report = String("name, met (ms), iters, throughput (Gelems/s)\n")
        var num_runs = len(self.info_vec)
        for i in range(num_runs):
            var sep = "\n" if i < num_runs - 1 else ""
            report += str(self.info_vec[i]) + sep
        print("------------------------------------------")
        print("Benchmark results")
        print("------------------------------------------")
        print(report)

        if self.config.out_file:
            with open(self.config.out_file.value(), "w") as f:
                f.write(report)


@value
@register_passable
struct Bencher:
    """Defines a Bencher struct which facilitates the timing of a target function.
    """

    var num_iters: Int
    """ Number of iterations to run the target function."""

    var elapsed: Int
    """ The total time elpased when running the target function."""

    fn __init__(inout self, num_iters: Int):
        """Constructs a Bencher object to run and time a function.

        Args:
            num_iters: Number of times to run the target function.
        """

        self.num_iters = num_iters
        self.elapsed = 0

    fn iter[iter_fn: fn () capturing -> None](inout self):
        """Returns the total elapsed time by running a target function a particular
        number of times.

        Parameters:
            iter_fn: The target function to benchmark.
        """

        var start = time.now()
        for _ in range(self.num_iters):
            iter_fn()
        var stop = time.now()
        self.elapsed = stop - start

    fn iter_custom[iter_fn: fn (Int) capturing -> Int](inout self):
        """Times a target function with custom number of iterations.

        Parameters:
            iter_fn: The target function to benchmark.
        """

        self.elapsed = iter_fn(self.num_iters)

    fn iter[iter_fn: fn () capturing raises -> None](inout self) raises:
        """Returns the total elapsed time by running a target function a particular
        number of times.

        Parameters:
            iter_fn: The target function to benchmark.
        """

        var start = time.now()
        for _ in range(self.num_iters):
            iter_fn()
        var stop = time.now()
        self.elapsed = stop - start

    # TODO (#31795):  overload should not be needed
    fn iter_custom[iter_fn: fn (Int) capturing raises -> Int](inout self):
        """Times a target function with custom number of iterations.

        Parameters:
            iter_fn: The target function to benchmark.
        """

        try:
            self.elapsed = iter_fn(self.num_iters)
        except e:
            abort(e)
