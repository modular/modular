# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

import time
from collections import Optional
from collections.string.string import _calc_initial_buffer_size_int32
from os import abort
from pathlib import Path
from sys.arg import argv

from gpu.host import DeviceContext
from stdlib.builtin.file import FileHandle
from stdlib.builtin.io import _snprintf
from testing import assert_true

from utils.numerics import FlushDenormals

from .benchmark import *
from .benchmark import _run_impl, _RunOptions


@value
struct BenchMetric(CollectionElement):
    """Defines a benchmark throughput metric."""

    var code: Int
    """Op-code of the Metric."""
    var name: String
    """Metric's name."""
    var unit: String
    """Metric's throughput rate unit (count/second)."""

    alias elements = BenchMetric(0, "throughput", "GElems/s")
    alias bytes = BenchMetric(1, "DataMovement", "GB/s")
    alias flops = BenchMetric(2, "Arithmetic", "GFLOPS/s")
    alias theoretical_flops = BenchMetric(
        3, "TheoreticalArithmetic", "GFLOPS/s"
    )

    alias DEFAULTS = List[BenchMetric](Self.elements, Self.bytes, Self.flops)
    """Default set of benchmark metrics."""

    fn __init__(out self, *, other: Self):
        """Explicitly construct a deep copy of the provided value.

        Args:
            other: The value to copy.
        """
        self = other

    fn __str__(self) -> String:
        """Gets a string representation of this metric.

        Returns:
            The string representation."""
        return self.name + " (" + self.unit + ")"

    fn __eq__(self, other: Self) -> Bool:
        """Compares two metrics for equality.

        Args:
            other: The metric to compare.

        Returns:
            True if the two metrics are equal.
        """
        return self.code == other.code

    fn __ne__(self, other: Self) -> Bool:
        """Compares two metrics for inequality.

        Args:
            other: The metric to compare.

        Returns:
            True if the two metrics are NOT equal.
        """
        return self.code != other.code

    fn check_name(self, alt_name: String) -> Bool:
        """Checks whether a string contains the metric's name.

        Args:
            alt_name: Alternative name of a metric.

        Returns:
            True if 'alt_name' is valid alternative of the metric's name.
        """
        return self.name.lower() == alt_name.lower()

    @staticmethod
    fn get_metric_from_list(
        name: String, metric_list: List[BenchMetric]
    ) raises -> BenchMetric:
        """Gets a metric from a given list using only the metric's name.

        Args:
            name: Metric's name.
            metric_list: List of metrics to search.

        Returns:
            The selected metric.
        """
        for m in metric_list:
            if m[].check_name(name):
                return m[]

        var err: String = "\n"
        err += str("-") * 80 + "\n"
        err += str("-") * 80 + "\n"
        err += "Couldn't match metric [" + name + "]\n"
        err += "Available throughput metrics (case-insensitive) in the list:\n"
        for m in metric_list:
            err += "    metric: [" + m[].name.lower() + "]\n"
        err += str("-") * 80 + "\n"
        err += str("-") * 80 + "\n"
        err += "[ERROR]: metric [" + name + "] is NOT supported!\n"
        raise Error(err)


@value
struct ThroughputMeasure(CollectionElement):
    """Records a throughput metric of metric BenchMetric and value."""

    var metric: BenchMetric
    """Type of throughput metric."""
    var value: Int
    """Measured count of throughput metric."""

    fn __init__(
        mut self,
        name: String,
        value: Int,
        reference: List[BenchMetric] = BenchMetric.DEFAULTS,
    ) raises:
        """Creates a `ThroughputMeasure` based on metric's name.

        Args:
            name: The name of BenchMetric in its corresponding reference.
            value: The measured value to assign to this metric.
            reference: Pointer variadic list of BenchMetrics that contains this metric.

        Example:
            For the default bench metrics BenchMetric.DEFAULTS the
            following are equivalent:
                - ThroughputMeasure(BenchMetric.fmas, 1024)
                - ThroughputMeasure("fmas", 1024)
                - ThroughputMeasure("fmas", 1024, BenchMetric.DEFAULTS)
        """
        var metric = BenchMetric.get_metric_from_list(name, reference)
        self.metric = metric
        self.value = value

    fn __init__(out self, *, other: Self):
        """Explicitly construct a deep copy of the provided value.

        Args:
            other: The value to copy.
        """
        self = other

    fn __str__(self) -> String:
        """Gets a string representation of this `ThroughputMeasure`.

        Returns:
            The string represntation.
        """
        return str(self.metric)

    fn compute(self, elapsed_sec: Float64) -> Float64:
        """Computes throughput rate for this metric per unit of time (second).

        Args:
            elapsed_sec: Elapsed time measured in seconds.

        Returns:
            The throughput values as a floating point 64.
        """
        # TODO: do we need support other units of time (ms, ns)?
        return (self.value) * 1e-9 / elapsed_sec


@always_inline
fn _str_fmt_width(str: String, str_width: Int) -> String:
    """Formats string with a given width.

    Returns:
        sprintf("%-*s", str_width, str)
    """
    debug_assert(str_width > 0, "Should have str_width>0")

    alias N = 2048
    var x = String._buffer_type()
    x.reserve(N)
    x.size += _snprintf["%-*s"](x.data, N, str_width, str.unsafe_ptr())
    debug_assert(x.size < N, "Attempted to access outside array bounds!")
    x.size += 1
    return String(x)


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
    var min_warmuptime_secs: Float64
    """Lower bound on warmup time in secs."""
    var max_batch_size: Int
    """The maximum number of iterations to perform per time measurement."""
    var max_iters: Int
    """Max number of iterations to run."""
    var num_repetitions: Int
    """Number of times the benchmark has to be repeated."""
    var flush_denormals: Bool
    """Whether or not the denormal values are flushed."""
    var show_progress: Bool
    """Whether or not to show the progress of each benchmark."""
    var tabular_view: Bool
    """Whether to print results in csv readable/tabular format."""
    var verbose_timing: Bool
    """Whether to print verbose timing results."""

    alias VERBOSE_TIMING_LABELS = List[String](
        "min (ms)", "mean (ms)", "max (ms)", "duration (ms)"
    )
    """Labels to print verbose timing results."""

    # TODO: to add median and stddev to verbose-timing

    fn __init__(
        mut self,
        /,
        out_file: Optional[Path] = None,
        min_runtime_secs: Float64 = 1.0,
        max_runtime_secs: Float64 = 2.0,
        min_warmuptime_secs: Float64 = 1.0,
        max_batch_size: Int = 0,
        max_iters: Int = 1_000_000_000,
        num_repetitions: Int = 1,
        flush_denormals: Bool = True,
    ) raises:
        """Constructs and initializes Benchmark config object with default and inputed values.

        Args:
            out_file: Output file to write results to.
            min_runtime_secs: Lower bound on benchmarking time in secs (default `0.1`).
            max_runtime_secs: Upper bound on benchmarking time in secs (default `1`).
            min_warmuptime_secs: Lower bound on warmup time in secs (default `1.0`).
            max_batch_size: The maximum number of iterations to perform per time measurement.
            max_iters: Max number of iterations to run (default `1_000_000_000`).
            num_repetitions: Number of times the benchmark has to be repeated.
            flush_denormals: Whether or not the denormal values are flushed.
        """

        self.min_runtime_secs = min_runtime_secs
        self.max_runtime_secs = max_runtime_secs
        self.min_warmuptime_secs = min_warmuptime_secs
        self.max_batch_size = max_batch_size
        self.max_iters = max_iters
        self.out_file = out_file
        self.num_repetitions = num_repetitions
        self.flush_denormals = flush_denormals
        self.show_progress = True
        # TODO: set tabular_view=True as default
        self.tabular_view = False
        self.verbose_timing = False

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
                elif args[i] == "--tabular":
                    self.tabular_view = True
                elif args[i] == "--no-progress":
                    self.show_progress = False
                elif args[i] == "--verbose":
                    self.verbose_timing = True
                # TODO: add an arg for bench batchsize

        argparse()

    fn __init__(out self, *, other: Self):
        """Explicitly construct a deep copy of the provided value.

        Args:
            other: The value to copy.
        """
        self = other


@value
struct BenchId:
    """Defines a benchmark Id struct to identify and represent a particular benchmark execution.
    """

    var func_name: String
    """The target function name."""
    var input_id: Optional[String]
    """The target function input id phrase."""

    fn __init__(out self, func_name: String, input_id: String):
        """Constructs a Benchmark Id object from input function name and Id phrase.

        Args:
            func_name: The target function name.
            input_id: The target function input id phrase.
        """

        self.func_name = func_name
        self.input_id = input_id

    @implicit
    fn __init__(out self, func_name: String):
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
    var measures: List[ThroughputMeasure]
    """Optional arg used to represent a list of ThroughputMeasure's."""

    var verbose_timing: Bool
    """Whether to print verbose timing results."""

    fn __init__(
        mut self,
        name: String,
        result: Report,
        measures: List[ThroughputMeasure] = List[ThroughputMeasure](),
        verbose_timing: Bool = False,
    ):
        """Constructs a `BenchmarkInfo` object to return benchmark report and
        statistics.

        Args:
            name: The name of the benchmark.
            result: The output report after executing a benchmark.
            measures: Optional arg used to represent a list of ThroughputMeasure's.
            verbose_timing: Whether to print verbose timing results.
        """

        self.name = name
        self.result = result
        self.measures = measures
        self.verbose_timing = verbose_timing

    fn __init__(out self, *, other: Self):
        """Explicitly construct a deep copy of the provided value.

        Args:
            other: The value to copy.
        """
        self = other

    fn __str__(self) -> String:
        """Gets a string representation of this `BenchmarkInfo` value.

        Returns:
            A string representing benchmark statistics.
        """

        var throughput: String = ""
        for i in range(len(self.measures)):
            var rate = self.measures[i].compute(self.result.mean(unit=Unit.s))
            throughput = throughput + "," + str(rate)

        # add verbose-timing results
        if self.verbose_timing:
            var verbose_timing_vals = List[Float64](
                self.result.min(unit=Unit.ms),
                self.result.mean(unit=Unit.ms),
                self.result.max(unit=Unit.ms),
                self.result.duration(unit=Unit.ms),
            )
            for t in verbose_timing_vals:
                throughput = throughput + "," + str(t[])

        return (
            '"'
            + self.name
            + '"'
            + ","
            + str(self.result.mean(unit=Unit.ms))
            + ","
            + str(self.result.iters())
            + throughput
        )

    fn _csv_str(self, column_width: VariadicList[Int]) -> String:
        """Formats Benchmark Statistical Info.

        Returns:
            A string representing benchmark statistics.
        """

        var name_width = column_width[0]
        var time_width = column_width[1]
        var iters_width = column_width[2]
        var rate_width = column_width[3]

        debug_assert(column_width[0] > 0, "Name width should be >0")
        alias N = 2048
        var x = String._buffer_type()
        x.reserve(N)

        @always_inline
        @parameter
        fn _append[fmt: StringLiteral, type: AnyType](width: Int, s: type):
            x.size += _snprintf[fmt](x.data + x.size, N, width, s)
            debug_assert(
                x.size < N, "Attempted to access outside array bounds!"
            )

        _append["%-*s, "](name_width, self.name.rstrip())
        _append["%*.6f, "](time_width, self.result.mean(unit=Unit.ms))
        _append["%*d"](iters_width, self.result.iters())

        if len(self.measures) > 0:
            for i in range(len(self.measures)):
                var rate = self.measures[i].compute(self.result.mean(Unit.s))
                _append[", %*.6f"](rate_width, rate)

        if self.verbose_timing:
            # verbose-timing details
            var verbose_timing_vals = List[Float64](
                self.result.min(unit=Unit.ms),
                self.result.mean(unit=Unit.ms),
                self.result.max(unit=Unit.ms),
                self.result.duration(unit=Unit.ms),
            )

            for t in verbose_timing_vals:
                var detail = t[]
                _append[", %*.6f"](time_width, detail)

        x.size += 1
        debug_assert(x.size < N, "Attempted to access outside array bounds!")
        return String(x)


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
        mut self,
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
        bench_fn: fn (mut Bencher, T) raises capturing [_] -> None,
    ](
        mut self,
        bench_id: BenchId,
        input: T,
        measures: List[ThroughputMeasure] = List[ThroughputMeasure](),
    ) raises:
        """Benchmarks an input function with input args of type AnyType.

        Parameters:
            T: Benchmark function input type.
            bench_fn: The function to be benchmarked.

        Args:
            bench_id: The benchmark Id object used for identification.
            input: Represents the target function's input arguments.
            measures: Optional arg used to represent a list of ThroughputMeasure's.
        """

        @parameter
        fn input_closure(mut b: Bencher) raises:
            """Executes benchmark for a target function.

            Args:
                b: The bencher object to facilitate benchmark execution.
            """

            bench_fn(b, input)

        self.bench_function[input_closure](bench_id, measures)

    fn bench_with_input[
        T: AnyType,
        bench_fn: fn (mut Bencher, T) raises capturing [_] -> None,
    ](
        mut self,
        bench_id: BenchId,
        input: T,
        *measures: ThroughputMeasure,
    ) raises:
        """Benchmarks an input function with input args of type AnyType.

        Parameters:
            T: Benchmark function input type.
            bench_fn: The function to be benchmarked.

        Args:
            bench_id: The benchmark Id object used for identification.
            input: Represents the target function's input arguments.
            measures: Variadic arg used to represent a list of ThroughputMeasure's.
        """
        var measures_list = List[ThroughputMeasure]()
        for m in measures:
            measures_list.append(m[])
        self.bench_with_input[T, bench_fn](bench_id, input, measures_list)

    fn bench_with_input[
        T: AnyTrivialRegType,
        bench_fn: fn (mut Bencher, T) raises capturing [_] -> None,
    ](
        mut self,
        bench_id: BenchId,
        input: T,
        measures: List[ThroughputMeasure] = List[ThroughputMeasure](),
    ) raises:
        """Benchmarks an input function with input args of type AnyTrivialRegType.

        Parameters:
            T: Benchmark function input type.
            bench_fn: The function to be benchmarked.

        Args:
            bench_id: The benchmark Id object used for identification.
            input: Represents the target function's input arguments.
            measures: Optional arg used to represent a list of ThroughputMeasure's.
        """

        @parameter
        fn input_closure(mut b: Bencher) raises:
            """Executes benchmark for a target function.

            Args:
                b: The bencher object to facilitate benchmark execution.
            """

            bench_fn(b, input)

        self.bench_function[input_closure](bench_id, measures)

    fn bench_with_input[
        T: AnyTrivialRegType,
        bench_fn: fn (mut Bencher, T) raises capturing [_] -> None,
    ](
        mut self,
        bench_id: BenchId,
        input: T,
        *measures: ThroughputMeasure,
    ) raises:
        """Benchmarks an input function with input args of type AnyTrivialRegType.

        Parameters:
            T: Benchmark function input type.
            bench_fn: The function to be benchmarked.

        Args:
            bench_id: The benchmark Id object used for identification.
            input: Represents the target function's input arguments.
            measures: Variadic arg used to represent a list of ThroughputMeasure's.
        """
        var measures_list = List[ThroughputMeasure]()
        for m in measures:
            measures_list.append(m[])
        self.bench_with_input[T, bench_fn](bench_id, input, measures_list)

    fn bench_function[
        bench_fn: fn (mut Bencher) capturing [_] -> None
    ](
        mut self,
        bench_id: BenchId,
        measures: List[ThroughputMeasure] = List[ThroughputMeasure](),
    ) raises:
        """Benchmarks or Tests an input function.

        Parameters:
            bench_fn: The function to be benchmarked.

        Args:
            bench_id: The benchmark Id object used for identification.
            measures: Optional arg used to represent a list of ThroughputMeasure's.
        """

        if self.mode == Mode.Benchmark:
            for _ in range(self.config.num_repetitions):
                self._bench[bench_fn](bench_id, measures)
        elif self.mode == Mode.Test:
            self._test[bench_fn]()

    fn bench_function[
        bench_fn: fn (mut Bencher) capturing [_] -> None
    ](mut self, bench_id: BenchId, *measures: ThroughputMeasure) raises:
        """Benchmarks or Tests an input function.

        Parameters:
            bench_fn: The function to be benchmarked.

        Args:
            bench_id: The benchmark Id object used for identification.
            measures: Variadic arg used to represent a list of ThroughputMeasure's.
        """
        var measures_list = List[ThroughputMeasure]()
        for m in measures:
            measures_list.append(m[])
        self.bench_function[bench_fn](bench_id, measures_list)

    # TODO (#31795): overload should not be needed
    fn bench_function[
        bench_fn: fn (mut Bencher) raises capturing [_] -> None
    ](
        mut self,
        bench_id: BenchId,
        measures: List[ThroughputMeasure] = List[ThroughputMeasure](),
    ) raises:
        """Benchmarks or Tests an input function.

        Parameters:
            bench_fn: The function to be benchmarked.

        Args:
            bench_id: The benchmark Id object used for identification.
            measures: Optional arg used to represent a list of ThroughputMeasure's.
        """

        @parameter
        fn abort_on_err(mut b: Bencher):
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

        self.bench_function[abort_on_err](bench_id, measures)

    fn bench_function[
        bench_fn: fn (mut Bencher) raises capturing [_] -> None
    ](mut self, bench_id: BenchId, *measures: ThroughputMeasure,) raises:
        """Benchmarks or Tests an input function.

        Parameters:
            bench_fn: The function to be benchmarked.

        Args:
            bench_id: The benchmark Id object used for identification.
            measures: Variadic arg used to represent a list of ThroughputMeasure's.
        """
        var measures_list = List[ThroughputMeasure]()
        for m in measures:
            measures_list.append(m[])
        self.bench_function[bench_fn](bench_id, measures_list)

    fn _test[bench_fn: fn (mut Bencher) capturing [_] -> None](mut self) raises:
        """Tests an input function by executing it only once.

        Parameters:
            bench_fn: The function to be benchmarked.
        """

        var b = Bencher(1)
        bench_fn(b)

    fn _bench[
        user_bench_fn: fn (mut Bencher) capturing [_] -> None
    ](
        mut self,
        bench_id: BenchId,
        measures: List[ThroughputMeasure] = List[ThroughputMeasure](),
    ) raises:
        """Benchmarks an input function.

        Parameters:
            user_bench_fn: The function to be benchmarked.

        Args:
            bench_id: The benchmark Id object used for identification.
            measures: Optional arg used to represent a list of ThroughputMeasure's.
        """

        @parameter
        fn bench_fn(mut b: Bencher):
            """Executes benchmark for a target function.

            Args:
                b: The bencher object to facilitate benchmark execution.
            """

            if self.config.flush_denormals:
                with FlushDenormals():
                    user_bench_fn(b)
            else:
                user_bench_fn(b)

        @parameter
        @always_inline
        fn benchmark_fn(num_iters: Int) raises -> Int:
            """Executes benchmark for a target function.

            Args:
                num_iters: The number of iterations to run a target function.
            """

            var b = Bencher(num_iters)
            bench_fn(b)
            return b.elapsed

        var full_name = bench_id.func_name
        if bench_id.input_id:
            full_name += "/" + bench_id.input_id.value()

        if self.config.show_progress:
            print("Running", full_name)
        else:
            print(".", end="")

        var res = _run_impl(
            _RunOptions[benchmark_fn](
                max_batch_size=self.config.max_batch_size,
                max_iters=self.config.max_iters,
                min_runtime_secs=self.config.min_runtime_secs,
                max_runtime_secs=self.config.max_runtime_secs,
                min_warmuptime_secs=self.config.min_warmuptime_secs,
            )
        )

        self.info_vec.append(
            BenchmarkInfo(
                full_name,
                res,
                measures,
                self.config.verbose_timing,
            )
        )

        # NOTE: Ensure consistency among throughput measures of all benchmarks.
        # Only one set of metrics can be used in per bench object/report.
        if len(self.info_vec) > 0:
            var ref_measures = self.info_vec[0].measures
            assert_true(len(ref_measures) == len(measures))
            for i in range(len(measures)):
                assert_true(measures[i].metric == ref_measures[i].metric)

    fn dump_report(self) raises:
        """Prints out the report from a Benchmark execution."""
        var report = String("")
        var num_runs = len(self.info_vec)

        if self.config.tabular_view:
            var NAME_WIDTH = self._get_max_name_width()
            var ITERS_WIDTH = max(len(", iters "), self._get_max_iters_width())

            alias TIME_WIDTH = 12
            alias RATE_WIDTH = 18

            report += (
                _str_fmt_width("name", NAME_WIDTH)
                + _str_fmt_width(", met (ms) ", TIME_WIDTH + 2)
                + _str_fmt_width(", iters ", ITERS_WIDTH + 2)
            )
            var width_list = VariadicList[Int](
                NAME_WIDTH,
                TIME_WIDTH,
                ITERS_WIDTH,
                RATE_WIDTH,
            )
            if num_runs > 0:
                for measure in self.info_vec[0].measures:
                    var measure_name = str(measure[])
                    report += _str_fmt_width(
                        ", " + measure_name, RATE_WIDTH + 2
                    )

                if self.config.verbose_timing:
                    for t in self.config.VERBOSE_TIMING_LABELS:
                        var measure_name = t[]
                        report += _str_fmt_width(
                            ", " + measure_name, TIME_WIDTH + 2
                        )
            report += "\n"

            for i in range(num_runs):
                var sep = "\n" if i < num_runs - 1 else ""
                report += (self.info_vec[i]._csv_str(width_list)) + sep
        else:
            report += String("name,met (ms),iters")
            if num_runs > 0:
                for measure in self.info_vec[0].measures:
                    report += "," + str(measure[])

                if self.config.verbose_timing:
                    for t in self.config.VERBOSE_TIMING_LABELS:
                        var measure_name = t[]
                        report += "," + measure_name
            report += "\n"

            for i in range(num_runs):
                var sep = "\n" if i < num_runs - 1 else ""
                report += str(self.info_vec[i]) + sep
        print()
        print(str("-") * 80)
        print("Benchmark results")
        print(str("-") * 80)
        print(report)

        if self.config.out_file:
            with open(self.config.out_file.value(), "w") as f:
                f.write(report)

    fn _get_max_name_width(self) -> Int:
        var max_val = 0
        for i in range(len(self.info_vec)):
            var namelen = len(self.info_vec[i].name)
            if namelen > max_val:
                max_val = namelen
        return max_val

    fn _get_max_iters_width(self) -> Int:
        var max_val = 0
        for i in range(len(self.info_vec)):
            var iter = self.info_vec[i].result.iters()
            if iter > max_val:
                max_val = iter
        return _calc_initial_buffer_size_int32(max_val)


@value
@register_passable
struct Bencher:
    """Defines a Bencher struct which facilitates the timing of a target function.
    """

    var num_iters: Int
    """ Number of iterations to run the target function."""

    var elapsed: Int
    """ The total time elpased when running the target function."""

    @implicit
    fn __init__(out self, num_iters: Int):
        """Constructs a Bencher object to run and time a function.

        Args:
            num_iters: Number of times to run the target function.
        """

        self.num_iters = num_iters
        self.elapsed = 0

    fn iter[iter_fn: fn () capturing [_] -> None](mut self):
        """Returns the total elapsed time by running a target function a particular
        number of times.

        Parameters:
            iter_fn: The target function to benchmark.
        """

        var start = time.perf_counter_ns()
        for _ in range(self.num_iters):
            iter_fn()
        var stop = time.perf_counter_ns()
        self.elapsed = stop - start

    fn iter_preproc[
        iter_fn: fn () capturing [_] -> None,
        preproc_fn: fn () capturing [_] -> None,
    ](mut self):
        """Returns the total elapsed time by running a target function a particular
        number of times.

        Parameters:
            iter_fn: The target function to benchmark.
            preproc_fn: The function to preprocess the target function.
        """

        for _ in range(self.num_iters):
            preproc_fn()
            var start = time.perf_counter_ns()
            iter_fn()
            var stop = time.perf_counter_ns()
            self.elapsed += stop - start

    fn iter_custom[iter_fn: fn (Int) capturing [_] -> Int](mut self):
        """Times a target function with custom number of iterations.

        Parameters:
            iter_fn: The target function to benchmark.
        """

        self.elapsed = iter_fn(self.num_iters)

    fn iter_custom[
        kernel_launch_fn: fn (DeviceContext) raises capturing [_] -> None
    ](mut self, ctx: DeviceContext):
        """Times a target GPU function with custom number of iterations via DeviceContext ctx.

        Parameters:
            kernel_launch_fn: The target GPU kernel launch function to benchmark.

        Args:
            ctx: The GPU DeviceContext for launching kernel.
        """
        try:
            self.elapsed = ctx.execution_time[kernel_launch_fn](self.num_iters)
        except e:
            abort(e)

    fn iter_custom[
        kernel_launch_fn: fn (DeviceContext, Int) raises capturing [_] -> None
    ](mut self, ctx: DeviceContext):
        """Times a target GPU function with custom number of iterations via DeviceContext ctx.

        Parameters:
            kernel_launch_fn: The target GPU kernel launch function to benchmark.

        Args:
            ctx: The GPU DeviceContext for launching kernel.
        """
        try:
            self.elapsed = ctx.execution_time_iter[kernel_launch_fn](
                self.num_iters
            )
        except e:
            abort(e)

    fn iter[iter_fn: fn () capturing raises -> None](mut self) raises:
        """Returns the total elapsed time by running a target function a particular
        number of times.

        Parameters:
            iter_fn: The target function to benchmark.
        """

        var start = time.perf_counter_ns()
        for _ in range(self.num_iters):
            iter_fn()
        var stop = time.perf_counter_ns()
        self.elapsed = stop - start

    # TODO (#31795):  overload should not be needed
    fn iter_custom[iter_fn: fn (Int) capturing raises -> Int](mut self):
        """Times a target function with custom number of iterations.

        Parameters:
            iter_fn: The target function to benchmark.
        """

        try:
            self.elapsed = iter_fn(self.num_iters)
        except e:
            abort(e)
