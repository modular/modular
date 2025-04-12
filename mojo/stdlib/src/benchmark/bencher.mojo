# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

import time
from collections import Optional, Dict
from collections.string.string import _calc_initial_buffer_size_int32
from os import abort
from pathlib import Path
from sys.arg import argv

from gpu.host import DeviceContext
from stdlib.builtin.file import FileHandle
from stdlib.builtin.io import _snprintf
from testing import assert_true
from collections.string import StaticString, StringSlice

from utils.numerics import FlushDenormals

from .benchmark import *
from .benchmark import _run_impl, _RunOptions


@value
struct BenchMetric(CollectionElement, Stringable, Writable):
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
        return String.write(self)

    fn write_to[W: Writer](self, mut writer: W):
        """Formats this BenchMetric to the provided Writer.

        Parameters:
            W: A type conforming to the Writable trait.

        Args:
            writer: The object to write to.
        """
        writer.write(self.name, " (", self.unit, ")")

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

        alias sep = StaticString("-") * 80 + "\n"
        var err = String(
            "\n",
            sep,
            sep,
            "Couldn't match metric [" + name + "]\n",
            "Available throughput metrics (case-insensitive) in the list:\n",
        )
        for m in metric_list:
            err += String("    metric: [" + m[].name.lower(), "]\n")
        err += String(
            sep, sep, "[ERROR]: metric [", name, "] is NOT supported!\n"
        )
        raise Error(err)


@value
struct ThroughputMeasure(CollectionElement):
    """Records a throughput metric of metric BenchMetric and value."""

    var metric: BenchMetric
    """Type of throughput metric."""
    var value: Int
    """Measured count of throughput metric."""

    fn __init__(
        out self,
        name: String,
        value: Int,
        reference: List[BenchMetric] = BenchMetric.DEFAULTS,
    ) raises:
        """Creates a `ThroughputMeasure` based on metric's name.

        Args:
            name: The name of BenchMetric in its corresponding reference.
            value: The measured value to assign to this metric.
            reference: List of BenchMetrics that contains this metric.

        Example:
            For the default bench metrics `BenchMetric.DEFAULTS` the
            following are equivalent:
                - `ThroughputMeasure(BenchMetric.fmas, 1024)`
                - `ThroughputMeasure("fmas", 1024)`
                - `ThroughputMeasure("fmas", 1024, BenchMetric.DEFAULTS)`
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
        return String(self.metric)

    fn write_to[W: Writer](self, mut writer: W):
        """Formats this ThroughputMeasure to the provided Writer.

        Parameters:
            W: A type conforming to the Writable trait.

        Args:
            writer: The object to write to.
        """
        return writer.write(self.metric)

    fn compute(self, elapsed_sec: Float64) -> Float64:
        """Computes throughput rate for this metric per unit of time (second).

        Args:
            elapsed_sec: Elapsed time measured in seconds.

        Returns:
            The throughput values as a floating point 64.
        """
        # TODO: do we need support other units of time (ms, ns)?
        return (self.value) * 1e-9 / elapsed_sec


@value
struct Format(Writable, Stringable):
    """Defines a format for the benchmark output when printing or writing to a
    file.
    """

    alias csv = StaticString("csv")
    """Comma separated values with no alignment."""
    alias tabular = StaticString("tabular")
    """Comma separated values with dynamically aligned columns."""
    alias table = StaticString("table")
    """Table format with dynamically aligned columns."""

    var value: StaticString
    """The format to print results."""

    @implicit
    fn __init__(out self, value: StringSlice):
        """Constructs a Format object from a string.

        Args:
            value: The format to print results.
        """
        if value == Format.csv:
            self.value = Format.csv
        elif value == Format.tabular:
            self.value = Format.tabular
        elif value == Format.table:
            self.value = Format.table
        else:
            self.value = ""
            var valid_formats = String(
                " valid formats: ",
                Format.csv,
                ", ",
                Format.tabular,
                ", ",
                Format.table,
            )
            abort("Invalid format option: ", value, valid_formats)

    fn __str__(self) -> String:
        """Returns the string representation of the format.

        Returns:
            The string representation of the format.
        """
        return String(self.value)

    fn write_to[W: Writer](self, mut writer: W):
        """Writes the format to a writer.

        Parameters:
            W: A type conforming to the Writable trait.

        Args:
            writer: The writer to write the `Format` to.
        """
        writer.write(self.value)

    fn __eq__(self, other: Self) -> Bool:
        """Checks if two Format objects are equal.

        Args:
            other: The `Format` to compare with.

        Returns:
            True if the two `Format` objects are equal, false otherwise.
        """
        return self.value == other.value


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
    """If True, print progress of each benchmark."""
    var format: Format
    """The format to print results. (default: "table")."""
    var out_file_format: Format
    """The format to write out the file with `dump_file` (default: "csv")."""
    var verbose_timing: Bool
    """Whether to print verbose timing results."""
    var verbose_metric_names: Bool
    """If True print the metric name and unit, else print the unit only."""
    alias VERBOSE_TIMING_LABELS = List[String](
        "min (ms)", "mean (ms)", "max (ms)", "duration (ms)"
    )
    """Labels to print verbose timing results."""

    # TODO: to add median and stddev to verbose-timing

    fn __init__(
        out self,
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
        self.format = Format.table
        self.out_file_format = Format.csv
        self.verbose_timing = False
        self.verbose_metric_names = True

        @parameter
        fn argparse() raises:
            """Parse cmd line args to define benchmark configuration."""

            var args = argv()
            var i = 1
            while i < len(args):
                if args[i] == "-o":
                    if i + 1 >= len(args):
                        raise Error("Missing value for -o option")
                    self.out_file = Path(args[i + 1])
                    i += 2
                elif args[i] == "-r":
                    if i + 1 >= len(args):
                        raise Error("Missing value for -r option")
                    self.num_repetitions = Int(args[i + 1])
                    i += 2
                elif args[i] == "--format":
                    if i + 1 >= len(args):
                        raise Error("Missing value for --format option")
                    self.format = Format(args[i + 1])
                    i += 2
                elif args[i] == "--no-progress":
                    self.show_progress = False
                    i += 1
                elif args[i] == "--verbose":
                    self.verbose_timing = True
                    i += 1
                # TODO: add an arg for bench batchsize
                else:
                    i += 1

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
struct BenchmarkInfo(CollectionElement):
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
        out self,
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


struct Bench(Writable):
    """Constructs a Benchmark object, used for running multiple benchmarks
    and comparing the results.

    Example:

    ```mojo
    from benchmark import (
        Bench,
        BenchConfig,
        Bencher,
        BenchId,
        ThroughputMeasure,
        BenchMetric,
        Format,
    )
    from utils import IndexList
    from gpu.host import DeviceContext
    from pathlib import Path

    fn example_kernel():
        print("example_kernel")

    var shape = IndexList[2](1024, 1024)
    var bench = Bench(BenchConfig(max_iters=100))

    @parameter
    @always_inline
    fn example(mut b: Bencher, shape: IndexList[2]) capturing raises:
        @parameter
        @always_inline
        fn kernel_launch(ctx: DeviceContext) raises:
            ctx.enqueue_function[example_kernel](
                grid_dim=shape[0], block_dim=shape[1]
            )

        var bench_ctx = DeviceContext()
        b.iter_custom[kernel_launch](bench_ctx)

    bench.bench_with_input[IndexList[2], example](
        BenchId("top_k_custom", "gpu"),
        shape,
        ThroughputMeasure(
            BenchMetric.elements, shape.flattened_length()
        ),
        ThroughputMeasure(
            BenchMetric.flops, shape.flattened_length() * 3 # number of ops
        ),
    )
    # Add more benchmarks like above to compare results

    # Pretty print in table format
    print(bench)

    # Dump report to csv file
    bench.config.out_file = Path("out.csv")
    bench.dump_report()

    # Print in tabular csv format
    bench.config.format = Format.tabular
    print(bench)
    ```

    You can pass arguments when running a program that makes use of `Bench`:

    ```sh
    mojo benchmark.mojo -o out.csv -r 10
    ```

    This will repeat the benchmarks 10 times and write the output to `out.csv`
    in csv format.
    """

    var config: BenchConfig
    """Constructs a Benchmark object based on specific configuration and mode."""
    var mode: Mode
    """Benchmark mode object representing benchmark or test mode."""
    var info_vec: List[BenchmarkInfo]
    """A list containing the benchmark info."""

    fn __init__(
        out self,
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

    fn dump_report(mut self) raises:
        """Prints out the report from a Benchmark execution. If
        `Bench.config.out_file` is set, it will also write the output in the format
        set in `out_file_format` to the file defined in `out_file`."""
        print(self)

        if self.config.out_file:
            var orig_format = self.config.format
            self.config.format = self.config.out_file_format
            with open(self.config.out_file.value(), "w") as f:
                f.write(self)
            self.config.format = orig_format

    fn pad(self, width: Int, string: String) -> String:
        """Pads a string to a given width.

        Args:
            width: The width to pad the string to.
            string: The string to pad.

        Returns:
            A string padded to the given width.
        """
        if self.config.format == Format.csv:
            return ""
        return " " * (width - len(string))

    fn __str__(self) -> String:
        """Returns a string representation of the benchmark results.

        Returns:
            A string representing the benchmark results.
        """
        return String.write(self)

    fn write_to[W: Writer](self, mut writer: W):
        """Writes the benchmark results to a writer.

        Parameters:
            W: A type conforming to the Writer trait.

        Args:
            writer: The writer to write to.
        """
        alias BENCH_LABEL = "name"
        alias ITERS_LABEL = "iters"
        alias MET_LABEL = "met (ms)"

        var name_width = self._get_max_name_width(BENCH_LABEL)
        var iters_width = self._get_max_iters_width(ITERS_LABEL)
        var timing_widths = self._get_max_timing_widths(MET_LABEL)
        var metrics = self._get_metrics()

        # +3 for 2x " | " characters and one for the first "|"
        var total_width = name_width + iters_width + 7

        # Calculate the total width of the table for line separators
        # +3 for " | " characters
        if self.config.format == Format.table and len(self.info_vec) > 0:
            for metric in metrics:
                try:
                    total_width += metrics[metric[]].max_width + 3
                except e:
                    abort(e)
            if self.config.verbose_timing:
                for timing_width in timing_widths:
                    total_width += timing_width[] + 3
            else:
                total_width += timing_widths[0] + 3

        var sep: StaticString
        if self.config.format == Format.table:
            sep = " | "
        elif self.config.format == Format.tabular:
            sep = ", "
        else:
            sep = ","

        var first_sep = "| " if self.config.format == Format.table else StaticString(
            ""
        )
        var line_sep = "-" * total_width

        if self.config.format == Format.table:
            writer.write(line_sep, "\n")

        writer.write(first_sep, BENCH_LABEL, self.pad(name_width, BENCH_LABEL))
        writer.write(sep, MET_LABEL, self.pad(timing_widths[0], MET_LABEL))
        writer.write(sep, ITERS_LABEL, self.pad(iters_width, ITERS_LABEL))

        # Return early if no runs were benchmarked
        if len(self.info_vec) == 0:
            if self.config.format == Format.table:
                writer.write(" |\n", line_sep, "\nNo benchmarks recorded...")
            writer.write("\n")
            return

        # Write the metrics labels
        for metric in metrics:
            name = metric[]
            writer.write(sep, name)
            try:
                writer.write(self.pad(metrics[name].max_width, name))
            except e:
                abort(e)

        # Write the timeing labels
        if self.config.verbose_timing:
            var labels = self.config.VERBOSE_TIMING_LABELS
            # skip the met label
            for i in range(len(labels)):
                writer.write(sep, labels[i])
                writer.write(self.pad(timing_widths[i + 1], labels[i]))

        if self.config.format == Format.table:
            writer.write(" |\n", line_sep)
        writer.write("\n")

        # Loop through the runs and write out the table rows
        var runs = self.info_vec
        for i in range(len(runs)):
            var run = runs[i]
            var result = run.result

            # TODO: remove when kbench adds the spec column
            if self.config.format == Format.csv:
                name = String('"', run.name, '"')
            else:
                name = run.name

            writer.write(first_sep, name, self.pad(name_width, name))

            # TODO: Move met (ms) to the end of the table to align with verbose
            # timing, don't repeat `Mean (ms)`, and make sure it works with
            # kernel benchmarking.
            var met = result.mean(unit=Unit.ms)
            writer.write(sep, met, self.pad(timing_widths[0], String(met)))

            var iters_pad = self.pad(iters_width, String(run.result.iters()))
            writer.write(sep, run.result.iters(), iters_pad)

            for metric in metrics:
                var name = metric[]
                try:
                    var rates = metrics[name].rates
                    var max_width = metrics[name].max_width
                    if i not in rates:
                        writer.write(sep, "N/A", self.pad(max_width, "N/A"))
                    else:
                        var rate = rates[i]
                        writer.write(
                            sep, rate, self.pad(max_width, String(rate))
                        )
                except e:
                    abort(e)

            if self.config.verbose_timing:
                var min = result.min(unit=Unit.ms)
                var max = result.max(unit=Unit.ms)
                var dur = result.duration(unit=Unit.ms)
                writer.write(sep, min, self.pad(timing_widths[1], String(min)))
                writer.write(sep, met, self.pad(timing_widths[2], String(met)))
                writer.write(sep, max, self.pad(timing_widths[3], String(max)))
                writer.write(sep, dur, self.pad(timing_widths[4], String(dur)))

            if self.config.format == Format.table:
                writer.write(" |")

            writer.write("\n")

        if self.config.format == Format.table:
            writer.write(line_sep, "\n")

    fn _get_max_name_width(self, label: StaticString) -> Int:
        var max_val = len(label)
        for i in range(len(self.info_vec)):
            var namelen = len(String(self.info_vec[i].name))
            max_val = max(max_val, namelen)
        return max_val

    fn _get_max_iters_width(self, label: StaticString) -> Int:
        var max_val = len(label)
        for i in range(len(self.info_vec)):
            var iters = self.info_vec[i].result.iters()
            max_val = max(max_val, len(String(iters)))
        return max_val

    fn _get_metrics(self) -> Dict[String, _Metric]:
        var metrics = Dict[String, _Metric]()
        var runs = len(self.info_vec)
        for i in range(runs):
            var run = self.info_vec[i]
            for j in range(len(run.measures)):
                var measure = run.measures[j]
                var rate = measure.compute(run.result.mean(unit=Unit.s))
                var width = len(String(rate))
                var name = measure.metric.unit
                if self.config.verbose_metric_names:
                    name = String(measure.metric)
                if name not in metrics:
                    metrics[name] = _Metric(
                        max(width, len(name)), Dict[Int, Float64]()
                    )
                    try:
                        metrics[name].rates[i] = rate
                    except e:
                        abort(e)
                else:
                    try:
                        metrics[name].max_width = max(
                            width, metrics[name].max_width
                        )
                        metrics[name].rates[i] = rate
                    except e:
                        abort(e)
        return metrics

    fn _get_max_timing_widths(self, met_label: StaticString) -> List[Int]:
        # If label is larger than any value, will pad to the label length

        var max_met = len(met_label)
        var max_min = len(self.config.VERBOSE_TIMING_LABELS[0])
        var max_mean = len(self.config.VERBOSE_TIMING_LABELS[1])
        var max_max = len(self.config.VERBOSE_TIMING_LABELS[2])
        var max_dur = len(self.config.VERBOSE_TIMING_LABELS[3])
        for i in range(len(self.info_vec)):
            # TODO: Move met (ms) to the end of the table to align with verbose
            # timing, don't repeat `Mean (ms)`, and make sure it works with
            # kernel benchmarking.
            var result = self.info_vec[i].result
            var mean_len = len(String(result.mean(unit=Unit.ms)))
            # met == mean execution time == mean
            max_met = max(max_met, mean_len)

            max_min = max(max_min, len(String(result.min(unit=Unit.ms))))
            max_mean = max(max_mean, mean_len)
            max_max = max(max_max, len(String(result.max(unit=Unit.ms))))
            max_dur = max(max_dur, len(String(result.duration(unit=Unit.ms))))
        return List[Int](max_met, max_min, max_mean, max_max, max_dur)


@value
struct _Metric(CollectionElement):
    var max_width: Int
    var rates: Dict[Int, Float64]


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

    fn iter_custom_multicontext[
        kernel_launch_fn: fn () raises capturing [_] -> None
    ](mut self, ctxs: List[DeviceContext]):
        """Times a target GPU function with custom number of iterations via DeviceContext ctx.

        Parameters:
            kernel_launch_fn: The target GPU kernel launch function to benchmark.

        Args:
            ctxs: The list of GPU DeviceContext's for launching kernel.
        """
        try:
            # Find the max elapsed time across the list of GPU DeviceContext's.
            self.elapsed = 0
            for i in range(len(ctxs)):
                self.elapsed = max(
                    self.elapsed,
                    ctxs[i].execution_time[kernel_launch_fn](self.num_iters),
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
