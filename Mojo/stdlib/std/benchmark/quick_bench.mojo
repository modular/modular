# ===----------------------------------------------------------------------=== #
# Copyright (c) 2026, Modular Inc. All rights reserved.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions:
# https://llvm.org/LICENSE.txt
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ===----------------------------------------------------------------------=== #
"""Provides simplified benchmarking interface with automatic boilerplate handling.

This module implements the `QuickBench` type, which wraps the full `Bench`
infrastructure to provide a simpler interface for common benchmarking tasks. It
automatically handles `Bencher` setup and the `keep()` calls needed to prevent
optimization.
"""

from .compiler import keep
from .bencher import Bench, Bencher, BenchId, ThroughputMeasure


struct QuickBench:
    """Defines a struct to facilitate benchmarking and avoiding `Bencher` boilerplate.
    """

    var m: Bench
    """Bench object to collect the results."""

    @inline(.always)
    def __init__(out self) raises:
        """Initializes the `Bench` object.

        Raises:
            If the operation fails.
        """
        self.m = Bench()

    @inline(.always)
    def dump_report(mut self) raises:
        """Prints out the report from a Benchmark execution collected in Bench object.

        Raises:
            If the operation fails.
        """
        self.m.dump_report()

    @inline(.always)
    def run[
        T_out: TrivialRegisterPassable, //
    ](
        mut self,
        func: Some[def() -> T_out],
        *,
        bench_id: BenchId,
        measures: List[ThroughputMeasure] = {},
    ) raises:
        """Benchmarks `func`, keeping its result alive so the call is not
        optimized away.

        To benchmark a function that takes arguments, wrap the call in a
        closure that captures them.

        Parameters:
            T_out: Output type of func.

        Args:
            func: The function to be benchmarked (run in benchmark iterations).
            bench_id: The benchmark Id object used for identification.
            measures: Optional arg used to represent a list of ThroughputMeasure's.

        Raises:
            If the operation fails.

        Examples:

        ```mojo
        from std.benchmark import BenchId, QuickBench
        from std.math import exp

        var qb = QuickBench()
        var x: Float32 = 1.0
        qb.run(lambda () {x} -> Float32: exp(x), bench_id=BenchId("exp"))
        qb.dump_report()
        ```
        """

        @inline(.always)
        def bench_iter(mut b: Bencher) {imm}:
            @inline(.always)
            def call_func() {imm}:
                var x = func()
                keep(x)

            b.iter(call_func)

        self.m.bench_function(bench_iter, bench_id, measures=measures)

    # TODO(MOCO-5296): Remove once inferring `T_out` from a closure that
    # returns `None` no longer fails to elaborate; the overload above then
    # covers it.
    @inline(.always)
    def run(
        mut self,
        func: Some[def() -> None],
        *,
        bench_id: BenchId,
        measures: List[ThroughputMeasure] = {},
    ) raises:
        """Benchmarks `func`, which returns nothing.

        Args:
            func: The function to be benchmarked (run in benchmark iterations).
            bench_id: The benchmark Id object used for identification.
            measures: Optional arg used to represent a list of ThroughputMeasure's.

        Raises:
            If the operation fails.
        """

        @inline(.always)
        def bench_iter(mut b: Bencher) {imm}:
            b.iter(func)

        self.m.bench_function(bench_iter, bench_id, measures=measures)
