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
"""Benchmarks signed floor division by a compile-time constant on the CPU.

Modeled on `div_bench.cpp` from https://github.com/benjoffe/fastdiv. Each case
divides 2^16 random full-range dividends by a constant, either independently
(`tput`) or as one dependent chain (`lat`), and compares two lowerings:

- `benjoffe`: `x // d`, which the compiler lowers to Ben Joffe's multiply-high
  sequence.
- `generic`: the truncating division plus sign fixup that `pop.floordiv`
  lowered to before, written out so LLVM sees the same IR for it.

```
./bazelw test //Mojo/stdlib/benchmarks:math/bench_floordiv.mojo.bench \
  --test_output=streamed
```
"""

from std.benchmark import (
    Bench,
    BenchConfig,
    BenchId,
    BenchMetric,
    ThroughputMeasure,
)
from std.random import random_ui64, seed
from std.sys import bit_width_of
from std.sys._assembly import inlined_assembly
from std.time import perf_counter_ns

comptime COUNT = 1 << 16


# Register-only barriers, like `OPAQUE_IN` and `SINK` in div_bench.cpp: the
# compiler cannot see through them, and unlike `keep` they do not force the
# value through memory.
@inline(.always)
def opaque[dtype: DType](x: Scalar[dtype]) -> Scalar[dtype]:
    return inlined_assembly[
        "", Scalar[dtype], constraints="=r,0", has_side_effect=True
    ](x)


@inline(.always)
def sink[dtype: DType](x: Scalar[dtype]):
    inlined_assembly["", NoneType, constraints="r", has_side_effect=True](x)


@inline(.always)
def floordiv_generic[
    dtype: DType, d: Scalar[dtype]
](x: Scalar[dtype]) -> Scalar[dtype]:
    var q = x / d
    var sign = (x ^ d) >> Scalar[dtype](bit_width_of[dtype]() - 1)
    return q + (q * d).eq(x).select(Scalar[dtype](0), sign)


@inline(.always)
def divide[
    dtype: DType, d: Scalar[dtype], benjoffe: Bool
](x: Scalar[dtype]) -> Scalar[dtype]:
    comptime if benjoffe:
        return x // d
    else:
        return floordiv_generic[dtype, d](x)


@inline(.never)
def run_throughput[
    dtype: DType, d: Scalar[dtype], benjoffe: Bool
](inputs: Span[Scalar[dtype], _]):
    for i in range(len(inputs)):
        sink(divide[dtype, d, benjoffe](opaque(inputs[i])))


@inline(.never)
def run_latency[
    dtype: DType, d: Scalar[dtype], benjoffe: Bool
](inputs: Span[Scalar[dtype], _]):
    var result = Scalar[dtype](0)
    for i in range(len(inputs)):
        result = divide[dtype, d, benjoffe](inputs[i] ^ result)
        sink(result)


def make_inputs[dtype: DType]() -> List[Scalar[dtype]]:
    seed(0xDEADBEEF)
    var inputs = List[Scalar[dtype]](capacity=COUNT)
    for _ in range(COUNT):
        inputs.append(random_ui64(0, UInt64.MAX).cast[dtype]())
    return inputs^


def check[dtype: DType, d: Scalar[dtype]](inputs: List[Scalar[dtype]]) raises:
    var edges: List[Scalar[dtype]] = [
        Scalar[dtype].MIN,
        Scalar[dtype].MIN + 1,
        -2,
        -1,
        0,
        1,
        2,
        Scalar[dtype].MAX - 1,
        Scalar[dtype].MAX,
        d - 1,
        d,
        d + 1,
        -d - 1,
        -d,
        -d + 1,
    ]
    for values in [edges.copy(), inputs.copy()]:
        for x in values:
            var got = divide[dtype, d, True](opaque(x))
            var want = divide[dtype, d, False](opaque(x))
            if got != want:
                raise Error(
                    t"floordiv mismatch: {x} // {d} gave {got}, want {want}"
                )


def add_divisor[
    dtype: DType, d: Scalar[dtype]
](mut m: Bench, inputs: List[Scalar[dtype]]) raises:
    check[dtype, d](inputs)
    comptime for latency in [False, True]:
        comptime for benjoffe in [False, True]:

            def call() {imm inputs}:
                comptime if latency:
                    run_latency[dtype, d, benjoffe](inputs)
                else:
                    run_throughput[dtype, d, benjoffe](inputs)

            var name = String(
                "floordiv_",
                "lat" if latency else "tput",
                "_",
                "benjoffe" if benjoffe else "generic",
            )
            m.bench_function(
                call,
                BenchId(name, String(dtype, "/d", d)),
                [ThroughputMeasure(BenchMetric.elements, COUNT)],
            )


def add_dtype[dtype: DType](mut m: Bench) raises:
    var inputs = make_inputs[dtype]()
    # The divisors div_bench.cpp sweeps, plus negatives and each mode of the
    # lowering at 64 bits (round-up 7, round-down 5, sign-selected 10).
    comptime for d in [
        3,
        5,
        7,
        10,
        13,
        100,
        360,
        86400,
        1000000007,
        -3,
        -7,
        -10,
        -100,
        -86400,
    ]:
        comptime if d <= Int(Scalar[dtype].MAX) and d >= Int(Scalar[dtype].MIN):
            add_divisor[dtype, Scalar[dtype](d)](m, inputs)


def main() raises:
    # Each pass takes microseconds, so without a floor on the runtime the first
    # cases are timed before the core has reached its full clock.
    var m = Bench(BenchConfig(num_repetitions=3, min_runtime_secs=0.1))
    var warmup = make_inputs[DType.int64]()
    var start = perf_counter_ns()
    while perf_counter_ns() - start < 500_000_000:
        run_throughput[DType.int64, 7, False](warmup)
    add_dtype[DType.int64](m)
    add_dtype[DType.int32](m)
    m.dump_report()
