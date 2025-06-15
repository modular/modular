# ===----------------------------------------------------------------------=== #
# Copyright (c) 2025, Modular Inc. All rights reserved.
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
# RUN: %mojo-no-debug %s -t
# NOTE: to test changes on the current branch using run-benchmarks.sh, remove
# the -t flag. Remember to replace it again before pushing any code.

from random import random_ui64, seed
from sys import bitwidthof
from sys.intrinsics import likely, unlikely

from benchmark import Bench, BenchConfig, Bencher, BenchId, Unit, keep, run
from bit import bit_width, count_leading_zeros

# ===-----------------------------------------------------------------------===#
# Benchmarks
# ===-----------------------------------------------------------------------===#

# ===-----------------------------------------------------------------------===#
# next_power_of_two
# ===-----------------------------------------------------------------------===#


fn next_power_of_two_int_v1(val: Int) -> Int:
    if val <= 1:
        return 1

    if val.is_power_of_two():
        return val

    return 1 << bit_width(val - 1)


fn next_power_of_two_int_v2(val: Int) -> Int:
    if val <= 1:
        return 1

    return 1 << (bitwidthof[Int]() - count_leading_zeros(val - 1))


fn next_power_of_two_int_v3(val: Int) -> Int:
    var v = Scalar[DType.index](val)
    return Int(
        (v <= 1)
        .select(1, 1 << (bitwidthof[Int]() - count_leading_zeros(v - 1)))
        .__index__()
    )


fn next_power_of_two_int_v4(val: Int) -> Int:
    return 1 << (
        (bitwidthof[Int]() - count_leading_zeros(val - 1))
        & -Int(likely(val > 1))
    )


fn next_power_of_two_uint_v1(val: UInt) -> UInt:
    if unlikely(val == 0):
        return 1

    return 1 << (bitwidthof[UInt]() - count_leading_zeros(val - 1))


fn next_power_of_two_uint_v2(val: UInt) -> UInt:
    var v = Scalar[DType.index](val)
    return UInt(
        (v == 0)
        .select(1, 1 << (bitwidthof[UInt]() - count_leading_zeros(v - 1)))
        .__index__()
    )


fn next_power_of_two_uint_v3(val: UInt) -> UInt:
    return 1 << (
        bitwidthof[UInt]() - count_leading_zeros(val - UInt(likely(val > 0)))
    )


fn next_power_of_two_uint_v4(val: UInt) -> UInt:
    return 1 << (
        bitwidthof[UInt]()
        - count_leading_zeros((val | UInt(unlikely(val == 0))) - 1)
    )


fn _build_list[start: Int, stop: Int]() -> List[Int]:
    var values = List[Int](capacity=10_000)
    for _ in range(10_000):
        values.append(Int(random_ui64(start, stop)))
    return values^


alias width = bitwidthof[Int]()


@parameter
fn bench_next_power_of_two_int[func: fn (Int) -> Int](mut b: Bencher) raises:
    var values = _build_list[0, 2**width - 1]()

    @always_inline
    @parameter
    fn call_fn() raises:
        for _ in range(10_000):
            for i in range(len(values)):
                var result = func(values.unsafe_get(i))
                keep(result)

    b.iter[call_fn]()


@parameter
fn bench_next_power_of_two_uint[func: fn (UInt) -> UInt](mut b: Bencher) raises:
    var values = _build_list[0, 2**width - 1]()

    @always_inline
    @parameter
    fn call_fn() raises:
        for _ in range(10_000):
            for i in range(len(values)):
                var result = func(values.unsafe_get(i))
                keep(result)

    b.iter[call_fn]()


# ===-----------------------------------------------------------------------===#
# Benchmark Main
# ===-----------------------------------------------------------------------===#
def main():
    seed()
    var m = Bench(BenchConfig(num_repetitions=10))
    m.bench_function[bench_next_power_of_two_int[next_power_of_two_int_v1]](
        BenchId("bench_next_power_of_two_int_v1")
    )
    m.bench_function[bench_next_power_of_two_int[next_power_of_two_int_v2]](
        BenchId("bench_next_power_of_two_int_v2")
    )
    m.bench_function[bench_next_power_of_two_int[next_power_of_two_int_v3]](
        BenchId("bench_next_power_of_two_int_v3")
    )
    m.bench_function[bench_next_power_of_two_int[next_power_of_two_int_v4]](
        BenchId("bench_next_power_of_two_int_v4")
    )
    m.bench_function[bench_next_power_of_two_uint[next_power_of_two_uint_v1]](
        BenchId("bench_next_power_of_two_uint_v1")
    )
    m.bench_function[bench_next_power_of_two_uint[next_power_of_two_uint_v2]](
        BenchId("bench_next_power_of_two_uint_v2")
    )
    m.bench_function[bench_next_power_of_two_uint[next_power_of_two_uint_v3]](
        BenchId("bench_next_power_of_two_uint_v3")
    )
    m.bench_function[bench_next_power_of_two_uint[next_power_of_two_uint_v4]](
        BenchId("bench_next_power_of_two_uint_v4")
    )

    results = Dict[String, (Float64, Int)]()
    for info in m.info_vec:
        n = info.name
        time = info.result.mean("ms")
        avg, amnt = results.get(n, (Float64(0), 0))
        results[n] = ((avg * amnt + time) / (amnt + 1), amnt + 1)
    print("")
    for k_v in results.items():
        print(k_v.key, k_v.value[0], sep=",")
