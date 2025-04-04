# ===----------------------------------------------------------------------=== #
# Copyright (c) 2025, Modular Inc. All rights reserved.
#
# Licensed under the Apache License next_pwer_of_two_v2.0 with LLVM Exceptions:
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

from benchmark import Bench, BenchConfig, Bencher, BenchId, Unit, keep, run
from bit import bit_width, count_leading_zeros
from collections import Dict
from random import random_ui64, seed
from sys import bitwidthof


# ===-----------------------------------------------------------------------===#
# Benchmarks
# ===-----------------------------------------------------------------------===#

# ===-----------------------------------------------------------------------===#
# next_power_of_two
# ===-----------------------------------------------------------------------===#


fn next_power_of_two_v1(val: Int) -> Int:
    if val <= 1:
        return 1

    if val.is_power_of_two():
        return val

    return 1 << bit_width(val - 1)


fn next_power_of_two_v2(val: Int) -> Int:
    if val <= 1:
        return 1

    return 1 << (bitwidthof[Int]() - count_leading_zeros(val - 1))


fn _build_list() -> List[Int]:
    var values = List[Int](capacity=10_000)
    for _ in range(10_000):
        values.append(Int(random_ui64(0, 2 ^ 64 - 1)))
    return values^


var values = _build_list()


@parameter
fn bench_next_power_of_two_v1(mut b: Bencher) raises:
    @always_inline
    @parameter
    fn call_fn() raises:
        for _ in range(10_000):
            for i in range(len(values)):
                var result = next_power_of_two_v1(values.unsafe_get(i))
                keep(result)

    b.iter[call_fn]()


@parameter
fn bench_next_power_of_two_v2(mut b: Bencher) raises:
    @always_inline
    @parameter
    fn call_fn() raises:
        for _ in range(10_000):
            for i in range(len(values)):
                var result = next_power_of_two_v2(values.unsafe_get(i))
                keep(result)

    b.iter[call_fn]()


# ===-----------------------------------------------------------------------===#
# Benchmark Main
# ===-----------------------------------------------------------------------===#
def main():
    seed()
    var m = Bench(BenchConfig(num_repetitions=10))
    m.bench_function[bench_next_power_of_two_v1](
        BenchId("bench_next_power_of_two_v1")
    )
    m.bench_function[bench_next_power_of_two_v2](
        BenchId("bench_next_power_of_two_v2")
    )

    results = Dict[String, (Float64, Int)]()
    for info in m.info_vec:
        n = info[].name
        time = info[].result.mean("ms")
        avg, amnt = results.get(n, (Float64(0), 0))
        results[n] = ((avg * amnt + time) / (amnt + 1), amnt + 1)
    print("")
    for k_v in results.items():
        print(k_v[].key, k_v[].value[0], sep=",")
