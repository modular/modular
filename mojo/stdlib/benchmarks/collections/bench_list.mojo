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

from collections import List

from benchmark import Bench, BenchConfig, Bencher, BenchId, black_box, keep


# ===-----------------------------------------------------------------------===#
# Benchmark Data
# ===-----------------------------------------------------------------------===#
fn make_int_list[size: Int]() -> List[Int]:
    var l = List[Int](capacity=size)
    for i in range(size):
        l.append(i)
    return l^


# ===-----------------------------------------------------------------------===#
# Benchmark List.pop (from end)
# ===-----------------------------------------------------------------------===#
@parameter
fn bench_list_pop_last[size: Int](mut b: Bencher) raises:
    """Benchmark popping the last element repeatedly."""

    @always_inline
    @parameter
    fn call_fn():
        var l = make_int_list[size]()
        for _ in range(size):
            keep(l.pop())

    b.iter[call_fn]()


# ===-----------------------------------------------------------------------===#
# Benchmark List.pop (from front)
# ===-----------------------------------------------------------------------===#
@parameter
fn bench_list_pop_front[size: Int](mut b: Bencher) raises:
    """Benchmark popping the first element repeatedly (worst case shift)."""

    @always_inline
    @parameter
    fn call_fn():
        var l = make_int_list[size]()
        for _ in range(size):
            keep(l.pop(0))

    b.iter[call_fn]()


# ===-----------------------------------------------------------------------===#
# Benchmark List.pop (from middle)
# ===-----------------------------------------------------------------------===#
@parameter
fn bench_list_pop_middle[size: Int](mut b: Bencher) raises:
    """Benchmark popping from the middle of the list."""

    @always_inline
    @parameter
    fn call_fn():
        var l = make_int_list[size]()
        for _ in range(size):
            keep(l.pop(len(l) // 2))

    b.iter[call_fn]()


# ===-----------------------------------------------------------------------===#
# Benchmark List.pop (String - non-trivially movable)
# ===-----------------------------------------------------------------------===#
@parameter
fn bench_list_pop_front_string[size: Int](mut b: Bencher) raises:
    """Benchmark popping String elements from front (non-trivial move)."""

    @always_inline
    @parameter
    fn call_fn():
        var l = List[String](capacity=size)
        for i in range(size):
            l.append(String("item_") + String(i))
        for _ in range(size):
            keep(l.pop(0))

    b.iter[call_fn]()


# ===-----------------------------------------------------------------------===#
# Benchmark Main
# ===-----------------------------------------------------------------------===#
def main():
    var m = Bench(BenchConfig(num_repetitions=10))
    comptime sizes = (100, 1000, 10_000)

    @parameter
    for i in range(len(sizes)):
        comptime size = sizes[i]

        # Pop from end (no shift needed)
        m.bench_function[bench_list_pop_last[size]](
            BenchId(String("bench_list_pop_last[", size, "]"))
        )

        # Pop from front (maximum shift - benefits most from memcpy)
        m.bench_function[bench_list_pop_front[size]](
            BenchId(String("bench_list_pop_front[", size, "]"))
        )

        # Pop from middle
        m.bench_function[bench_list_pop_middle[size]](
            BenchId(String("bench_list_pop_middle[", size, "]"))
        )

        # Pop String from front (non-trivially movable, uses loop)
        m.bench_function[bench_list_pop_front_string[size]](
            BenchId(String("bench_list_pop_front_string[", size, "]"))
        )

    print(m)
