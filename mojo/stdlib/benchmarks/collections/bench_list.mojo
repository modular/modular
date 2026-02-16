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
# Benchmark List.pop() from end
# ===-----------------------------------------------------------------------===#
@parameter
fn bench_list_pop_last[size: Int](mut b: Bencher) raises:
    """Pop all elements from the end of a list."""

    @always_inline
    @parameter
    fn setup() -> List[Int]:
        return make_int_list[size]()

    @always_inline
    @parameter
    fn bench(mut l: List[Int]):
        for _ in range(size):
            keep(black_box(l).pop())

    b.iter_with_setup[List[Int], setup, bench]()


# ===-----------------------------------------------------------------------===#
# Benchmark List.pop() from front
# ===-----------------------------------------------------------------------===#
@parameter
fn bench_list_pop_front[size: Int](mut b: Bencher) raises:
    """Pop all elements from the front of a list."""

    @always_inline
    @parameter
    fn setup() -> List[Int]:
        return make_int_list[size]()

    @always_inline
    @parameter
    fn bench(mut l: List[Int]):
        for _ in range(size):
            keep(black_box(l).pop(0))

    b.iter_with_setup[List[Int], setup, bench]()


# ===-----------------------------------------------------------------------===#
# Benchmark List.pop() from middle
# ===-----------------------------------------------------------------------===#
@parameter
fn bench_list_pop_middle[size: Int](mut b: Bencher) raises:
    """Pop all elements from the middle of a list."""

    @always_inline
    @parameter
    fn setup() -> List[Int]:
        return make_int_list[size]()

    @always_inline
    @parameter
    fn bench(mut l: List[Int]):
        for _ in range(size):
            keep(black_box(l).pop(len(black_box(l)) // 2))

    b.iter_with_setup[List[Int], setup, bench]()


# ===-----------------------------------------------------------------------===#
# Benchmark Main
# ===-----------------------------------------------------------------------===#
def main():
    var m = Bench(BenchConfig(num_repetitions=5))
    comptime sizes = (10, 100, 1000, 10_000)

    @parameter
    for i in range(len(sizes)):
        comptime size = sizes[i]
        m.bench_function[bench_list_pop_last[size]](
            BenchId(String("bench_list_pop_last[", size, "]"))
        )
        m.bench_function[bench_list_pop_front[size]](
            BenchId(String("bench_list_pop_front[", size, "]"))
        )
        m.bench_function[bench_list_pop_middle[size]](
            BenchId(String("bench_list_pop_middle[", size, "]"))
        )

    m.dump_report()
