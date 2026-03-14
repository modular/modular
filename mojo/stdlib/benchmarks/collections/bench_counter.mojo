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

from std.collections import Counter

from std.benchmark import Bench, BenchConfig, Bencher, BenchId, black_box, keep


# ===-----------------------------------------------------------------------===#
# Benchmark Data
# ===-----------------------------------------------------------------------===#
def make_counter[size: Int]() -> Counter[Int]:
    """Build a Counter with `size` distinct keys, count[i] = i + 1."""
    var c = Counter[Int]()
    for i in range(size):
        c[i] = i + 1
    return c^


# ===-----------------------------------------------------------------------===#
# Benchmark most_common — heap path (n << total)
# ===-----------------------------------------------------------------------===#
@parameter
def bench_most_common_small_n[total: Int, n: Int](mut b: Bencher) raises:
    """Benchmark most_common(n) via the heap path (n < total // 2)."""
    var c = make_counter[total]()

    @always_inline
    def call_fn() unified {read}:
        var result = black_box(c).most_common(UInt(black_box(n)))
        keep(result)

    b.iter(call_fn)


# ===-----------------------------------------------------------------------===#
# Benchmark most_common — full sort path (n >= total // 2)
# ===-----------------------------------------------------------------------===#
@parameter
def bench_most_common_large_n[total: Int, n: Int](mut b: Bencher) raises:
    """Benchmark most_common(n) via the full sort path (n >= total // 2)."""
    var c = make_counter[total]()

    @always_inline
    def call_fn() unified {read}:
        var result = black_box(c).most_common(UInt(black_box(n)))
        keep(result)

    b.iter(call_fn)


# ===-----------------------------------------------------------------------===#
# Benchmark Main
# ===-----------------------------------------------------------------------===#
def main() raises:
    var m = Bench(BenchConfig(num_repetitions=10))

    # total=100: heap path (n=5), full sort path (n=80)
    m.bench_function[bench_most_common_small_n[100, 5]](
        BenchId("bench_most_common_heap[total=100,n=5]")
    )
    m.bench_function[bench_most_common_large_n[100, 80]](
        BenchId("bench_most_common_sort[total=100,n=80]")
    )

    # total=1000: heap path (n=10), full sort path (n=800)
    m.bench_function[bench_most_common_small_n[1000, 10]](
        BenchId("bench_most_common_heap[total=1000,n=10]")
    )
    m.bench_function[bench_most_common_large_n[1000, 800]](
        BenchId("bench_most_common_sort[total=1000,n=800]")
    )

    # total=10_000: heap path (n=10), full sort path (n=8000)
    m.bench_function[bench_most_common_small_n[10_000, 10]](
        BenchId("bench_most_common_heap[total=10000,n=10]")
    )
    m.bench_function[bench_most_common_large_n[10_000, 8_000]](
        BenchId("bench_most_common_sort[total=10000,n=8000]")
    )

    print(m)
