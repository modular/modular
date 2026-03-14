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
# Benchmark most_common — heap path (n < total // 2)
# ===-----------------------------------------------------------------------===#
@parameter
def bench_most_common_heap[total: Int, n: Int](mut b: Bencher) raises:
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
def bench_most_common_sort[total: Int, n: Int](mut b: Bencher) raises:
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

    # --- Crossover comparison: same n on both sides of the total // 2 threshold ---
    # For each total, n = total//2 - 1 takes the heap path,
    # n = total//2 takes the sort path. Result counts differ by 1, making
    # this the fairest apples-to-apples comparison of the two code paths.

    # total=100, threshold=50
    m.bench_function[bench_most_common_heap[100, 49]](
        BenchId("bench_most_common_heap[total=100,n=49]")
    )
    m.bench_function[bench_most_common_sort[100, 50]](
        BenchId("bench_most_common_sort[total=100,n=50]")
    )

    # total=1000, threshold=500
    m.bench_function[bench_most_common_heap[1000, 499]](
        BenchId("bench_most_common_heap[total=1000,n=499]")
    )
    m.bench_function[bench_most_common_sort[1000, 500]](
        BenchId("bench_most_common_sort[total=1000,n=500]")
    )

    # total=10_000, threshold=5000
    m.bench_function[bench_most_common_heap[10_000, 4_999]](
        BenchId("bench_most_common_heap[total=10000,n=4999]")
    )
    m.bench_function[bench_most_common_sort[10_000, 5_000]](
        BenchId("bench_most_common_sort[total=10000,n=5000]")
    )

    # --- Small-n scaling: heap path, fixed total=10_000 ---
    # Shows how the heap path scales as n grows toward the threshold.
    m.bench_function[bench_most_common_heap[10_000, 10]](
        BenchId("bench_most_common_heap[total=10000,n=10]")
    )
    m.bench_function[bench_most_common_heap[10_000, 100]](
        BenchId("bench_most_common_heap[total=10000,n=100]")
    )
    m.bench_function[bench_most_common_heap[10_000, 1_000]](
        BenchId("bench_most_common_heap[total=10000,n=1000]")
    )

    # --- Large-n scaling: sort path, fixed total=10_000 ---
    m.bench_function[bench_most_common_sort[10_000, 5_000]](
        BenchId("bench_most_common_sort[total=10000,n=5000]")
    )
    m.bench_function[bench_most_common_sort[10_000, 8_000]](
        BenchId("bench_most_common_sort[total=10000,n=8000]")
    )
    m.bench_function[bench_most_common_sort[10_000, 10_000]](
        BenchId("bench_most_common_sort[total=10000,n=10000]")
    )

    print(m)
