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

from collections import Deque

from benchmark import Bench, BenchConfig, Bencher, BenchId, black_box, keep


# ===-----------------------------------------------------------------------===#
# Benchmark Deque copy (trivial type: Int)
# ===-----------------------------------------------------------------------===#
@parameter
fn bench_deque_copy_int[size: Int](mut b: Bencher) raises:
    var q = Deque[Int]()
    for i in range(size):
        q.append(i)

    @always_inline
    @parameter
    fn call_fn():
        var p = black_box(q).copy()
        keep(p)

    b.iter[call_fn]()
    keep(Bool(q))


# ===-----------------------------------------------------------------------===#
# Benchmark Deque copy (non-trivial type: String)
# ===-----------------------------------------------------------------------===#
@parameter
fn bench_deque_copy_string[size: Int](mut b: Bencher) raises:
    var q = Deque[String]()
    for i in range(size):
        q.append(String("item_") + String(i))

    @always_inline
    @parameter
    fn call_fn():
        var p = black_box(q).copy()
        keep(len(p[size - 1]))

    b.iter[call_fn]()
    keep(Bool(q))


# ===-----------------------------------------------------------------------===#
# Benchmark Deque extend (trivial type: Int)
# ===-----------------------------------------------------------------------===#
@parameter
fn bench_deque_extend_int[size: Int](mut b: Bencher) raises:
    var lst = List[Int]()
    for i in range(size):
        lst.append(i)

    @always_inline
    @parameter
    fn call_fn():
        var q = Deque[Int]()
        q.extend(black_box(lst).copy())
        keep(q[size - 1])

    b.iter[call_fn]()


# ===-----------------------------------------------------------------------===#
# Benchmark Deque extend (non-trivial type: String)
# ===-----------------------------------------------------------------------===#
@parameter
fn bench_deque_extend_string[size: Int](mut b: Bencher) raises:
    var lst = List[String]()
    for i in range(size):
        lst.append(String("item_") + String(i))

    @always_inline
    @parameter
    fn call_fn():
        var q = Deque[String]()
        q.extend(black_box(lst).copy())
        keep(len(q[size - 1]))

    b.iter[call_fn]()


# ===-----------------------------------------------------------------------===#
# Benchmark Deque append triggering realloc (trivial type: Int)
# ===-----------------------------------------------------------------------===#
@parameter
fn bench_deque_append_int[size: Int](mut b: Bencher) raises:
    @always_inline
    @parameter
    fn call_fn():
        var q = Deque[Int]()
        for i in range(size):
            q.append(black_box(i))
        keep(q[size - 1])

    b.iter[call_fn]()


# ===-----------------------------------------------------------------------===#
# Benchmark Main
# ===-----------------------------------------------------------------------===#
def main():
    var m = Bench(BenchConfig(num_repetitions=20))
    comptime sizes = (100, 1_000, 10_000, 100_000)

    comptime
    for i in range(len(sizes)):
        comptime size = sizes[i]
        m.bench_function[bench_deque_copy_int[size]](
            BenchId(String("bench_deque_copy_int[", size, "]"))
        )
        m.bench_function[bench_deque_copy_string[size]](
            BenchId(String("bench_deque_copy_string[", size, "]"))
        )
        m.bench_function[bench_deque_extend_int[size]](
            BenchId(String("bench_deque_extend_int[", size, "]"))
        )
        m.bench_function[bench_deque_extend_string[size]](
            BenchId(String("bench_deque_extend_string[", size, "]"))
        )
        m.bench_function[bench_deque_append_int[size]](
            BenchId(String("bench_deque_append_int[", size, "]"))
        )

    results = Dict[String, Tuple[Float64, Int]]()
    for info in m.info_vec:
        n = info.name
        time = info.result.mean("ms")
        avg, amnt = results.get(n, (Float64(0), 0))
        results[n] = (
            (avg * Float64(amnt) + time) / Float64((amnt + 1)),
            amnt + 1,
        )
    print("")
    for k_v in results.items():
        print(k_v.key, k_v.value[0], sep=",")
