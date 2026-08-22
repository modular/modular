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

from std.collections import Deque

from std.builtin.rebind import rebind
from std.benchmark import Bench, BenchConfig, Bencher, BenchId, black_box, keep


# ===-----------------------------------------------------------------------===#
# Benchmark Deque copy (trivial type: Int)
# ===-----------------------------------------------------------------------===#
def bench_deque_copy_int[size: Int](mut b: Bencher) raises:
    var q = Deque[Int]()
    for i in range(size):
        q.append(i)

    @always_inline
    def call_fn() {imm}:
        var p = black_box(q).copy()
        keep(p)

    b.iter(call_fn)
    keep(Bool(q))


# ===-----------------------------------------------------------------------===#
# Benchmark Deque copy (non-trivial type: String)
# ===-----------------------------------------------------------------------===#
def bench_deque_copy_string[size: Int](mut b: Bencher) raises:
    var q = Deque[String]()
    for i in range(size):
        q.append(String("item_") + String(i))

    @always_inline
    def call_fn() {imm}:
        var p = black_box(q).copy()
        keep(p[size - 1].byte_length())

    b.iter(call_fn)
    keep(Bool(q))


# ===-----------------------------------------------------------------------===#
# Benchmark Deque extend (trivial type: Int)
# ===-----------------------------------------------------------------------===#
def bench_deque_extend_int[size: Int](mut b: Bencher) raises:
    var lst = List[Int]()
    for i in range(size):
        lst.append(i)

    @always_inline
    def call_fn() {imm}:
        var q = Deque[Int]()
        q.extend(black_box(lst).copy())
        keep(q[size - 1])

    b.iter(call_fn)


# ===-----------------------------------------------------------------------===#
# Benchmark Deque extend (non-trivial type: String)
# ===-----------------------------------------------------------------------===#
def bench_deque_extend_string[size: Int](mut b: Bencher) raises:
    var lst = List[String]()
    for i in range(size):
        lst.append(String("item_") + String(i))

    @always_inline
    def call_fn() {imm}:
        var q = Deque[String]()
        q.extend(black_box(lst).copy())
        keep(q[size - 1].byte_length())

    b.iter(call_fn)


# ===-----------------------------------------------------------------------===#
# Benchmark Deque append triggering realloc (trivial type: Int)
# ===-----------------------------------------------------------------------===#
def bench_deque_append_int[size: Int](mut b: Bencher) raises:
    @always_inline
    def call_fn() {imm}:
        var q = Deque[Int]()
        for i in range(size):
            q.append(black_box(i))
        keep(q[size - 1])

    b.iter(call_fn)


# ===-----------------------------------------------------------------------===#
# Benchmark Main
# ===-----------------------------------------------------------------------===#
def main() raises:
    var m = Bench(BenchConfig(num_repetitions=20))
    comptime sizes = (100, 1_000, 10_000, 100_000)

    comptime for i in range(len(sizes)):
        comptime size = rebind[Int](sizes[i])
        comptime suffix = String("[", size, "]")
        m.bench_function(
            bench_deque_copy_int[size],
            BenchId(String("bench_deque_copy_int", suffix)),
        )
        m.bench_function(
            bench_deque_copy_string[size],
            BenchId(String("bench_deque_copy_string", suffix)),
        )
        m.bench_function(
            bench_deque_extend_int[size],
            BenchId(String("bench_deque_extend_int", suffix)),
        )
        m.bench_function(
            bench_deque_extend_string[size],
            BenchId(String("bench_deque_extend_string", suffix)),
        )
        m.bench_function(
            bench_deque_append_int[size],
            BenchId(String("bench_deque_append_int", suffix)),
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
