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

from memory import Span

from benchmark import Bench, BenchConfig, Bencher, BenchId, black_box, keep


# ===-----------------------------------------------------------------------===#
# Benchmark Span.copy_from (trivial type: Int)
# ===-----------------------------------------------------------------------===#
@parameter
fn bench_span_copy_from_int[size: Int](mut b: Bencher) raises:
    var src = List[Int]()
    var dst = List[Int]()
    for i in range(size):
        src.append(i)
        dst.append(0)

    @always_inline
    @parameter
    fn call_fn():
        Span(dst).copy_from(black_box(Span(src)))
        keep(dst[size - 1])

    b.iter[call_fn]()
    keep(Bool(src))


# ===-----------------------------------------------------------------------===#
# Benchmark Span.copy_from (non-trivial type: String)
# ===-----------------------------------------------------------------------===#
@parameter
fn bench_span_copy_from_string[size: Int](mut b: Bencher) raises:
    var src = List[String]()
    var dst = List[String]()
    for i in range(size):
        src.append(String("item_") + String(i))
        dst.append(String("____") + String(i))

    @always_inline
    @parameter
    fn call_fn():
        Span(dst).copy_from(black_box(Span(src)))
        keep(len(dst[size - 1]))

    b.iter[call_fn]()
    keep(Bool(src))


# ===-----------------------------------------------------------------------===#
# Benchmark Main
# ===-----------------------------------------------------------------------===#
def main():
    var m = Bench(BenchConfig(num_repetitions=20))
    comptime sizes = (100, 1_000, 10_000, 100_000)

    comptime
    for i in range(len(sizes)):
        comptime size = sizes[i]
        m.bench_function[bench_span_copy_from_int[size]](
            BenchId(String("bench_span_copy_from_int[", size, "]"))
        )
        m.bench_function[bench_span_copy_from_string[size]](
            BenchId(String("bench_span_copy_from_string[", size, "]"))
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
