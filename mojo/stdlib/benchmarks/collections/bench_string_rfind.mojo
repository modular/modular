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

"""Benchmarks for String.rfind(), which uses _memrchr internally.

The needle appears only at byte 0, forcing a full reverse scan of the string.
This is the worst case for _memrchr and best demonstrates the SIMD speedup.
"""

from benchmark import Bench, BenchConfig, Bencher, BenchId, black_box, keep


# ===-----------------------------------------------------------------------===#
# Benchmark String.rfind (single char, needle at start = full reverse scan)
# ===-----------------------------------------------------------------------===#
@parameter
fn bench_rfind_char[size: Int](mut b: Bencher) raises:
    # 'Z' only at position 0; all other bytes are 'x'.
    var s = String("Z") + String("x") * (size - 1)

    @always_inline
    fn call_fn() unified {read}:
        var pos = black_box(s).rfind("Z")
        keep(pos)

    b.iter(call_fn)


# ===-----------------------------------------------------------------------===#
# Benchmark Main
# ===-----------------------------------------------------------------------===#
def main():
    var m = Bench(BenchConfig(num_repetitions=20))
    comptime sizes = (100, 1_000, 10_000, 100_000)

    comptime
    for i in range(len(sizes)):
        comptime size = sizes[i]
        m.bench_function[bench_rfind_char[size]](
            BenchId(String("bench_rfind_char[", size, "]"))
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
