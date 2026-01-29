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

from sys import simd_width_of

from algorithm import vectorize
from benchmark import Bench, BenchConfig, Bencher, BenchId, keep


# ===-----------------------------------------------------------------------===#
# Benchmark vectorize
# ===-----------------------------------------------------------------------===#
@parameter
fn bench_vectorize[n: Int](mut b: Bencher) raises:
    alias dtype = DType.uint8
    var ptr = alloc[Scalar[dtype]](n)

    for i in range(n):
        ptr[i] = -1

    @always_inline
    @parameter
    fn call_fn() raises:
        @always_inline
        fn func[width: Int](idx: Int) unified {mut ptr}:
            ptr.store[width=width](idx, SIMD[dtype, width](42))

        vectorize[simd_width_of[dtype]()](n, func)
        keep(ptr)

    b.iter[call_fn]()
    ptr.free()


def main():
    var m = Bench(BenchConfig(num_repetitions=1))
    m.bench_function[bench_vectorize[31]](BenchId("bench_vectorize_31"))
    m.bench_function[bench_vectorize[32]](BenchId("bench_vectorize_32"))
    m.bench_function[bench_vectorize[127]](BenchId("bench_vectorize_127"))
    m.bench_function[bench_vectorize[128]](BenchId("bench_vectorize_128"))
    m.bench_function[bench_vectorize[1023]](BenchId("bench_vectorize_1023"))
    m.bench_function[bench_vectorize[1024]](BenchId("bench_vectorize_1024"))
    m.bench_function[bench_vectorize[8191]](BenchId("bench_vectorize_8191"))
    m.bench_function[bench_vectorize[8192]](BenchId("bench_vectorize_8192"))
    m.bench_function[bench_vectorize[32767]](BenchId("bench_vectorize_32767"))
    m.bench_function[bench_vectorize[32768]](BenchId("bench_vectorize_32768"))
    m.bench_function[bench_vectorize[131071]](BenchId("bench_vectorize_131071"))
    m.bench_function[bench_vectorize[131072]](BenchId("bench_vectorize_131072"))
    m.dump_report()
