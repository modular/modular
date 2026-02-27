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


from benchmark import Bench, BenchConfig, Bencher, BenchId, keep
from memory import memset

# ===-----------------------------------------------------------------------===#
# Benchmarks
# ===-----------------------------------------------------------------------===#


@parameter
fn bench_memset[count: Int](mut b: Bencher) raises:
    var ptr = alloc[Byte](count)

    @always_inline
    @parameter
    fn call_fn():
        memset(ptr, 0x42, count)
        keep(ptr)

    b.iter[call_fn]()
    ptr.free()


# ===-----------------------------------------------------------------------===#
# Benchmark Main
# ===-----------------------------------------------------------------------===#
def main():
    var m = Bench(BenchConfig(num_repetitions=1))

    # Small sizes — exercises tail / remainder handling.
    m.bench_function[bench_memset[1]](BenchId("memset_1B"))
    m.bench_function[bench_memset[2]](BenchId("memset_2B"))
    m.bench_function[bench_memset[3]](BenchId("memset_3B"))
    m.bench_function[bench_memset[4]](BenchId("memset_4B"))
    m.bench_function[bench_memset[7]](BenchId("memset_7B"))
    m.bench_function[bench_memset[8]](BenchId("memset_8B"))
    m.bench_function[bench_memset[15]](BenchId("memset_15B"))
    m.bench_function[bench_memset[16]](BenchId("memset_16B"))

    # Medium sizes — mix of full vectors and tail.
    m.bench_function[bench_memset[31]](BenchId("memset_31B"))
    m.bench_function[bench_memset[32]](BenchId("memset_32B"))
    m.bench_function[bench_memset[33]](BenchId("memset_33B"))
    m.bench_function[bench_memset[63]](BenchId("memset_63B"))
    m.bench_function[bench_memset[64]](BenchId("memset_64B"))
    m.bench_function[bench_memset[65]](BenchId("memset_65B"))
    m.bench_function[bench_memset[100]](BenchId("memset_100B"))
    m.bench_function[bench_memset[127]](BenchId("memset_127B"))
    m.bench_function[bench_memset[128]](BenchId("memset_128B"))
    m.bench_function[bench_memset[255]](BenchId("memset_255B"))
    m.bench_function[bench_memset[256]](BenchId("memset_256B"))

    # Large sizes — bulk throughput.
    m.bench_function[bench_memset[512]](BenchId("memset_512B"))
    m.bench_function[bench_memset[1024]](BenchId("memset_1KiB"))
    m.bench_function[bench_memset[4096]](BenchId("memset_4KiB"))
    m.bench_function[bench_memset[16384]](BenchId("memset_16KiB"))
    m.bench_function[bench_memset[65536]](BenchId("memset_64KiB"))
    m.bench_function[bench_memset[1048576]](BenchId("memset_1MiB"))

    m.dump_report()
