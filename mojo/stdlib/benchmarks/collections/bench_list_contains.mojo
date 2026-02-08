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


# ===-----------------------------------------------------------------------===#
# Benchmark Data
# ===-----------------------------------------------------------------------===#
fn make_list[size: Int]() -> List[Int]:
    var l = List[Int](capacity=size)
    for i in range(size):
        l.append(i)
    return l^


fn make_list_float64[size: Int]() -> List[Float64]:
    var l = List[Float64](capacity=size)
    for i in range(size):
        l.append(Float64(i) * 1.5)
    return l^


fn make_list_uint8[size: Int]() -> List[UInt8]:
    """Creates a UInt8 list where value 0 first appears at position size // 2."""
    var l = List[UInt8](capacity=size)
    for i in range(size):
        if i < size // 2:
            l.append(UInt8((i % 255) + 1))  # 1-255, never 0
        else:
            l.append(UInt8(i % 256))
    return l^


# ===-----------------------------------------------------------------------===#
# Benchmark List contains (Int)
# ===-----------------------------------------------------------------------===#
@parameter
fn bench_list_contains_hit[size: Int](mut b: Bencher) raises:
    """Search for an element present in the middle of the list."""
    var items = make_list[size]()
    var target = size // 2

    @always_inline
    @parameter
    fn call_fn() raises:
        for _ in range(100):
            var res = target in items
            keep(res)

    b.iter[call_fn]()
    keep(Bool(items))
    keep(target)


@parameter
fn bench_list_contains_miss[size: Int](mut b: Bencher) raises:
    """Search for an element not present in the list (worst case)."""
    var items = make_list[size]()
    var target = size  # Not in the list

    @always_inline
    @parameter
    fn call_fn() raises:
        for _ in range(100):
            var res = target in items
            keep(res)

    b.iter[call_fn]()
    keep(Bool(items))
    keep(target)


# ===-----------------------------------------------------------------------===#
# Benchmark List contains (Float64)
# ===-----------------------------------------------------------------------===#
@parameter
fn bench_list_contains_float64[size: Int](mut b: Bencher) raises:
    """Search for a Float64 element present in the middle of the list."""
    var items = make_list_float64[size]()
    var target = Float64(size // 2) * 1.5

    @always_inline
    @parameter
    fn call_fn() raises:
        for _ in range(100):
            var res = target in items
            keep(res)

    b.iter[call_fn]()
    keep(Bool(items))
    keep(target)


# ===-----------------------------------------------------------------------===#
# Benchmark List contains (UInt8)
# ===-----------------------------------------------------------------------===#
@parameter
fn bench_list_contains_uint8[size: Int](mut b: Bencher) raises:
    """Search for a UInt8 element whose first occurrence is at size // 2."""
    var items = make_list_uint8[size]()
    var target = UInt8(0)

    @always_inline
    @parameter
    fn call_fn() raises:
        for _ in range(100):
            var res = target in items
            keep(res)

    b.iter[call_fn]()
    keep(Bool(items))
    keep(target)


# ===-----------------------------------------------------------------------===#
# Benchmark Main
# ===-----------------------------------------------------------------------===#
def main():
    var m = Bench(BenchConfig(num_repetitions=50))
    comptime sizes = (10, 50, 100, 1000, 10_000, 100_000)

    @parameter
    for i in range(len(sizes)):
        comptime size = sizes[i]
        m.bench_function[bench_list_contains_hit[size]](
            BenchId(String("bench_list_contains_hit[", size, "]"))
        )
        m.bench_function[bench_list_contains_miss[size]](
            BenchId(String("bench_list_contains_miss[", size, "]"))
        )
        m.bench_function[bench_list_contains_float64[size]](
            BenchId(String("bench_list_contains_float64[", size, "]"))
        )
        m.bench_function[bench_list_contains_uint8[size]](
            BenchId(String("bench_list_contains_uint8[", size, "]"))
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
