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

from collections import Set

from benchmark import Bench, BenchConfig, Bencher, BenchId, keep


# ===-----------------------------------------------------------------------===#
# Benchmark Data
# ===-----------------------------------------------------------------------===#
fn make_int_set[size: Int]() -> Set[Int]:
    var s = Set[Int]()
    for i in range(size):
        s.add(i)
    return s^


fn make_string_set[size: Int]() -> Set[String]:
    var s = Set[String]()
    for i in range(size):
        s.add(String("key_") + String(i))
    return s^


# ===-----------------------------------------------------------------------===#
# Benchmark Set.__eq__ (Int keys)
# ===-----------------------------------------------------------------------===#
@parameter
fn bench_set_eq_int[size: Int](mut b: Bencher) raises:
    """Benchmark equality check of two equal Int sets."""
    var s1 = make_int_set[size]()
    var s2 = make_int_set[size]()

    @always_inline
    @parameter
    fn call_fn():
        var res = s1 == s2
        keep(res)

    b.iter[call_fn]()
    keep(Bool(s1))
    keep(Bool(s2))


# ===-----------------------------------------------------------------------===#
# Benchmark Set.__eq__ (String keys - expensive hash)
# ===-----------------------------------------------------------------------===#
@parameter
fn bench_set_eq_string[size: Int](mut b: Bencher) raises:
    """Benchmark equality check of two equal String sets."""
    var s1 = make_string_set[size]()
    var s2 = make_string_set[size]()

    @always_inline
    @parameter
    fn call_fn():
        var res = s1 == s2
        keep(res)

    b.iter[call_fn]()
    keep(Bool(s1))
    keep(Bool(s2))


# ===-----------------------------------------------------------------------===#
# Benchmark Set.__eq__ early exit (different sizes)
# ===-----------------------------------------------------------------------===#
@parameter
fn bench_set_eq_diff_size[size: Int](mut b: Bencher) raises:
    """Benchmark equality fast-path rejection when sizes differ."""
    var s1 = make_int_set[size]()
    var s2 = make_int_set[size + 1]()

    @always_inline
    @parameter
    fn call_fn():
        var res = s1 == s2
        keep(res)

    b.iter[call_fn]()
    keep(Bool(s1))
    keep(Bool(s2))


# ===-----------------------------------------------------------------------===#
# Benchmark Set.__eq__ early exit (same size, different elements)
# ===-----------------------------------------------------------------------===#
@parameter
fn bench_set_eq_diff_elems[size: Int](mut b: Bencher) raises:
    """Benchmark equality when sets have same size but different elements."""
    var s1 = make_int_set[size]()
    var s2 = Set[Int]()
    for i in range(size):
        s2.add(i + size)

    @always_inline
    @parameter
    fn call_fn():
        var res = s1 == s2
        keep(res)

    b.iter[call_fn]()
    keep(Bool(s1))
    keep(Bool(s2))


# ===-----------------------------------------------------------------------===#
# Benchmark Set.__contains__
# ===-----------------------------------------------------------------------===#
@parameter
fn bench_set_contains[size: Int](mut b: Bencher) raises:
    """Benchmark membership check for 10 elements."""
    var s = make_int_set[size]()

    @always_inline
    @parameter
    fn call_fn():
        for key in range(10):
            var res = key in s
            keep(res)

    b.iter[call_fn]()
    keep(Bool(s))


# ===-----------------------------------------------------------------------===#
# Benchmark Set.add
# ===-----------------------------------------------------------------------===#
@parameter
fn bench_set_add[size: Int](mut b: Bencher) raises:
    """Benchmark adding 10 existing elements (duplicate check) to a set."""
    var s = make_int_set[size]()

    @always_inline
    @parameter
    fn call_fn():
        for key in range(10):
            s.add(key)

    b.iter[call_fn]()
    keep(Bool(s))


# ===-----------------------------------------------------------------------===#
# Benchmark Set.union
# ===-----------------------------------------------------------------------===#
@parameter
fn bench_set_union[size: Int](mut b: Bencher) raises:
    """Benchmark union of two sets with 50% overlap."""
    var s1 = make_int_set[size]()
    var s2 = Set[Int]()
    var half = size // 2
    for i in range(half, half + size):
        s2.add(i)

    @always_inline
    @parameter
    fn call_fn():
        var res = s1 | s2
        keep(Bool(res))

    b.iter[call_fn]()
    keep(Bool(s1))
    keep(Bool(s2))


# ===-----------------------------------------------------------------------===#
# Benchmark Set.intersection
# ===-----------------------------------------------------------------------===#
@parameter
fn bench_set_intersection[size: Int](mut b: Bencher) raises:
    """Benchmark intersection of two sets with 50% overlap."""
    var s1 = make_int_set[size]()
    var s2 = Set[Int]()
    var half = size // 2
    for i in range(half, half + size):
        s2.add(i)

    @always_inline
    @parameter
    fn call_fn():
        var res = s1 & s2
        keep(Bool(res))

    b.iter[call_fn]()
    keep(Bool(s1))
    keep(Bool(s2))


# ===-----------------------------------------------------------------------===#
# Benchmark Set.difference
# ===-----------------------------------------------------------------------===#
@parameter
fn bench_set_difference[size: Int](mut b: Bencher) raises:
    """Benchmark difference of two sets with 50% overlap."""
    var s1 = make_int_set[size]()
    var s2 = Set[Int]()
    var half = size // 2
    for i in range(half, half + size):
        s2.add(i)

    @always_inline
    @parameter
    fn call_fn():
        var res = s1 - s2
        keep(Bool(res))

    b.iter[call_fn]()
    keep(Bool(s1))
    keep(Bool(s2))


# ===-----------------------------------------------------------------------===#
# Benchmark Main
# ===-----------------------------------------------------------------------===#
def main():
    var m = Bench(BenchConfig(num_repetitions=5))
    comptime sizes = (10, 100, 1000, 10_000)

    @parameter
    for i in range(len(sizes)):
        comptime size = sizes[i]

        # Equality benchmarks
        m.bench_function[bench_set_eq_int[size]](
            BenchId(String("bench_set_eq_int[", size, "]"))
        )
        m.bench_function[bench_set_eq_string[size]](
            BenchId(String("bench_set_eq_string[", size, "]"))
        )
        m.bench_function[bench_set_eq_diff_size[size]](
            BenchId(String("bench_set_eq_diff_size[", size, "]"))
        )
        m.bench_function[bench_set_eq_diff_elems[size]](
            BenchId(String("bench_set_eq_diff_elems[", size, "]"))
        )

        # Basic operations
        m.bench_function[bench_set_contains[size]](
            BenchId(String("bench_set_contains[", size, "]"))
        )
        m.bench_function[bench_set_add[size]](
            BenchId(String("bench_set_add[", size, "]"))
        )

        # Set algebra
        m.bench_function[bench_set_union[size]](
            BenchId(String("bench_set_union[", size, "]"))
        )
        m.bench_function[bench_set_intersection[size]](
            BenchId(String("bench_set_intersection[", size, "]"))
        )
        m.bench_function[bench_set_difference[size]](
            BenchId(String("bench_set_difference[", size, "]"))
        )

    results = Dict[String, Tuple[Float64, Int]]()
    for info in m.info_vec:
        n = info.name
        time = info.result.mean("ms")
        avg, amnt = results.get(n, (Float64(0), 0))
        results[n] = ((avg * amnt + time) / (amnt + 1), amnt + 1)
    print("")
    for k_v in results.items():
        print(k_v.key, k_v.value[0], sep=",")
