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

from collections import List
from memory import Span

from benchmark import Bench, BenchConfig, Bencher, BenchId, keep


# ===-----------------------------------------------------------------------===#
# Benchmark Data
# ===-----------------------------------------------------------------------===#
fn make_int_list[size: Int]() -> List[Int]:
    var l = List[Int](capacity=size)
    for i in range(size):
        l.append(i)
    return l^


fn make_string_list[size: Int]() -> List[String]:
    var l = List[String](capacity=size)
    for i in range(size):
        l.append(String("item_") + String(i))
    return l^


# ===-----------------------------------------------------------------------===#
# Benchmark List.__init__(Span) — trivial type (Int)
# ===-----------------------------------------------------------------------===#
@parameter
fn bench_list_init_span_int[size: Int](mut b: Bencher) raises:
    """Benchmark constructing a List from a Span of Ints."""
    var src = make_int_list[size]()
    var sp = Span(src)

    @always_inline
    @parameter
    fn call_fn():
        keep(List[Int](sp))

    b.iter[call_fn]()


# ===-----------------------------------------------------------------------===#
# Benchmark List.__init__(Span) — non-trivial type (String)
# ===-----------------------------------------------------------------------===#
@parameter
fn bench_list_init_span_string[size: Int](mut b: Bencher) raises:
    """Benchmark constructing a List from a Span of Strings."""
    var src = make_string_list[size]()
    var sp = Span(src)

    @always_inline
    @parameter
    fn call_fn():
        keep(List[String](sp))

    b.iter[call_fn]()


# ===-----------------------------------------------------------------------===#
# Benchmark List.insert — trivial type (Int)
# ===-----------------------------------------------------------------------===#
@parameter
fn bench_list_insert_front_int[size: Int](mut b: Bencher) raises:
    """Benchmark inserting at front of an Int list (worst case shift)."""

    @always_inline
    @parameter
    fn call_fn():
        var l = make_int_list[size]()
        l.insert(0, 0)
        keep(l)

    b.iter[call_fn]()


@parameter
fn bench_list_insert_middle_int[size: Int](mut b: Bencher) raises:
    """Benchmark inserting at middle of an Int list."""

    @always_inline
    @parameter
    fn call_fn():
        var l = make_int_list[size]()
        l.insert(size // 2, 0)
        keep(l)

    b.iter[call_fn]()


# ===-----------------------------------------------------------------------===#
# Benchmark List.insert — non-trivial type (String)
# ===-----------------------------------------------------------------------===#
@parameter
fn bench_list_insert_front_string[size: Int](mut b: Bencher) raises:
    """Benchmark inserting at front of a String list (worst case shift)."""

    @always_inline
    @parameter
    fn call_fn():
        var l = make_string_list[size]()
        l.insert(0, String("new"))
        keep(l)

    b.iter[call_fn]()


# ===-----------------------------------------------------------------------===#
# Benchmark List.__getitem__(slice) — trivial type (Int), step=1
# ===-----------------------------------------------------------------------===#
@parameter
fn bench_list_slice_int[size: Int](mut b: Bencher) raises:
    """Benchmark slicing a List of Ints with step=1."""
    var l = make_int_list[size]()

    @always_inline
    @parameter
    fn call_fn():
        keep(l[1:])

    b.iter[call_fn]()


@parameter
fn bench_list_slice_half_int[size: Int](mut b: Bencher) raises:
    """Benchmark slicing the middle half of a List of Ints."""
    var l = make_int_list[size]()
    var quarter = size // 4

    @always_inline
    @parameter
    fn call_fn():
        keep(l[quarter : quarter * 3])

    b.iter[call_fn]()


# ===-----------------------------------------------------------------------===#
# Benchmark List.__getitem__(slice) — non-trivial type (String)
# ===-----------------------------------------------------------------------===#
@parameter
fn bench_list_slice_string[size: Int](mut b: Bencher) raises:
    """Benchmark slicing a List of Strings with step=1."""
    var l = make_string_list[size]()

    @always_inline
    @parameter
    fn call_fn():
        keep(l[1:])

    b.iter[call_fn]()


# ===-----------------------------------------------------------------------===#
# Benchmark List.__getitem__(slice) — step != 1
# ===-----------------------------------------------------------------------===#
@parameter
fn bench_list_slice_stride_int[size: Int](mut b: Bencher) raises:
    """Benchmark slicing a List of Ints with step=2 (no memcpy fast path)."""
    var l = make_int_list[size]()

    @always_inline
    @parameter
    fn call_fn():
        keep(l[::2])

    b.iter[call_fn]()


# ===-----------------------------------------------------------------------===#
# Benchmark Main
# ===-----------------------------------------------------------------------===#
def main():
    var m = Bench(BenchConfig(num_repetitions=10))
    comptime sizes = (10, 100, 1000, 10_000)

    @parameter
    for i in range(len(sizes)):
        comptime size = sizes[i]

        # __init__(Span) benchmarks
        m.bench_function[bench_list_init_span_int[size]](
            BenchId(String("bench_list_init_span_int[", size, "]"))
        )
        m.bench_function[bench_list_init_span_string[size]](
            BenchId(String("bench_list_init_span_string[", size, "]"))
        )

        # insert benchmarks
        m.bench_function[bench_list_insert_front_int[size]](
            BenchId(String("bench_list_insert_front_int[", size, "]"))
        )
        m.bench_function[bench_list_insert_middle_int[size]](
            BenchId(String("bench_list_insert_middle_int[", size, "]"))
        )
        m.bench_function[bench_list_insert_front_string[size]](
            BenchId(String("bench_list_insert_front_string[", size, "]"))
        )

        # __getitem__(slice) benchmarks
        m.bench_function[bench_list_slice_int[size]](
            BenchId(String("bench_list_slice_int[", size, "]"))
        )
        m.bench_function[bench_list_slice_half_int[size]](
            BenchId(String("bench_list_slice_half_int[", size, "]"))
        )
        m.bench_function[bench_list_slice_string[size]](
            BenchId(String("bench_list_slice_string[", size, "]"))
        )
        m.bench_function[bench_list_slice_stride_int[size]](
            BenchId(String("bench_list_slice_stride_int[", size, "]"))
        )

    print(m)
