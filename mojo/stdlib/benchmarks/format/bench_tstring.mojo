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

from std.collections import Optional
from std.collections.string._utf8 import _is_valid_utf8
from std.collections.string.string_slice import _split
from std.os import abort
from std.pathlib import _dir_of_current_file
from std.random import seed
from std.sys import stderr

from std.benchmark import Bench, BenchConfig, Bencher, BenchId, black_box, keep


@fieldwise_init
struct NullWriter(Writer):
    var array: InlineArray[Byte, 1024]

    @always_inline
    fn __init__(out self):
        self.array = {uninitialized = True}

    @always_inline
    fn write_string(mut self, string: StringSlice):
        var bytes = string.as_bytes()
        for i in range(len(bytes)):
            (self.array.unsafe_ptr() + i).store(bytes.unsafe_get(i))
        keep(self.array)


@fieldwise_init
struct NullWritable(Writable):
    @always_inline
    fn write_to(self, mut writer: Some[Writer]):
        writer.write_string(black_box(StaticString("null")))


@always_inline
fn null_print(tstring: Some[Writable]):
    var writer = NullWriter()
    tstring.write_to(black_box(writer))


@parameter
fn bench_tstring_single_value(mut b: Bencher) raises:
    @always_inline
    fn call_fn() unified {}:
        for _ in range(100):
            var a = NullWritable()
            null_print(t"{a}")
            keep(a)

    b.iter(call_fn)


@parameter
fn bench_tstring_only_literal(mut b: Bencher) raises:
    @always_inline
    fn call_fn() unified {}:
        for _ in range(100):
            null_print(t"The quick brown fox jumps over the lazy dog")

    b.iter(call_fn)


@parameter
fn bench_tstring_many_values_no_literals(mut b: Bencher) raises:
    @always_inline
    fn call_fn() unified {}:
        for _ in range(100):
            var a = NullWritable()
            var b = NullWritable()
            var c = NullWritable()
            var d = NullWritable()
            var e = NullWritable()
            null_print(t"{a}{b}{c}{d}{e}")
            keep(a)
            keep(b)
            keep(c)
            keep(d)
            keep(e)

    b.iter(call_fn)


@parameter
fn bench_tstring_long_literals(mut b: Bencher) raises:
    @always_inline
    fn call_fn() unified {}:
        for _ in range(100):
            var a = NullWritable()
            var b = NullWritable()
            # fmt: off
            null_print(
                t"Lorem ipsum dolor sit amet, consectetur adipiscing elit. {a}"
                t" Sed do eiusmod tempor incididunt ut labore et dolore magna"
                t" aliqua. Ut enim ad minim veniam, quis nostrud {b}"
                t" exercitation ullamco laboris nisi ut aliquip."
            )
            # fmt: on
            keep(a)
            keep(b)

    b.iter(call_fn)


@parameter
fn bench_tstring_many_values_many_literals(mut b: Bencher) raises:
    @always_inline
    fn call_fn() unified {}:
        for _ in range(100):
            var a = NullWritable()
            var b = NullWritable()
            var c = NullWritable()
            var d = NullWritable()
            var e = NullWritable()
            var f = NullWritable()
            var g = NullWritable()
            var h = NullWritable()
            null_print(t"a={a} b={b} c={c} d={d} e={e} f={f} g={g} h={h}")
            keep(a)
            keep(b)
            keep(c)
            keep(d)
            keep(e)
            keep(f)
            keep(g)
            keep(h)

    b.iter(call_fn)


def main() raises:
    seed()

    var m = Bench(BenchConfig(num_repetitions=1))
    m.bench_function[bench_tstring_single_value](
        BenchId("bench_tstring_single_value")
    )
    m.bench_function[bench_tstring_only_literal](
        BenchId("bench_tstring_only_literal")
    )
    m.bench_function[bench_tstring_many_values_no_literals](
        BenchId("bench_tstring_many_values_no_literals")
    )
    m.bench_function[bench_tstring_long_literals](
        BenchId("bench_tstring_long_literals")
    )
    m.bench_function[bench_tstring_many_values_many_literals](
        BenchId("bench_tstring_many_values_many_literals")
    )
    print(m)
