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

"""Benchmarks for StringSlice byte-level access methods."""

from std.os import abort
from std.pathlib import _dir_of_current_file
from std.sys import stderr

from std.benchmark import Bench, BenchConfig, Bencher, BenchId, black_box, keep


# ===-----------------------------------------------------------------------===#
# Benchmark Data
# ===-----------------------------------------------------------------------===#
def make_string[
    length: Int = 0
](filename: String = "UN_charter_EN.txt") -> String:
    """Make a `String` from the `./data` directory.

    Parameters:
        length: The length in bytes of the resulting `String`. If == 0, use
            the whole file content.

    Args:
        filename: The name of the file inside the `./data` directory.
    """
    try:
        directory = _dir_of_current_file() / "data"
        var f = open(directory / filename, "r")

        comptime if length > 0:
            var items = f.read_bytes(length)
            i = 0
            while length > len(items):
                items.append(items[i])
                i = i + 1 if i < len(items) - 1 else 0
            return String(unsafe_from_utf8=items)
        else:
            return String(unsafe_from_utf8=f.read_bytes())
    except e:
        print(e, file=stderr)
    abort(String())


# ===-----------------------------------------------------------------------===#
# Benchmark: byte= subscript access
# ===-----------------------------------------------------------------------===#
@parameter
def bench_byte_subscript[
    length: Int = 0,
    filename: StaticString = "UN_charter_EN",
](mut b: Bencher) raises:
    var text = make_string[length](filename + ".txt")
    var n = text.byte_length()

    @always_inline
    def call_fn() unified {read}:
        var s = 0
        for i in range(black_box(n)):
            s += Int(ord(black_box(text)[byte=i]))
        keep(s)

    b.iter(call_fn)


# ===-----------------------------------------------------------------------===#
# Benchmark: raw pointer access (baseline)
# ===-----------------------------------------------------------------------===#
@parameter
def bench_raw_pointer[
    length: Int = 0,
    filename: StaticString = "UN_charter_EN",
](mut b: Bencher) raises:
    var text = make_string[length](filename + ".txt")
    var n = text.byte_length()
    var ptr = text.unsafe_ptr()

    @always_inline
    def call_fn() unified {read}:
        var s = 0
        for i in range(black_box(n)):
            s += Int(black_box(ptr)[i])
        keep(s)

    b.iter(call_fn)


# ===-----------------------------------------------------------------------===#
# Benchmark: byte= subscript on multibyte UTF-8 text
# ===-----------------------------------------------------------------------===#
@parameter
def bench_byte_subscript_utf8[
    length: Int = 0,
    filename: StaticString = "UN_charter_RU",
](mut b: Bencher) raises:
    var text = make_string[length](filename + ".txt")
    var n = text.byte_length()

    @always_inline
    def call_fn() unified {read}:
        var s = 0
        var i = 0
        while i < black_box(n):
            var c = black_box(text)[byte=i]
            s += Int(ord(c))
            i += c.byte_length()
        keep(s)

    b.iter(call_fn)


# ===-----------------------------------------------------------------------===#
# Main
# ===-----------------------------------------------------------------------===#
def main() raises:
    var m = Bench(BenchConfig(num_repetitions=1, max_iters=100))

    comptime lengths = (1024, 16384)
    comptime ascii_files = (StaticString("UN_charter_EN"),)
    comptime utf8_files = (StaticString("UN_charter_RU"),)

    comptime for li in range(len(lengths)):
        comptime length = lengths[li]

        comptime for fi in range(len(ascii_files)):
            comptime fname = ascii_files[fi]
            comptime suffix = String("_", fname, "_", length)

            m.bench_function[bench_byte_subscript[length, fname]](
                BenchId(String("bench_byte_subscript", suffix))
            )
            m.bench_function[bench_raw_pointer[length, fname]](
                BenchId(String("bench_raw_pointer", suffix))
            )

        comptime for fi in range(len(utf8_files)):
            comptime fname = utf8_files[fi]
            comptime suffix = String("_", fname, "_", length)

            m.bench_function[bench_byte_subscript_utf8[length, fname]](
                BenchId(String("bench_byte_subscript_utf8", suffix))
            )

    var results = Dict[String, Tuple[Float64, Int]]()
    for info in m.info_vec:
        var n = info.name
        var time = info.result.mean("ms")
        var avg, amnt = results.get(n, (Float64(0), 0))
        results[n] = (
            (avg * Float64(amnt) + time) / Float64((amnt + 1)),
            amnt + 1,
        )
    print("")
    for k_v in results.items():
        print(k_v.key, k_v.value[0], sep=", ")
