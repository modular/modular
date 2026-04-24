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
"""Benchmarks for grapheme cluster segmentation.

Compares `count_graphemes()` against `count_codepoints()` across different
text profiles (English, Spanish, Arabic, Russian, Chinese) and lengths.
"""

from std.os import abort
from std.pathlib import _dir_of_current_file
from std.sys import stderr

from std.benchmark import Bench, BenchConfig, Bencher, BenchId, black_box, keep


# ===-----------------------------------------------------------------------===#
# Benchmark Data
# ===-----------------------------------------------------------------------===#
# TODO: duplicated from `bench_string.mojo`. Consolidate into a shared
# benchmark utility module once one exists.
def make_string[
    length: Int = 0
](filename: String = "UN_charter_EN.txt") -> String:
    """Make a `String` from the `./data` directory.

    Parameters:
        length: The length in bytes. If == 0 -> the whole file.

    Args:
        filename: The name of the file inside the `./data` directory.
    """
    try:
        directory = _dir_of_current_file() / "data"
        var f = open(directory / filename, "r")

        comptime if length == 0:
            return String(unsafe_from_utf8=f.read_bytes())

        # Repeat the file content until we have at least `length` bytes, then
        # truncate back to the nearest UTF-8 codepoint boundary <= length so
        # the result is valid UTF-8 (important for grapheme segmentation).
        var full = f.read_bytes()
        var items = List[Byte](capacity=length + len(full))
        while len(items) < length:
            items.extend(full.copy())
        var cut = length
        while cut < len(items) and (items[cut] & 0xC0) == 0x80:
            cut -= 1
        while len(items) > cut:
            _ = items.pop()
        return String(unsafe_from_utf8=items)
    except e:
        print(e, file=stderr)
    abort(String())


# ===-----------------------------------------------------------------------===#
# Benchmarks
# ===-----------------------------------------------------------------------===#
@parameter
def bench_count_graphemes[
    length: Int = 0, filename: StaticString = "UN_charter_EN"
](mut b: Bencher) raises:
    var items = make_string[length](filename + ".txt")

    @always_inline
    def call_fn() unified {read}:
        var res = black_box(items).count_graphemes()
        keep(res)

    b.iter(call_fn)


@parameter
def bench_count_codepoints[
    length: Int = 0, filename: StaticString = "UN_charter_EN"
](mut b: Bencher) raises:
    var items = make_string[length](filename + ".txt")

    @always_inline
    def call_fn() unified {read}:
        var res = black_box(items).count_codepoints()
        keep(res)

    b.iter(call_fn)


@parameter
def bench_grapheme_iter[
    length: Int = 0, filename: StaticString = "UN_charter_EN"
](mut b: Bencher) raises:
    var items = make_string[length](filename + ".txt")

    @always_inline
    def call_fn() unified {read}:
        var count = 0
        for _ in black_box(items).graphemes():
            count += 1
        keep(count)

    b.iter(call_fn)


# ===-----------------------------------------------------------------------===#
# Benchmark Main
# ===-----------------------------------------------------------------------===#
def main() raises:
    var m = Bench(BenchConfig(num_repetitions=1))
    comptime filenames = (
        StaticString("UN_charter_EN"),
        StaticString("UN_charter_ES"),
        StaticString("UN_charter_AR"),
        StaticString("UN_charter_RU"),
        StaticString("UN_charter_zh-CN"),
    )

    comptime lengths = (100, 1000, 10_000, 100_000, 1_000_000)

    comptime for i in range(len(lengths)):
        comptime length = lengths[i]

        comptime for j in range(len(filenames)):
            comptime fname = filenames[j]
            # NOTE: fname is intentionally omitted from the bench ID so the
            # per-language rows share a name and the aggregation loop below
            # averages across languages (mirrors `bench_string.mojo`).
            comptime suffix = String("[", length, "]")
            m.bench_function[bench_count_codepoints[length, fname]](
                BenchId(String("bench_count_codepoints", suffix))
            )
            m.bench_function[bench_count_graphemes[length, fname]](
                BenchId(String("bench_count_graphemes", suffix))
            )
            m.bench_function[bench_grapheme_iter[length, fname]](
                BenchId(String("bench_grapheme_iter", suffix))
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
