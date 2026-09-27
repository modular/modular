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

from std.collections.string._utf8 import _is_valid_utf8
from std.collections.string.string_span import _split
from std.os import abort
from std.pathlib import _dir_of_current_file
from std.random import seed
from std.sys import stderr

from std.benchmark import Bench, BenchConfig, Bencher, BenchId, black_box, keep


# ===-----------------------------------------------------------------------===#
# Benchmark Data
# ===-----------------------------------------------------------------------===#
def make_string(length: Int, filename: String) -> String:
    """Make a `String` made of items in the `./data` directory.

    Args:
        length: The length in bytes of the resulting `String`. If == 0 -> the
            whole file content.
        filename: The name of the file inside the `./data` directory.
    """

    try:
        var directory = _dir_of_current_file() / "data"
        var f = open(directory / filename, "r")

        if length > 0:
            var items = f.read_bytes(length)
            var i = 0
            while length > len(items):
                items.append(items[i])
                i = i + 1 if i < len(items) - 1 else 0
            # `length` can truncate mid-codepoint; drop trailing bytes until
            # the buffer ends on a valid UTF-8 boundary again.
            while len(items) > 0 and not _is_valid_utf8(Span(items)):
                _ = items.pop()
            return String(unsafe_from_utf8=items)
        else:
            return String(unsafe_from_utf8=f.read_bytes())
    except e:
        print(e, file=stderr)
    abort(String())


@fieldwise_init
struct BenchInput(ImplicitlyCopyable):
    var length: Int
    var filename: StaticString
    var old: StaticString
    var new: StaticString


# ===-----------------------------------------------------------------------===#
# Benchmark string init
# ===-----------------------------------------------------------------------===#
def bench_string_init(mut b: Bencher) raises:
    @inline(.always)
    def call_fn():
        for _ in range(1000):
            var string = String()
            keep(string)

    b.iter(call_fn)


# ===-----------------------------------------------------------------------===#
# Benchmark string count
# ===-----------------------------------------------------------------------===#
def bench_string_count(mut b: Bencher, input: BenchInput) raises:
    var items = make_string(input.length, input.filename + ".txt")

    @inline(.always)
    def call_fn() {imm}:
        var amnt = black_box(items).count(black_box(input.old))
        keep(amnt)

    b.iter(call_fn)


# ===-----------------------------------------------------------------------===#
# Benchmark string split
# ===-----------------------------------------------------------------------===#
def bench_string_split(mut b: Bencher, input: BenchInput) raises:
    var items = StringSlice(
        make_string(input.length, input.filename + ".txt")
    ).as_imm()

    @inline(.always)
    def call_fn() {imm}:
        var res = _split[has_maxsplit=False](
            black_box(items), black_box(input.old), black_box(-1)
        )
        keep(res)

    b.iter(call_fn)


def bench_string_split_none(mut b: Bencher, input: BenchInput) raises:
    var items = StringSlice(
        make_string(input.length, input.filename + ".txt")
    ).as_imm()

    @inline(.always)
    def call_fn() {imm}:
        var res = _split[has_maxsplit=False](
            black_box(items), None, black_box(-1)
        )
        keep(res)

    b.iter(call_fn)


# ===-----------------------------------------------------------------------===#
# Benchmark string join
# ===-----------------------------------------------------------------------===#
def bench_string_join[short: Bool](mut b: Bencher) raises:
    var count: Int
    comptime if short:
        count = 100
    else:
        count = 1000

    var word_list = List[String](capacity=count)
    for i in range(count):
        word_list.append(String(i))

    var separator = String(",")

    @inline(.always)
    def call_fn() {imm}:
        for _ in range(1_000):
            var res = black_box(separator).join(black_box(word_list))
            keep(res)

    b.iter(call_fn)


# ===-----------------------------------------------------------------------===#
# Benchmark string splitlines
# ===-----------------------------------------------------------------------===#
def bench_string_splitlines(mut b: Bencher, input: BenchInput) raises:
    var items = StringSlice(make_string(input.length, input.filename + ".txt"))

    @inline(.always)
    def call_fn() {imm}:
        for _ in range(1_000_000 // input.length):
            var res = black_box(items).splitlines()
            keep(res)

    b.iter(call_fn)


# ===-----------------------------------------------------------------------===#
# Benchmark string lower
# ===-----------------------------------------------------------------------===#
def bench_string_lower(mut b: Bencher, input: BenchInput) raises:
    var items = make_string(input.length, input.filename + ".txt")

    @inline(.always)
    def call_fn() {imm}:
        var res = black_box(items).lower()
        keep(res)

    b.iter(call_fn)


# ===-----------------------------------------------------------------------===#
# Benchmark string upper
# ===-----------------------------------------------------------------------===#
def bench_string_upper(mut b: Bencher, input: BenchInput) raises:
    var items = make_string(input.length, input.filename + ".txt")

    @inline(.always)
    def call_fn() {imm}:
        var res = black_box(items).upper()
        keep(res)

    b.iter(call_fn)


# ===-----------------------------------------------------------------------===#
# Benchmark string replace
# ===-----------------------------------------------------------------------===#
def bench_string_replace(mut b: Bencher, input: BenchInput) raises:
    var items = make_string(input.length, input.filename + ".txt")

    @inline(.always)
    def call_fn() {imm}:
        var res = black_box(items).replace(
            black_box(input.old), black_box(input.new)
        )
        keep(res)

    b.iter(call_fn)


# ===-----------------------------------------------------------------------===#
# Benchmark string count_codepoints
# ===-----------------------------------------------------------------------===#
def bench_string_count_codepoints(mut b: Bencher, input: BenchInput) raises:
    var items = make_string(input.length, input.filename + ".txt")

    @inline(.always)
    def call_fn() {imm}:
        var res = black_box(items).count_codepoints()
        keep(res)

    b.iter(call_fn)


# ===-----------------------------------------------------------------------===#
# Benchmark string find single
# ===-----------------------------------------------------------------------===#
def bench_string_find_single(mut b: Bencher, input: BenchInput) raises:
    var items = make_string(input.length, input.filename + ".txt")

    @inline(.always)
    def call_fn() {imm}:
        # this is to help with instability when measuring small strings
        for _ in range(10**6 // input.length):
            var res = black_box(items).find(
                black_box("Z")
            )  # something that probably won't be there
            keep(res)

    b.iter(call_fn)


# ===-----------------------------------------------------------------------===#
# Benchmark string find multiple
# ===-----------------------------------------------------------------------===#
def bench_string_find_multiple(mut b: Bencher, input: BenchInput) raises:
    var items = make_string(input.length, input.filename + ".txt")
    var sequence = "ZZZZ"  # something that probably won't be there

    @inline(.always)
    def call_fn() {imm}:
        # this is to help with instability when measuring small strings
        for _ in range(10**6 // input.length):
            var res = black_box(items).find(black_box(sequence))
            keep(res)

    b.iter(call_fn)


# ===-----------------------------------------------------------------------===#
# Benchmark string startswith
# ===-----------------------------------------------------------------------===#
def bench_string_startswith(mut b: Bencher, input: BenchInput) raises:
    var items = make_string(input.length, input.filename + ".txt")
    var prefix = "ZZZZ"  # something that is not there

    @inline(.always)
    def call_fn() {imm}:
        # this is to help with instability when measuring small strings
        for _ in range(10**6 // input.length):
            var res = black_box(items).startswith(black_box(prefix))
            keep(res)

    b.iter(call_fn)


# ===-----------------------------------------------------------------------===#
# Benchmark string endswith
# ===-----------------------------------------------------------------------===#
def bench_string_endswith(mut b: Bencher, input: BenchInput) raises:
    var items = make_string(input.length, input.filename + ".txt")
    var suffix = "ZZZZ"  # something that is not there

    @inline(.always)
    def call_fn() {imm}:
        # this is to help with instability when measuring small strings
        for _ in range(10**6 // input.length):
            var res = black_box(items).endswith(black_box(suffix))
            keep(res)

    b.iter(call_fn)


# ===-----------------------------------------------------------------------===#
# Benchmark string _is_valid_utf8
# ===-----------------------------------------------------------------------===#
def bench_string_is_valid_utf8(mut b: Bencher, input: BenchInput) raises:
    var items = make_string(input.length, input.filename + ".html")

    @inline(.always)
    def call_fn() {imm}:
        var res = _is_valid_utf8(black_box(items).as_bytes())
        keep(res)

    b.iter(call_fn)


# ===-----------------------------------------------------------------------===#
# Benchmark write_utf8
# ===-----------------------------------------------------------------------===#
def bench_write_utf8(mut b: Bencher, input: BenchInput) raises:
    var items = make_string(input.length, input.filename + ".txt")
    var codepoints_iter = items.codepoints()
    # appending to a list to avoid paying the overhead of codepoint parsing
    var codepoints = List[Codepoint](capacity=len(codepoints_iter))
    for c in codepoints_iter:
        codepoints.append(c)

    @inline(.always)
    def call_fn() {imm}:
        var data = Array[Byte, 4](uninitialized=True)
        # this is to help with instability when measuring small strings
        for _ in range(10**6 // input.length):
            for i in range(len(codepoints)):
                var res = black_box(codepoints.unsafe_get(i)).unsafe_write_utf8(
                    black_box(data).unsafe_ptr()
                )
                keep(res)

    b.iter(call_fn)


# ===-----------------------------------------------------------------------===#
# Benchmark string write
# ===-----------------------------------------------------------------------===#
def bench_string_write[short: Bool](mut b: Bencher) raises:
    var items = make_string(1000, "UN_charter_EN.txt")
    # workaround for "allows writing to mem location ..."
    # even though I tried using an immutable StringSlice
    var items_2 = items.copy()
    var items_3 = items.copy()
    var items_4 = items.copy()
    var items_5 = items.copy()

    @inline(.always)
    def call_fn() {imm}:
        for _ in range(1_000_000):
            var res: String

            comptime if short:  # less than 24 bytes
                res = String(
                    black_box(0),
                    black_box(" is "),
                    black_box("a"),
                    black_box(String(" number")),
                )
            else:  # 5001 bytes long
                res = String(
                    black_box(0),
                    black_box(items),
                    black_box(items_2),
                    black_box(items_3),
                    black_box(items_4),
                    black_box(items_5),
                )
            keep(res)

    b.iter(call_fn)


# ===-----------------------------------------------------------------------===#
# Benchmark string repr
# ===-----------------------------------------------------------------------===#


@fieldwise_init
struct NullWriter(ImplicitlyCopyable, Writer):
    def write_string(mut self, string: StringSlice):
        keep(string)


def bench_string_repr(mut b: Bencher, input: BenchInput):
    var items = make_string(input.length, input.filename + ".txt")

    @inline(.always)
    def call_fn() {imm}:
        # this is to help with instability when measuring small strings
        for _ in range(10**6 // input.length):
            var writer = NullWriter()
            black_box(items).write_repr_to(writer)
            keep(writer)

    b.iter(call_fn)


# ===-----------------------------------------------------------------------===#
# Benchmark Main
# ===-----------------------------------------------------------------------===#
def _add_bench[
    F: def(mut Bencher, BenchInput) raises -> None
](mut m: Bench, func: F, name: StaticString, input: BenchInput) raises:
    def bench(mut b: Bencher) raises {imm func, imm input}:
        func(b, input)

    m.bench_function(bench, BenchId(String(name, "[", input.length, "]")))


def main() raises:
    seed()
    var m = Bench(BenchConfig(num_repetitions=1))
    var filenames: List[StaticString] = [
        "UN_charter_EN",
        "UN_charter_ES",
        "UN_charter_AR",
        "UN_charter_RU",
        "UN_charter_zh-CN",
    ]
    var old_chars: List[StaticString] = ["a", "ó", "ل", "и", "一"]
    var new_chars: List[StaticString] = ["A", "Ó", "ل", "И", "一"]

    # At an average 5 letters per word and 300 words per page (in the English
    # language):
    #
    # - 10: 2 words
    # - 30: 6 words
    # - 50: 10 words
    # - 100: 20 words
    # - 1000: ~ 1/2 page (200 words)
    # - 10_000: ~ 7 pages (2k words)
    # - 100_000: ~ 67 pages (20k words)
    # - 1_000_000: ~ 667 pages (200k words)
    var lengths = [10, 30, 50, 100, 1000, 10_000, 100_000, 1_000_000]

    m.bench_function(bench_string_init, BenchId("bench_string_init"))
    m.bench_function(
        bench_string_write[True], BenchId(String("bench_string_write_short"))
    )
    m.bench_function(
        bench_string_write[False], BenchId(String("bench_string_write_long"))
    )

    # Lengths and languages stay runtime values. As parameters, every
    # combination compiled its own copy of each benchmark below (600 in all),
    # which took this file over 10 minutes to build.
    for length in lengths:
        for j in range(len(filenames)):
            var input = BenchInput(
                length, filenames[j], old_chars[j], new_chars[j]
            )
            _add_bench(m, bench_string_count, "bench_string_count", input)
            _add_bench(m, bench_string_split, "bench_string_split", input)
            _add_bench(
                m, bench_string_split_none, "bench_string_split_none", input
            )
            _add_bench(
                m, bench_string_splitlines, "bench_string_splitlines", input
            )
            _add_bench(m, bench_string_lower, "bench_string_lower", input)
            _add_bench(m, bench_string_upper, "bench_string_upper", input)
            _add_bench(m, bench_string_replace, "bench_string_replace", input)
            _add_bench(
                m,
                bench_string_count_codepoints,
                "bench_string_count_codepoints",
                input,
            )
            _add_bench(
                m, bench_string_find_single, "bench_string_find_single", input
            )
            _add_bench(
                m,
                bench_string_find_multiple,
                "bench_string_find_multiple",
                input,
            )
            _add_bench(
                m, bench_string_startswith, "bench_string_startswith", input
            )
            _add_bench(m, bench_string_endswith, "bench_string_endswith", input)
            _add_bench(
                m,
                bench_string_is_valid_utf8,
                "bench_string_is_valid_utf8",
                input,
            )
            _add_bench(m, bench_write_utf8, "bench_write_utf8", input)
            _add_bench(m, bench_string_repr, "bench_string_repr", input)

    m.bench_function(
        bench_string_join[True],
        BenchId(String("bench_string_join_short")),
    )
    m.bench_function(
        bench_string_join[False],
        BenchId(String("bench_string_join_long")),
    )

    # NOTE: do not delete this. This is supposed to measure the average for
    # different languages. You can use print(m) if you wish to see the
    # per-language breakdown
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
