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

from collections import Optional
from collections.string._utf8 import _is_valid_utf8
from collections.string.string_slice import _split
from os import abort
from pathlib import _dir_of_current_file
from random import seed
from sys import stderr

from benchmark import Bench, BenchConfig, Bencher, BenchId, keep


# ===-----------------------------------------------------------------------===#
# Benchmark Data
# ===-----------------------------------------------------------------------===#
fn make_string[
    length: Int = 0
](filename: String = "UN_charter_EN.txt") -> String:
    """Make a `String` made of items in the `./data` directory.

    Parameters:
        length: The length in bytes of the resulting `String`. If == 0 -> the
            whole file content.

    Args:
        filename: The name of the file inside the `./data` directory.
    """

    try:
        directory = _dir_of_current_file() / "data"
        var f = open(directory / filename, "r")

        @parameter
        if length > 0:
            var items = f.read_bytes(length)
            i = 0
            while length > len(items):
                items.append(items[i])
                i = i + 1 if i < len(items) - 1 else 0
            return String(bytes=items)
        else:
            return String(bytes=f.read_bytes())
    except e:
        print(e, file=stderr)
    return abort[String]()


# ===-----------------------------------------------------------------------===#
# Benchmark string init
# ===-----------------------------------------------------------------------===#
@parameter
fn bench_string_init(mut b: Bencher) raises:
    @always_inline
    @parameter
    fn call_fn():
        for _ in range(1000):
            var d = String()
            keep(d.unsafe_ptr())

    b.iter[call_fn]()


# ===-----------------------------------------------------------------------===#
# Benchmark string count
# ===-----------------------------------------------------------------------===#
@parameter
fn bench_string_count[
    length: Int = 0,
    filename: StaticString = "UN_charter_EN",
    sequence: StaticString = "a",
](mut b: Bencher) raises:
    var items = make_string[length](filename + ".txt")

    @always_inline
    @parameter
    fn call_fn() raises:
        var amnt = items.count(sequence)
        keep(amnt)

    b.iter[call_fn]()
    keep(Bool(items))


# ===-----------------------------------------------------------------------===#
# Benchmark string split
# ===-----------------------------------------------------------------------===#
@parameter
fn bench_string_split[
    length: Int = 0,
    filename: StaticString = "UN_charter_EN",
    sequence: Optional[StaticString] = None,
](mut b: Bencher) raises:
    var items = (
        make_string[length](filename + ".txt").as_string_slice().get_immutable()
    )

    @always_inline
    @parameter
    fn call_fn() raises:
        var res: List[__type_of(items)]

        @parameter
        if sequence:
            res = _split[has_maxsplit=False](items, sequence.value(), -1)
        else:
            res = _split[has_maxsplit=False](items, None, -1)
        keep(res.unsafe_ptr())

    b.iter[call_fn]()
    keep(Bool(items))


# ===-----------------------------------------------------------------------===#
# Benchmark string join
# ===-----------------------------------------------------------------------===#
@parameter
fn bench_string_join[short: Bool](mut b: Bencher) raises:
    @parameter
    if short:
        count = 100
    else:
        count = 1000

    word_list = List[String](capacity=count)
    for i in range(count):
        word_list.append(String(i))

    @always_inline
    @parameter
    fn call_fn() raises:
        for _ in range(1_000):
            res = String(",").join(word_list)
            keep(Bool(res))

    b.iter[call_fn]()
    keep(Bool(word_list))


# ===-----------------------------------------------------------------------===#
# Benchmark string splitlines
# ===-----------------------------------------------------------------------===#
@parameter
fn bench_string_splitlines[
    length: Int = 0, filename: StaticString = "UN_charter_EN"
](mut b: Bencher) raises:
    var items = make_string[length](filename + ".txt").as_string_slice()

    @always_inline
    @parameter
    fn call_fn() raises:
        for _ in range(1_000_000 // length):
            var res = items.splitlines()
            keep(res.unsafe_ptr())

    b.iter[call_fn]()
    keep(Bool(items))


# ===-----------------------------------------------------------------------===#
# Benchmark string lower
# ===-----------------------------------------------------------------------===#
@parameter
fn bench_string_lower[
    length: Int = 0, filename: StaticString = "UN_charter_EN"
](mut b: Bencher) raises:
    var items = make_string[length](filename + ".txt")

    @always_inline
    @parameter
    fn call_fn() raises:
        var res = items.lower()
        keep(res.unsafe_ptr())

    b.iter[call_fn]()
    keep(Bool(items))


# ===-----------------------------------------------------------------------===#
# Benchmark string upper
# ===-----------------------------------------------------------------------===#
@parameter
fn bench_string_upper[
    length: Int = 0, filename: StaticString = "UN_charter_EN"
](mut b: Bencher) raises:
    var items = make_string[length](filename + ".txt")

    @always_inline
    @parameter
    fn call_fn() raises:
        var res = items.upper()
        keep(res.unsafe_ptr())

    b.iter[call_fn]()
    keep(Bool(items))


# ===-----------------------------------------------------------------------===#
# Benchmark string replace
# ===-----------------------------------------------------------------------===#
@parameter
fn bench_string_replace[
    length: Int = 0,
    filename: StaticString = "UN_charter_EN",
    old: StaticString = "a",
    new: StaticString = "A",
](mut b: Bencher) raises:
    var items = make_string[length](filename + ".txt")

    @always_inline
    @parameter
    fn call_fn() raises:
        var res = items.replace(old, new)
        keep(res.unsafe_ptr())

    b.iter[call_fn]()
    keep(Bool(items))


# ===-----------------------------------------------------------------------===#
# Benchmark string find single
# ===-----------------------------------------------------------------------===#
@parameter
fn bench_string_find_single[
    length: Int = 0, filename: StaticString = "UN_charter_EN"
](mut b: Bencher) raises:
    var items = make_string[length](filename + ".txt")

    @always_inline
    @parameter
    fn call_fn() raises:
        # this is to help with instability when measuring small strings
        for _ in range(10**6 // length):
            var res = items.find("Z")  # something that probably won't be there
            keep(res)

    b.iter[call_fn]()
    keep(Bool(items))


# ===-----------------------------------------------------------------------===#
# Benchmark string find multiple
# ===-----------------------------------------------------------------------===#
@parameter
fn bench_string_find_multiple[
    length: Int = 0, filename: StaticString = "UN_charter_EN"
](mut b: Bencher) raises:
    var items = make_string[length](filename + ".txt")
    var sequence = "ZZZZ"  # something that probably won't be there

    @always_inline
    @parameter
    fn call_fn() raises:
        # this is to help with instability when measuring small strings
        for _ in range(10**6 // length):
            var res = items.find(sequence)
            keep(res)

    b.iter[call_fn]()
    keep(Bool(items))
    keep(Bool(sequence))


# ===-----------------------------------------------------------------------===#
# Benchmark string _is_valid_utf8
# ===-----------------------------------------------------------------------===#
@parameter
fn bench_string_is_valid_utf8[
    length: Int = 0, filename: StaticString = "UN_charter_EN"
](mut b: Bencher) raises:
    var items = make_string[length](filename + ".html")

    @always_inline
    @parameter
    fn call_fn() raises:
        var res = _is_valid_utf8(items.as_bytes())
        keep(res)

    b.iter[call_fn]()
    keep(Bool(items))


# ===-----------------------------------------------------------------------===#
# Benchmark write_utf8
# ===-----------------------------------------------------------------------===#
@parameter
fn bench_write_utf8[
    length: Int = 0, filename: StaticString = "UN_charter_EN"
](mut b: Bencher) raises:
    var items = make_string[length](filename + ".txt")
    var codepoints_iter = items.codepoints()
    # appending to a list to avoid paying the overhead of codepoint parsing
    var codepoints = List[Codepoint](capacity=len(codepoints_iter))
    for c in codepoints_iter:
        codepoints.append(c)

    @always_inline
    @parameter
    fn call_fn() raises:
        var data = InlineArray[Byte, 4](uninitialized=True)
        # this is to help with instability when measuring small strings
        for _ in range(10**6 // length):
            for i in range(len(codepoints)):
                var res = codepoints.unsafe_get(i).unsafe_write_utf8(
                    data.unsafe_ptr()
                )
                keep(Bool(res))

    b.iter[call_fn]()
    keep(Bool(items))
    keep(Bool(codepoints))


# ===-----------------------------------------------------------------------===#
# Benchmark string write
# ===-----------------------------------------------------------------------===#
@parameter
fn bench_string_write[short: Bool](mut b: Bencher) raises:
    var items = make_string[1000]("UN_charter_EN.txt")
    # workaround for "allows writing to mem location ..."
    # even though I tried using an immutable StringSlice
    var items_2 = items.copy()
    var items_3 = items.copy()
    var items_4 = items.copy()
    var items_5 = items.copy()

    @always_inline
    @parameter
    fn call_fn() raises:
        for _ in range(1_000_000):
            var res: String

            @parameter
            if short:  # less than 24 bytes
                res = String.write(0, " is ", "a", String(" number"))
            else:  # 5001 bytes long
                res = String.write(0, items, items_2, items_3, items_4, items_5)
            keep(Bool(res))

    b.iter[call_fn]()
    keep(Bool(items))
    keep(Bool(items_2))
    keep(Bool(items_3))
    keep(Bool(items_4))
    keep(Bool(items_5))


# ===-----------------------------------------------------------------------===#
# Benchmark Main
# ===-----------------------------------------------------------------------===#
def main():
    seed()
    var m = Bench(BenchConfig(num_repetitions=1))
    alias filenames = (
        StaticString("UN_charter_EN"),
        StaticString("UN_charter_ES"),
        StaticString("UN_charter_AR"),
        StaticString("UN_charter_RU"),
        StaticString("UN_charter_zh-CN"),
    )
    alias old_chars = (
        StaticString("a"),
        StaticString("ó"),
        StaticString("ل"),
        StaticString("и"),
        StaticString("一"),
    )
    alias new_chars = (
        StaticString("A"),
        StaticString("Ó"),
        StaticString("ل"),
        StaticString("И"),
        StaticString("一"),
    )

    alias lengths = (10, 30, 50, 100, 1000, 10_000, 100_000, 1_000_000)
    """At an average 5 letters per word and 300 words per page
    (in the English language):

    - 10: 2 words
    - 30: 6 words
    - 50: 10 words
    - 100: 20 words
    - 1000: ~ 1/2 page (200 words)
    - 10_000: ~ 7 pages (2k words)
    - 100_000: ~ 67 pages (20k words)
    - 1_000_000: ~ 667 pages (200k words)
    """

    m.bench_function[bench_string_init](BenchId("bench_string_init"))
    m.bench_function[bench_string_write[True]](
        BenchId(String("bench_string_write_short"))
    )
    m.bench_function[bench_string_write[False]](
        BenchId(String("bench_string_write_long"))
    )

    @parameter
    for i in range(len(lengths)):
        alias length = lengths[i]

        @parameter
        for j in range(len(filenames)):
            alias fname = filenames[j]
            alias old = StaticString(old_chars[j])
            alias new = new_chars[j]
            alias suffix = String("[", length, "]")  # "(" + fname + ")"
            m.bench_function[bench_string_count[length, fname, old]](
                BenchId(String("bench_string_count", suffix))
            )
            m.bench_function[bench_string_split[length, fname, old]](
                BenchId(String("bench_string_split", suffix))
            )
            m.bench_function[bench_string_split[length, fname]](
                BenchId(String("bench_string_split_none", suffix))
            )
            m.bench_function[bench_string_splitlines[length, fname]](
                BenchId(String("bench_string_splitlines", suffix))
            )
            m.bench_function[bench_string_lower[length, fname]](
                BenchId(String("bench_string_lower", suffix))
            )
            m.bench_function[bench_string_upper[length, fname]](
                BenchId(String("bench_string_upper", suffix))
            )
            m.bench_function[bench_string_replace[length, fname, old, new]](
                BenchId(String("bench_string_replace", suffix))
            )
            m.bench_function[bench_string_find_single[length, fname]](
                BenchId(String("bench_string_find_single", suffix))
            )
            m.bench_function[bench_string_find_multiple[length, fname]](
                BenchId(String("bench_string_find_multiple", suffix))
            )
            m.bench_function[bench_string_is_valid_utf8[length, fname]](
                BenchId(String("bench_string_is_valid_utf8", suffix))
            )
            m.bench_function[bench_write_utf8[length, fname]](
                BenchId(String("bench_write_utf8", suffix))
            )

    m.bench_function[bench_string_join[True]](
        BenchId(String("bench_string_join_short"))
    )
    m.bench_function[bench_string_join[False]](
        BenchId(String("bench_string_join_long"))
    )

    print(m)
