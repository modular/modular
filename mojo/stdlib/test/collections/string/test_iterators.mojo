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

from std.collections.string.string_slice import (
    _to_string_list,
    get_static_string,
)
from std.sys.info import size_of, simd_width_of

from std.testing import assert_equal, assert_false, assert_true, assert_raises
from std.testing import TestSuite

# ===----------------------------------------------------------------------=== #
# Reusable testing data
# ===----------------------------------------------------------------------=== #

comptime EVERY_CODEPOINT_LENGTH_STR = StringSlice("߷കൈ🔄!")
"""A string that contains at least one of 1-, 2-, 3-, and 4-byte UTF-8
sequences.

Visualized as:

```text
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃                      ߷കൈ🔄!                    ┃
┣━━━━━━━┳━━━━━━━━━━━┳━━━━━━━━━━━┳━━━━━━━━━━━━━━━┳━━┫
┃   ߷  ┃     ക     ┃     ൈ    ┃       🔄      ┃! ┃
┣━━━━━━━╋━━━━━━━━━━━╋━━━━━━━━━━━╋━━━━━━━━━━━━━━━╋━━┫
┃ 2039  ┃   3349    ┃   3400    ┃    128260     ┃33┃
┣━━━┳━━━╋━━━┳━━━┳━━━╋━━━┳━━━┳━━━╋━━━┳━━━┳━━━┳━━━╋━━┫
┃223┃183┃224┃180┃149┃224┃181┃136┃240┃159┃148┃132┃33┃
┗━━━┻━━━┻━━━┻━━━┻━━━┻━━━┻━━━┻━━━┻━━━┻━━━┻━━━┻━━━┻━━┛
  0   1   2   3   4   5   6   7   8   9  10  11  12
```

For further visualization and analysis involving this sequence, see:
<https://connorgray.com/ephemera/project-log#2025-01-13>.
"""

# ===----------------------------------------------------------------------=== #
# Tests
# ===----------------------------------------------------------------------=== #


def test_chars_iter() raises:
    # Test `for` loop iteration support
    for char in StringSlice("abc").codepoints():
        assert_true(
            char in (Codepoint.ord("a"), Codepoint.ord("b"), Codepoint.ord("c"))
        )

    # Test empty string chars
    var s0 = StringSlice("")
    var s0_iter = s0.codepoints()

    with assert_raises():
        _ = s0_iter.__next__()  # raises StopIteration
    assert_true(s0_iter.peek_next() is None)
    assert_true(s0_iter.next() is None)

    # Test simple ASCII string chars
    var s1 = StringSlice("abc")
    var s1_iter = s1.codepoints()

    assert_equal(s1_iter.next().value(), Codepoint.ord("a"))
    assert_equal(s1_iter.next().value(), Codepoint.ord("b"))
    assert_equal(s1_iter.next().value(), Codepoint.ord("c"))
    assert_true(s1_iter.next() is None)

    # Multibyte character decoding: A visual character composed of a combining
    # sequence of 2 codepoints.
    var s2 = StringSlice("á")
    assert_equal(s2.byte_length(), 3)
    assert_equal(s2.count_codepoints(), 2)

    var iter = s2.codepoints()
    assert_equal(iter.__next__(), Codepoint.ord("a"))
    # U+0301 Combining Acute Accent
    assert_equal(iter.__next__().to_u32(), 0x0301)
    with assert_raises():
        _ = iter.__next__()  # raises StopIteration

    # A piece of text containing, 1-byte, 2-byte, 3-byte, and 4-byte codepoint
    # sequences.
    var s3 = EVERY_CODEPOINT_LENGTH_STR
    assert_equal(s3.byte_length(), 13)
    assert_equal(s3.count_codepoints(), 5)
    var s3_iter = s3.codepoints()

    # Iterator __len__ returns length in codepoints, not bytes.
    assert_equal(s3_iter.__len__(), 5)
    assert_equal(s3_iter._slice.byte_length(), 13)
    assert_equal(s3_iter.__next__(), Codepoint.ord("߷"))

    assert_equal(s3_iter.__len__(), 4)
    assert_equal(s3_iter._slice.byte_length(), 11)
    assert_equal(s3_iter.__next__(), Codepoint.ord("ക"))

    # Combining character, visually comes first, but codepoint-wise comes
    # after the character it combines with.
    assert_equal(s3_iter.__len__(), 3)
    assert_equal(s3_iter._slice.byte_length(), 8)
    assert_equal(s3_iter.__next__(), Codepoint.ord("ൈ"))

    assert_equal(s3_iter.__len__(), 2)
    assert_equal(s3_iter._slice.byte_length(), 5)
    assert_equal(s3_iter.__next__(), Codepoint.ord("🔄"))

    assert_equal(s3_iter.__len__(), 1)
    assert_equal(s3_iter._slice.byte_length(), 1)
    assert_equal(s3_iter.__next__(), Codepoint.ord("!"))

    assert_equal(s3_iter.__len__(), 0)
    assert_equal(s3_iter._slice.byte_length(), 0)
    with assert_raises():
        _ = s3_iter.__next__()  # raises StopIteration


def test_string_slice_codepoint_slices_reversed() raises:
    # Test ASCII
    var s: StaticString = "xyz"
    var iter = s.codepoint_slices_reversed()
    assert_equal(iter.__next__(), "z")
    assert_equal(iter.__next__(), "y")
    assert_equal(iter.__next__(), "x")

    # Test concatenation
    s = "abc"
    var concat = String()
    for v in s.codepoint_slices_reversed():
        concat += v
    assert_equal(concat, "cba")

    # Test Unicode
    s = "hello🌍"
    concat = String()
    for v in s.codepoint_slices_reversed():
        concat += v
    assert_equal(concat, "🌍olleh")

    # Test empty string
    s = ""
    concat = String()
    for v in s.codepoint_slices_reversed():
        concat += v
    assert_equal(concat, "")


def test_string_join_iterable() raises:
    var elems = [1, 2, 3]
    assert_equal("".join(iter(elems)), "123")
    assert_equal(",".join(iter(elems)), "1,2,3")
    assert_equal("".join("123".codepoints()), "123")
    assert_equal(",".join("123".codepoints()), "1,2,3")
    assert_equal("".join("123".codepoint_slices()), "123")
    assert_equal(",".join("123".codepoint_slices()), "1,2,3")


def test_string_join_iterable_owned() raises:
    assert_equal("".join(iter([1, 2, 3])), "123")
    assert_equal(",".join(iter([1, 2, 3])), "1,2,3")


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
