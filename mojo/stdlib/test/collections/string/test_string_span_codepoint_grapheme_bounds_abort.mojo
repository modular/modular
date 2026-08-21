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
#
# Verifies that invalid `codepoint=`/`grapheme=` `ContiguousSlice` indexing
# (out of bounds, reversed, or negative) aborts on `StringSlice` and `String`
# instead of silently clamping. Mirrors
# `test_string_slice_bounds_abort.mojo`'s coverage of `byte=` slicing.
#
# ===----------------------------------------------------------------------=== #

from std.testing import _assert_aborts, TestSuite


def test_codepoint_end_oob() raises:
    # 3 codepoints: one flag emoji per codepoint slot.
    var s = StringSlice("🔄🔥🔄")

    def trigger() raises {s} -> None:
        _ = s[codepoint=0:100]

    _assert_aborts(
        trigger,
        contains="slice end index 100 is out of bounds, valid range is 0 to 3",
    )


def test_codepoint_reversed() raises:
    var s = StringSlice("🔄🔥🔄")

    def trigger() raises {s} -> None:
        _ = s[codepoint=3:1]

    _assert_aborts(
        trigger,
        contains="slice start index 3 is greater than slice end index 1",
    )


def test_codepoint_negative_start() raises:
    var s = StringSlice("🔄🔥🔄")

    def trigger() raises {s} -> None:
        _ = s[codepoint=-1:]

    _assert_aborts(
        trigger,
        contains="slice start index -1 is out of bounds, valid range is 0 to 3",
    )


def test_string_codepoint_end_oob() raises:
    # Through `String`'s `codepoint=` accessor, which delegates to
    # `StringSlice`.
    var s = String("🔄🔥🔄")

    def trigger() raises {s} -> None:
        _ = s[codepoint=0:100]

    _assert_aborts(
        trigger,
        contains="slice end index 100 is out of bounds, valid range is 0 to 3",
    )


def test_grapheme_end_oob() raises:
    # "cafe" + combining acute accent -- 4 graphemes, 5 codepoints.
    var s = StringSlice("café")

    def trigger() raises {s} -> None:
        _ = s[grapheme=0:100]

    _assert_aborts(
        trigger,
        contains="slice end index 100 is out of bounds, valid range is 0 to 4",
    )


def test_grapheme_reversed() raises:
    var s = StringSlice("café")

    def trigger() raises {s} -> None:
        _ = s[grapheme=3:1]

    _assert_aborts(
        trigger,
        contains="slice start index 3 is greater than slice end index 1",
    )


def test_grapheme_negative_start() raises:
    var s = StringSlice("café")

    def trigger() raises {s} -> None:
        _ = s[grapheme=-1:]

    _assert_aborts(
        trigger,
        contains="slice start index -1 is out of bounds, valid range is 0 to 4",
    )


def test_grapheme_start_oob() raises:
    var s = StringSlice("café")

    def trigger() raises {s} -> None:
        _ = s[grapheme=100:]

    _assert_aborts(
        trigger,
        contains="slice start index 100 is out of bounds, valid range is 0 to 4",
    )


def test_grapheme_end_negative() raises:
    var s = StringSlice("café")

    def trigger() raises {s} -> None:
        _ = s[grapheme=0:-1]

    _assert_aborts(
        trigger,
        contains="slice end index -1 is out of bounds, valid range is 0 to 4",
    )


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
