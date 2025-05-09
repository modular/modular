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

"""Tests for the regex module."""

from regex import compile, search, _match, fullmatch, split, sub
from collections import Optional
from testing import assert_true, assert_false, assert_equal


def test_simple_search():
    """Test basic search functionality."""
    var haystack = "hello world"
    var needle = "world"
    var result = search(needle, haystack)

    assert_true(
        result is not None, "Search should find 'world' in 'hello world'"
    )
    if result:
        assert_equal(
            result.value().start, 6, "Match should start at position 6"
        )
        assert_equal(result.value().end, 11, "Match should end at position 11")
        assert_equal(
            result.value().group(), "world", "Matched text should be 'world'"
        )


def test_case_insensitive_search():
    """Test case insensitive search."""
    var haystack = "Hello World"
    var needle = "world"

    var result1 = search(needle, haystack)
    assert_true(
        result1 is None,
        "Case-sensitive search should not find 'world' in 'Hello World'",
    )

    var result2 = search(needle, haystack, case_sensitive=False)
    assert_true(
        result2 is not None,
        "Case-insensitive search should find 'world' in 'Hello World'",
    )
    if result2:
        assert_equal(
            result2.value().start, 6, "Match should start at position 6"
        )
        assert_equal(
            result2.value().group(), "World", "Matched text should be 'World'"
        )


def test_match_start():
    """Test matching at the start of a string."""
    var text = "hello world"

    var result1 = _match("hello", text)
    assert_true(
        result1 is not None,
        "Should match 'hello' at the start of 'hello world'",
    )

    var result2 = _match("world", text)
    assert_true(
        result2 is None,
        "Should not match 'world' at the start of 'hello world'",
    )


def test_fullmatch():
    """Test matching the entire string."""
    var text = "hello world"

    var result1 = fullmatch("hello", text)
    assert_true(
        result1 is None, "Should not full match 'hello' with 'hello world'"
    )

    var result2 = fullmatch("hello world", text)
    assert_true(
        result2 is not None,
        "Should full match 'hello world' with 'hello world'",
    )


def test_split():
    """Test splitting a string by a pattern."""
    var text = "apple,banana,orange"
    var parts = split(",", text)

    assert_equal(len(parts), 3, "Split should result in 3 parts")
    assert_equal(parts[0], "apple", "First part should be 'apple'")
    assert_equal(parts[1], "banana", "Second part should be 'banana'")
    assert_equal(parts[2], "orange", "Third part should be 'orange'")


def test_replace():
    """Test replacing patterns in a string."""
    var text = "hello world, hello universe"
    var result = sub("hello", "hi", text)

    assert_equal(
        result,
        "hi world, hi universe",
        "Should replace all occurrences of 'hello' with 'hi'",
    )

    # Test limiting the number of replacements
    var result2 = sub("hello", "hi", text, count=1)
    assert_equal(
        result2,
        "hi world, hello universe",
        "Should replace only the first occurrence",
    )


def main():
    """Run all regex tests."""
    test_simple_search()
    test_case_insensitive_search()
    test_match_start()
    test_fullmatch()
    test_split()
    test_replace()
