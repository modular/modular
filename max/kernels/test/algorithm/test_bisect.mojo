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
"""`algorithm.bisect` semantics, checked against the insertion-point contract.

Every case also runs the `load(i)` form, so the span overloads cannot drift
from the device-load ones that the kernels call.
"""

from algorithm.bisect import bisect, bisect_left, bisect_right
from std.testing import TestSuite, assert_equal


def _check(values: List[Int], x: Int) raises:
    """Asserts both call shapes agree with the expected insertion points."""
    var span = ImmSpan[Int](values)
    var at = lambda (i: Int) -> Int: span[i]

    # Reference insertion points, computed the slow way.
    var expect_left = 0
    while expect_left < len(values) and values[expect_left] < x:
        expect_left += 1
    var expect_right = 0
    while expect_right < len(values) and values[expect_right] <= x:
        expect_right += 1

    assert_equal(bisect_left(span, x), expect_left)
    assert_equal(bisect_right(span, x), expect_right)
    assert_equal(bisect(span, x), expect_right)
    assert_equal(bisect_left(len(values), x, at), expect_left)
    assert_equal(bisect_right(len(values), x, at), expect_right)
    assert_equal(bisect(len(values), x, at), expect_right)


def test_bisect_distinct_values() raises:
    var values: List[Int] = [1, 3, 5, 7]
    for x in range(-1, 10):
        _check(values, x)


def test_bisect_duplicates() raises:
    var values: List[Int] = [1, 3, 3, 3, 5]
    for x in range(0, 7):
        _check(values, x)


def test_bisect_empty() raises:
    _check(List[Int](), 4)


def test_bisect_single() raises:
    var values: List[Int] = [4]
    for x in range(2, 7):
        _check(values, x)


def test_bisect_all_equal() raises:
    var values: List[Int] = [2, 2, 2, 2]
    for x in range(0, 5):
        _check(values, x)


def test_bisect_probe_ascending_matches_left() raises:
    var values: List[Int] = [1, 3, 3, 5, 7]
    var span = ImmSpan[Int](values)
    var x = 3

    var below_split = lambda (i: Int) {span, x} -> Bool: span[i] < x
    assert_equal(bisect_left(len(span), below_split), bisect_left(span, x))


def test_bisect_probe_descending_counts_above() raises:
    # A descending range: the probe is the caller's order, so the first index
    # whose value is no longer above `x` counts the values above it.
    var values: List[Int] = [9, 5, 4, 1]
    var span = ImmSpan[Int](values)
    var x = 4

    var above_split = lambda (i: Int) {span, x} -> Bool: span[i] > x
    assert_equal(bisect_left(len(span), above_split), 2)


def test_bisect_probe_all_true_and_all_false() raises:
    var yes = True
    var no = False

    var always = lambda (i: Int) {yes} -> Bool: yes
    var never = lambda (i: Int) {no} -> Bool: no

    assert_equal(bisect_left(5, always), 5)
    assert_equal(bisect_left(5, never), 0)
    assert_equal(bisect_left(0, always), 0)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
