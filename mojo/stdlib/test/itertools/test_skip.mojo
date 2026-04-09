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

from std.itertools import skip
from std.testing import (
    TestSuite,
    assert_equal,
    assert_raises,
)


def test_skip_basic() raises:
    """Tests basic skip behavior."""
    var nums = [1, 2, 3, 4, 5]
    var it = skip(nums, 2)

    assert_equal(next(it), 3)
    assert_equal(next(it), 4)
    assert_equal(next(it), 5)
    with assert_raises(contains="StopIteration"):
        _ = next(it)


def test_skip_more_than_available() raises:
    """Tests skip when count exceeds iterable length."""
    var nums = [1, 2, 3]
    var it = skip(nums, 10)

    with assert_raises(contains="StopIteration"):
        _ = next(it)


def test_skip_zero() raises:
    """Tests skip with count=0 yields all elements."""
    var nums = [1, 2, 3]
    var it = skip(nums, 0)

    assert_equal(next(it), 1)
    assert_equal(next(it), 2)
    assert_equal(next(it), 3)
    with assert_raises(contains="StopIteration"):
        _ = next(it)


def test_skip_empty() raises:
    """Tests skip on an empty iterable."""
    var empty = List[Int]()
    var it = skip(empty, 5)

    with assert_raises(contains="StopIteration"):
        _ = next(it)


def test_skip_all() raises:
    """Tests skip with count equal to iterable length."""
    var nums = [10, 20, 30]
    var it = skip(nums, 3)

    with assert_raises(contains="StopIteration"):
        _ = next(it)


def test_skip_single() raises:
    """Tests skip with count=1."""
    var nums = [42, 99]
    var it = skip(nums, 1)

    assert_equal(next(it), 99)
    with assert_raises(contains="StopIteration"):
        _ = next(it)


def test_skip_in_for_loop() raises:
    """Tests skip iterator in a for loop."""
    var nums = [1, 2, 3, 4, 5]
    var results = List[Int]()

    for num in skip(nums, 2):
        results.append(num)

    assert_equal(len(results), 3)
    assert_equal(results[0], 3)
    assert_equal(results[1], 4)
    assert_equal(results[2], 5)


def test_skip_from_range() raises:
    """Tests skip on a range."""
    var it = skip(range(5), 3)

    assert_equal(next(it), 3)
    assert_equal(next(it), 4)
    with assert_raises(contains="StopIteration"):
        _ = next(it)


def test_skip_then_take_composition() raises:
    """Tests composing skip and take to select a middle slice."""
    from std.itertools import take

    var nums = [1, 2, 3, 4, 5, 6, 7]
    var results = List[Int]()

    # skip(1).take(3) equivalent: elements at indices 1, 2, 3
    for num in take(skip(nums, 1), 3):
        results.append(num)

    assert_equal(len(results), 3)
    assert_equal(results[0], 2)
    assert_equal(results[1], 3)
    assert_equal(results[2], 4)


def test_skip_bounds() raises:
    """Tests bounds() on a skip iterator."""
    var nums = [1, 2, 3, 4, 5]

    # count less than iterable length
    var it1 = skip(nums, 2)
    var lower1, upper1 = it1.bounds()
    assert_equal(lower1, 3)
    assert_equal(upper1.value(), 3)

    # count greater than iterable length
    var it2 = skip(nums, 10)
    var lower2, upper2 = it2.bounds()
    assert_equal(lower2, 0)
    assert_equal(upper2.value(), 0)

    # count == 0
    var it3 = skip(nums, 0)
    var lower3, upper3 = it3.bounds()
    assert_equal(lower3, 5)
    assert_equal(upper3.value(), 5)

    # empty iterable
    var empty = List[Int]()
    var it4 = skip(empty, 5)
    var lower4, upper4 = it4.bounds()
    assert_equal(lower4, 0)
    assert_equal(upper4.value(), 0)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
