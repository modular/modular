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

from collections.interval import Interval, IntervalElement, IntervalTree
from testing import assert_equal, assert_false, assert_not_equal, assert_true
from testing import TestSuite


def test_interval():
    # Create an interval from 1 to 10 (exclusive)
    var interval = Interval(1, 10)

    # Test basic properties
    assert_equal(interval.start, 1)
    assert_equal(interval.end, 10)
    assert_equal(len(interval), 9)
    assert_equal(len(Interval(-10, -1)), 9)

    # Test string representations
    assert_equal(String(interval), "(1, 10)")
    assert_equal(repr(interval), "Interval[Int](1, 10)")

    # Test equality comparisons
    assert_equal(interval, Interval(1, 10))
    assert_not_equal(interval, Interval(1, 11))

    # Test less than comparisons
    assert_true(
        interval < Interval(2, 11), msg=String(interval, " < Interval(2, 11)")
    )
    assert_false(
        interval < Interval(1, 11), msg=String(interval, " < Interval(1, 11)")
    )

    # Test greater than comparisons
    assert_true(
        interval > Interval(0, 9), msg=String(interval, " > Interval(0, 9)")
    )
    assert_false(
        interval > Interval(1, 11), msg=String(interval, " > Interval(1, 11)")
    )

    # Test less than or equal comparisons
    assert_true(
        interval <= Interval(1, 10), msg=String(interval, " <= Interval(1, 10)")
    )
    assert_true(
        interval <= Interval(1, 11), msg=String(interval, " <= Interval(1, 11)")
    )
    assert_false(
        interval <= Interval(0, 9), msg=String(interval, " <= Interval(0, 9)")
    )

    # Test greater than or equal comparisons
    assert_true(
        interval >= Interval(1, 10), msg=String(interval, " >= Interval(1, 10)")
    )
    assert_true(
        interval >= Interval(2, 9), msg=String(interval, " >= Interval(2, 9)")
    )
    assert_false(
        interval >= Interval(1, 11), msg=String(interval, " >= Interval(1, 11)")
    )

    # Test interval containment
    assert_true(
        interval in Interval(1, 11), msg=String(interval, " in Interval(1, 11)")
    )
    assert_false(
        interval in Interval(1, 9), msg=String(interval, " in Interval(1, 9)")
    )
    assert_true(
        interval in Interval(1, 10), msg=String(interval, " in Interval(1, 10)")
    )
    assert_true(
        interval in Interval(1, 11), msg=String(interval, " in Interval(1, 11)")
    )
    assert_false(
        interval in Interval(1, 9), msg=String(interval, " in Interval(1, 9)")
    )

    # Test point containment
    assert_true(1 in interval, msg="1 in interval")
    assert_false(0 in interval)

    # Test interval overlap
    assert_true(interval.overlaps(Interval(1, 10)))
    assert_true(interval.overlaps(Interval(1, 9)))
    assert_false(interval.overlaps(Interval(-10, -1)))

    # Test interval union
    assert_equal(interval.union(Interval(1, 10)), Interval(1, 10))
    assert_equal(interval.union(Interval(1, 9)), Interval(1, 10))

    # Test interval intersection
    assert_equal(interval.intersection(Interval(1, 10)), Interval(1, 10))
    assert_equal(interval.intersection(Interval(1, 9)), Interval(1, 9))
    assert_equal(interval.intersection(Interval(3, 5)), Interval(3, 5))

    # Test empty interval checks
    assert_true(Bool(interval))
    assert_false(Bool(Interval(0, 0)))


struct MyType(
    Comparable,
    Floatable,
    ImplicitlyCopyable,
    IntervalElement,
    Stringable,
):
    var value: Float64

    fn __init__(out self):
        self.value = 0.0

    fn __init__(out self, value: Float64, /):
        self.value = value

    fn __lt__(self, other: Self) -> Bool:
        return self.value < other.value

    fn __eq__(self, other: Self) -> Bool:
        return self.value == other.value

    fn __sub__(self, other: Self) -> Self:
        return Self(self.value - other.value)

    fn __int__(self) -> Int:
        return Int(self.value)

    fn __float__(self) -> Float64:
        return self.value

    fn write_to(self, mut writer: Some[Writer]):
        writer.write(self.value)

    fn __str__(self) -> String:
        return String.write(self)


def test_interval_floating():
    # Create an interval with floating point values using MyType wrapper.
    var interval = Interval(MyType(2.4), MyType(3.5))

    # Verify the interval start and end values are correctly set.
    assert_equal(interval.start.value, 2.4)
    assert_equal(interval.end.value, 3.5)

    # Test union operation with overlapping interval.
    var union = interval.union(Interval(MyType(3.0), MyType(4.5)))

    # Verify union produces expected interval bounds.
    assert_equal(union, Interval(MyType(2.4), MyType(4.5)))

    # Verify length of union interval is correct.
    assert_equal(len(union), 2)


def test_interval_write_repr():
    var interval = Interval(1, 10)

    # test write_to (string form)
    assert_equal(String(interval), "(1, 10)")

    # test write_repr_to (debug form)
    assert_equal(repr(interval), "Interval[Int](1, 10)")


def test_interval_tree_write_repr():
    var tree = IntervalTree[Int, MyType]()
    tree.insert((1, 5), MyType(1.0))

    # basic sanity: repr should contain typename and len
    var r = repr(tree)
    assert_true("IntervalTree[Int, MyType]" in r)
    assert_true("len=1" in r)


def main():
    TestSuite.discover_tests[__functions_in_module()]().run()
