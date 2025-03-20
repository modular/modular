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
# RUN: %mojo %s

from testing import assert_equal, assert_false, assert_true, assert_not_equal
from collections.interval import Interval, IntervalTree


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
    assert_equal(repr(interval), "Interval(1, 10)")

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


struct MyType:
    var value: Float64

    fn __init__(out self):
        self.value = 0.0

    fn __init__(out self, value: Float64, /):
        self.value = value

    fn __copyinit__(out self, existing: Self, /):
        self.value = existing.value

    fn __moveinit__(out self, owned existing: Self, /):
        self.value = existing.value

    fn __gt__(self, other: Self) -> Bool:
        return self.value > other.value

    fn __lt__(self, other: Self) -> Bool:
        return self.value < other.value

    fn __ge__(self, other: Self) -> Bool:
        return self.value >= other.value

    fn __le__(self, other: Self) -> Bool:
        return self.value <= other.value

    fn __eq__(self, other: Self) -> Bool:
        return self.value == other.value

    fn __ne__(self, other: Self) -> Bool:
        return self.value != other.value

    fn __sub__(self, other: Self) -> Self:
        return Self(self.value - other.value)

    fn __int__(self) -> Int:
        return Int(self.value)

    fn __float__(self) -> Float64:
        return self.value

    fn write_to[W: Writer](self, mut writer: W):
        writer.write(self.value)

    fn __str__(self) -> String:
        return String.write(self)


def test_interval_floating():
    # Create an interval with floating point values using MyType wrapper.
    var interval = Interval(MyType(2.4), MyType(3.5))

    # Verify the interval start and end values are correctly set.
    assert_equal(rebind[Float64](interval.start.value), 2.4)
    assert_equal(rebind[Float64](interval.end.value), 3.5)

    # Test union operation with overlapping interval.
    var union = interval.union(Interval(MyType(3.0), MyType(4.5)))

    # Verify union produces expected interval bounds.
    assert_equal(union, Interval(MyType(2.4), MyType(4.5)))

    # Verify length of union interval is correct.
    assert_equal(len(union), 2)


def test_interval_tree():
    var tree = IntervalTree[Int, MyType]()
    tree.insert((15, 20), MyType(33.0))
    tree.insert((10, 30), MyType(34.0))
    tree.insert((17, 19), MyType(35.0))
    tree.insert((5, 20), MyType(36.0))
    tree.insert((12, 15), MyType(37.0))
    tree.insert((30, 40), MyType(38.0))
    print(tree)

    var elems = tree.search((10, 15))
    assert_equal(len(elems), 3)
    assert_equal(Float64(elems[0]), 34.0)
    assert_equal(Float64(elems[1]), 37.0)
    assert_equal(Float64(elems[2]), 36.0)


def main():
    test_interval()
    test_interval_floating()
    test_interval_tree()
