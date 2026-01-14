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
from testing import *


@fieldwise_init
struct TestWritable(Writable):
    var x: Int

    fn write_to(self, mut writer: Some[Writer]):
        writer.write("write_to: ", self.x)

    fn write_repr_to(self, mut writer: Some[Writer]):
        writer.write("write_repr_to: ", self.x)


# Test structs for reflection-based default write_to and __eq__
@fieldwise_init
struct SimplePoint(Equatable, ImplicitlyCopyable, Writable):
    """A simple struct that uses default reflection-based write_to and __eq__.
    """

    var x: Int
    var y: Int

    # Uses default reflection-based write_to from Writable trait
    # Uses default reflection-based __eq__ from Equatable trait


@fieldwise_init
struct NestedStruct(Equatable, ImplicitlyCopyable, Writable):
    """A struct with nested fields using default write_to and __eq__."""

    var point: SimplePoint
    var name: String

    # Uses default reflection-based write_to from Writable trait
    # Uses default reflection-based __eq__ from Equatable trait


@fieldwise_init
struct EmptyStruct(Equatable, ImplicitlyCopyable, Writable):
    """A struct with no fields."""

    pass
    # Uses default reflection-based write_to from Writable trait
    # Uses default reflection-based __eq__ from Equatable trait


def test_repr():
    var t = TestWritable(42)
    assert_equal(repr(t), "write_repr_to: 42")


def test_string_constructor():
    var s = String(TestWritable(42))
    assert_equal(s, "write_to: 42")


def test_format_string():
    assert_equal("{}".format(TestWritable(42)), "write_to: 42")
    assert_equal(String("{}").format(TestWritable(42)), "write_to: 42")
    assert_equal(StringSlice("{}").format(TestWritable(42)), "write_to: 42")

    assert_equal("{!r}".format(TestWritable(42)), "write_repr_to: 42")
    assert_equal(String("{!r}").format(TestWritable(42)), "write_repr_to: 42")
    assert_equal(
        StringSlice("{!r}").format(TestWritable(42)), "write_repr_to: 42"
    )


def test_default_write_to_simple():
    """Test the reflection-based default write_to with a simple struct."""
    var p = SimplePoint(1, 2)
    # Note: get_type_name returns module-qualified names
    assert_equal(String(p), "test_format.SimplePoint(x=1, y=2)")
    assert_equal(repr(p), "test_format.SimplePoint(x=1, y=2)")


def test_default_write_to_nested():
    """Test the reflection-based default write_to with nested structs."""
    var s = NestedStruct(SimplePoint(3, 4), "test")
    # Note: String's write_repr_to doesn't add quotes (write_to is same as write_repr_to for String)
    assert_equal(
        String(s),
        (
            "test_format.NestedStruct(point=test_format.SimplePoint(x=3, y=4),"
            " name=test)"
        ),
    )


def test_default_write_to_empty():
    """Test the reflection-based default write_to with an empty struct."""
    var e = EmptyStruct()
    assert_equal(String(e), "test_format.EmptyStruct()")


def test_default_eq_simple():
    """Test the reflection-based default __eq__ with a simple struct."""
    var p1 = SimplePoint(1, 2)
    var p2 = SimplePoint(1, 2)
    var p3 = SimplePoint(1, 3)
    var p4 = SimplePoint(2, 2)

    assert_true(p1 == p2)
    assert_false(p1 != p2)
    assert_false(p1 == p3)
    assert_true(p1 != p3)
    assert_false(p1 == p4)


def test_default_eq_nested():
    """Test the reflection-based default __eq__ with nested structs."""
    var s1 = NestedStruct(SimplePoint(1, 2), "hello")
    var s2 = NestedStruct(SimplePoint(1, 2), "hello")
    var s3 = NestedStruct(SimplePoint(1, 2), "world")
    var s4 = NestedStruct(SimplePoint(3, 4), "hello")

    assert_true(s1 == s2)
    assert_false(s1 == s3)
    assert_false(s1 == s4)


def test_default_eq_empty():
    """Test the reflection-based default __eq__ with an empty struct."""
    var e1 = EmptyStruct()
    var e2 = EmptyStruct()

    assert_true(e1 == e2)


def main():
    TestSuite.discover_tests[__functions_in_module()]().run()
