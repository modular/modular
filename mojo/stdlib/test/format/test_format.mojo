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
from test_utils.reflection import SimplePoint, NestedStruct, EmptyStruct


@fieldwise_init
struct TestWritable(Writable):
    var x: Int

    fn write_to(self, mut writer: Some[Writer]):
        writer.write("write_to: ", self.x)

    fn write_repr_to(self, mut writer: Some[Writer]):
        writer.write("write_repr_to: ", self.x)


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
    assert_equal(String(p), "SimplePoint(x=1, y=2)")
    assert_equal(repr(p), "SimplePoint(x=Int(1), y=Int(2))")


def test_default_write_to_nested():
    """Test the reflection-based default write_to with nested structs."""
    var s = NestedStruct(SimplePoint(3, 4), "test")
    # Note: String's write_repr_to doesn't add quotes (write_to is same as write_repr_to for String)
    assert_equal(
        String(s),
        "NestedStruct(point=SimplePoint(x=3, y=4), name=test)",
    )
    assert_equal(
        repr(s),
        "NestedStruct(point=SimplePoint(x=Int(3), y=Int(4)), name='test')",
    )


def test_default_write_to_empty():
    """Test the reflection-based default write_to with an empty struct."""
    var e = EmptyStruct()
    assert_equal(String(e), "EmptyStruct()")
    assert_equal(repr(e), "EmptyStruct()")


def main():
    TestSuite.discover_tests[__functions_in_module()]().run()
