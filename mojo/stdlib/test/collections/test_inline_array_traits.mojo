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

from collections import InlineArray

from testing import assert_equal, assert_false, assert_raises, assert_true
from testing import TestSuite

# ===-----------------------------------------------------------------------===#
# String representation tests
# ===-----------------------------------------------------------------------===#


def test_str_and_repr():
    """Test __str__ and __repr__ methods."""
    var array = InlineArray[Int, 3](1, 2, 3)

    # Test __str__ method
    var str_result = array.__str__()
    assert_equal(str_result, "InlineArray[Int, 3](1, 2, 3)")

    # Test __repr__ method
    var repr_result = array.__repr__()
    assert_equal(repr_result, "InlineArray[Int, 3](1, 2, 3)")

    # They should be equal
    assert_equal(str_result, repr_result)


def test_different_types():
    """Test with different element types."""
    # Test with String
    var string_array = InlineArray[String, 2]("hello", "world")
    var str_result = string_array.__str__()
    assert_equal(str_result, "InlineArray[String, 2]('hello', 'world')")

    # Test with single element
    var single = InlineArray[Int, 1](42)
    var single_str = single.__str__()
    assert_equal(single_str, "InlineArray[Int, 1](42)")

    # Test with default values
    var default_array = InlineArray[Int, 3](fill=0)
    var default_str = default_array.__str__()
    assert_equal(default_str, "InlineArray[Int, 3](0, 0, 0)")

    # Test with filled array
    var filled_array = InlineArray[Int, 4](fill=99)
    var filled_str = filled_array.__str__()
    assert_equal(filled_str, "InlineArray[Int, 4](99, 99, 99, 99)")


def test_write_to():
    """Test Writable trait implementation."""
    var array = InlineArray[Int, 3](10, 20, 30)
    var output = String()
    array.write_to(output)

    assert_equal(output, "InlineArray[Int, 3](10, 20, 30)")


# ===-----------------------------------------------------------------------===#
# Main
# ===-----------------------------------------------------------------------===#


fn main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()