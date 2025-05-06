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


from json import loads
from json.parser import JSONValue
from testing import assert_equal, assert_true, assert_false, assert_raises


def test_loads():
    var o1 = loads(
        """
    {
        "name": "John",
        "age": 30,
        "city": "New York"
    }"""
    )
    assert_equal(
        String(o1), '{"name": "John", "age": 30.0, "city": "New York"}'
    )

    assert_equal(String(loads("[1.0, 2.0, 3.0]")), "[1.0, 2.0, 3.0]")

    assert_equal(String(loads("true")), "true")

    assert_equal(String(loads("false")), "false")

    assert_equal(String(loads("null")), "null")

    assert_equal(String(loads("123")), "123.0")

    assert_equal(String(loads("123.456")), "123.456")

    assert_equal(String(loads("123.456e3")), "123456.0")


def test_primitive_serialization():
    """Test serialization of primitive JSON values."""
    # Null
    assert_equal(String(loads("null")), "null")

    # Booleans
    assert_equal(String(loads("true")), "true")
    assert_equal(String(loads("false")), "false")

    # Numbers
    assert_equal(String(loads("42")), "42.0")
    assert_equal(String(loads("3.14159")), "3.14159")
    assert_equal(String(loads("1.23e-4")), "0.000123")

    # String
    assert_equal(String(loads('"hello"')), '"hello"')


def test_array_serialization():
    """Test serialization of JSON arrays."""
    # Empty array
    assert_equal(String(loads("[]")), "[]")

    # Simple array
    assert_equal(String(loads("[1, 2, 3]")), "[1.0, 2.0, 3.0]")

    # Mixed array
    assert_equal(
        String(loads('[1, "two", true, null]')), '[1.0, "two", true, null]'
    )

    # Nested array
    assert_equal(String(loads("[[1, 2], [3, 4]]")), "[[1.0, 2.0], [3.0, 4.0]]")


def test_object_serialization():
    """Test serialization of JSON objects."""
    # Empty object
    assert_equal(String(loads("{}")), "{}")

    # Simple object
    assert_equal(String(loads('{"key": "value"}')), '{"key": "value"}')

    # Object with mixed values
    var mixed_obj = loads(
        """
    {
        "string": "value",
        "number": 42,
        "boolean": true,
        "null": null,
        "array": [1, 2, 3]
    }
    """
    )

    var serialized = String(mixed_obj)

    # Check that all properties are present
    assert_true('"string": "value"' in serialized)
    assert_true('"number": 42.0' in serialized)
    assert_true('"boolean": true' in serialized)
    assert_true('"null": null' in serialized)
    assert_true('"array": [1.0, 2.0, 3.0]' in serialized)


def test_nested_structures():
    """Test serialization of deeply nested JSON structures."""
    var nested_json = """
    {
        "level1": {
            "level2": {
                "level3": {
                    "level4": {
                        "array": [1, 2, [3, 4, [5, 6]]],
                        "object": {"key": "value"}
                    }
                }
            }
        }
    }
    """

    var parsed = loads(nested_json)
    var serialized = String(parsed)

    # Re-parse the serialized output to ensure it's valid JSON
    var reparsed = loads(serialized)

    # The re-parsed value should be equivalent to the original
    assert_equal(String(reparsed), serialized)


def test_invalid_json():
    with assert_raises(contains="Empty JSON input"):
        _ = loads("")

    with assert_raises(contains="Empty JSON input (only whitespace)"):
        _ = loads(" ")

    with assert_raises(
        contains="Invalid JSON value starting with: 'a' at position 0"
    ):
        _ = loads("a")

    with assert_raises(
        contains=(
            "Unexpected trailing content after JSON value: 'a' at position 3"
        )
    ):
        _ = loads("123a")

    with assert_raises(contains="Invalid JSON value"):
        _ = loads("[")


def main():
    test_loads()
    test_primitive_serialization()
    test_array_serialization()
    test_object_serialization()
    test_nested_structures()
    test_invalid_json()
