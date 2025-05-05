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


def test_whitespace_handling():
    """Test whitespace handling."""
    # Leading/trailing whitespace should be ignored
    assert_equal(String(loads(" \t\n\r{}\n\r\t ")), "{}")

    # Whitespace between elements should be ignored
    assert_equal(String(loads("[1, \n\t 2, \r3]")), "[1.0, 2.0, 3.0]")

    # Whitespace in object keys/values should be handled
    assert_equal(String(loads('{ "key" : \t\n "value" }')), '{"key": "value"}')

    # Only whitespace
    try:
        _ = loads("   \t\n\r   ")
        assert_true(
            False, "Should have raised an error for whitespace-only input"
        )
    except:
        # Any error is OK
        pass


def test_complex_json():
    """Test parsing of complex JSON with all types."""
    var complex_json = """
    {
        "string": "Hello, world!",
        "number": 123.456,
        "integer": 42,
        "scientific": 1.23e-4,
        "boolean_true": true,
        "boolean_false": false,
        "null_value": null,
        "array": [1, 2, 3, 4, 5],
        "nested_object": {
            "key1": "value1",
            "key2": "value2"
        },
        "mixed_array": [
            "string",
            123,
            true,
            null,
            { "nested": "object" }
        ]
    }
    """

    var parsed = loads(complex_json)

    # Convert to string and parse again to verify serialization/deserialization
    var json_str = String(parsed)
    var reparsed = loads(json_str)

    # The reparsed value should be equivalent to the original parsed value
    assert_equal(String(reparsed), String(parsed))


def test_object_and_arrays():
    """Test parsing of objects and arrays."""
    # Empty object
    assert_equal(String(loads("{}")), "{}")

    # Empty array
    assert_equal(String(loads("[]")), "[]")

    # Simple object
    assert_equal(String(loads('{"key": "value"}')), '{"key": "value"}')

    # Simple array
    assert_equal(String(loads("[1, 2, 3]")), "[1.0, 2.0, 3.0]")


def test_real_world_json_formats():
    """Test parsing of real-world JSON formats and standards."""

    # GeoJSON example
    var geojson = """
    {
        "type": "Feature",
        "geometry": {
            "type": "Point",
            "coordinates": [125.6, 10.1]
        },
        "properties": {
            "name": "Dinagat Islands"
        }
    }
    """
    var parsed_geojson = loads(geojson)
    assert_true('"type": "Feature"' in String(parsed_geojson))
    assert_true('"coordinates": [125.6, 10.1]' in String(parsed_geojson))

    # JSON-API example
    var jsonapi = """
    {
        "data": [{
            "type": "articles",
            "id": "1",
            "attributes": {
                "title": "JSON:API paints my bike red!",
                "body": "The shortest article. Ever."
            },
            "relationships": {
                "author": {
                    "data": {"id": "42", "type": "people"}
                }
            }
        }],
        "included": [
            {
                "type": "people",
                "id": "42",
                "attributes": {
                    "name": "John",
                    "age": 80
                }
            }
        ]
    }
    """
    var parsed_jsonapi = loads(jsonapi)
    assert_true('"type": "articles"' in String(parsed_jsonapi))
    assert_true('"id": "42"' in String(parsed_jsonapi))

    # JSON Web Token (JWT) payload example
    var jwt_payload = """
    {
        "sub": "1234567890",
        "name": "John Doe",
        "admin": true,
        "iat": 1516239022,
        "exp": 1516242622
    }
    """
    var parsed_jwt = loads(jwt_payload)
    assert_true('"sub": "1234567890"' in String(parsed_jwt))
    assert_true('"admin": true' in String(parsed_jwt))

    # Package.json example
    var package_json = """
    {
        "name": "my-package",
        "version": "1.0.0",
        "description": "A package for testing JSON parsing",
        "main": "index.js",
        "scripts": {
            "test": "echo \\"Error: no test specified\\" && exit 1",
            "start": "node index.js"
        },
        "keywords": [
            "test",
            "json",
            "parser"
        ],
        "author": "Test Author",
        "license": "MIT",
        "dependencies": {
            "express": "^4.17.1",
            "lodash": "^4.17.21"
        }
    }
    """
    var parsed_package = loads(package_json)
    assert_true('"name": "my-package"' in String(parsed_package))
    assert_true('"express": "^4.17.1"' in String(parsed_package))


def test_number_edge_cases():
    """Test parsing of number edge cases in JSON."""
    # Integer edge cases
    assert_equal(String(loads("0")), "0.0")  # Zero
    assert_equal(
        String(loads("123456789012345678")), "1.234567890123457e+17"
    )  # Large integer
    assert_equal(
        String(loads("-123456789012345678")), "-1.234567890123457e+17"
    )  # Large negative

    # Decimal edge cases
    assert_equal(String(loads("0.0")), "0.0")  # Zero decimal
    assert_equal(String(loads("0.000000000000001")), "1e-15")  # Small decimal
    assert_equal(
        String(loads("-0.000000000000001")), "-1e-15"
    )  # Small negative decimal
    assert_equal(
        String(loads("1.0000000000000001")), "1.0"
    )  # Decimal with precision loss

    # Scientific notation edge cases
    assert_equal(String(loads("1e0")), "1.0")  # Power of 0
    assert_equal(String(loads("1e+0")), "1.0")  # Explicit positive power of 0
    assert_equal(String(loads("1e-0")), "1.0")  # Negative power of 0
    assert_equal(String(loads("1.23E+5")), "123000.0")  # Capital E

    # Precision boundary cases
    assert_equal(
        String(loads("9007199254740991")), "9007199254740991.0"
    )  # Max safe integer
    assert_equal(
        String(loads("-9007199254740991")), "-9007199254740991.0"
    )  # Min safe integer

    # Invalid formats should fail
    with assert_raises():
        _ = loads("1e")  # Missing exponent
    with assert_raises():
        _ = loads("1e+")  # Missing exponent after sign
    with assert_raises():
        _ = loads("1.2e-z")  # Invalid character in exponent


def test_empty_structures():
    """Test parsing of empty JSON structures and whitespace handling."""
    # Empty object with various whitespace
    assert_equal(String(loads("{}")), "{}")
    assert_equal(String(loads(" { } ")), "{}")
    assert_equal(String(loads("\n{\r\n}\t")), "{}")

    # Empty array with various whitespace
    assert_equal(String(loads("[]")), "[]")
    assert_equal(String(loads(" [ ] ")), "[]")
    assert_equal(String(loads("\n[\r\n]\t")), "[]")

    # Nested empty structures
    assert_equal(String(loads('{"empty_array":[]}')), '{"empty_array": []}')
    assert_equal(String(loads("[{}]")), "[{}]")
    assert_equal(String(loads('{"empty_object":{}}')), '{"empty_object": {}}')
    assert_equal(String(loads("[[]]")), "[[]]")

    # Complex nested empty structures
    assert_equal(
        String(loads('{"a":{},"b":[],"c":{"d":[{}]}}')),
        '{"a": {}, "b": [], "c": {"d": [{}]}}',
    )


def test_object_key_uniqueness():
    """Test handling of duplicate keys in JSON objects."""
    # When duplicate keys exist, the last one should win
    var obj = loads('{"key": "first", "key": "second"}')
    assert_equal(String(obj), '{"key": "second"}')

    # More complex case with multiple duplicate keys
    var complex_obj = loads(
        """
    {
        "id": 1,
        "name": "original",
        "id": 2,
        "value": 100,
        "name": "updated"
    }
    """
    )

    var serialized = String(complex_obj)
    assert_true('"id": 2' in serialized)
    assert_true('"name": "updated"' in serialized)
    assert_true('"value": 100' in serialized)
    assert_false('"id": 1' in serialized)
    assert_false('"name": "original"' in serialized)


def test_deeply_nested_structures():
    """Test extremely deeply nested JSON structures."""
    # Create a deeply nested array structure
    var deep_array_json = String("[")
    for _ in range(99):
        deep_array_json += "["
    deep_array_json += "42"
    for _ in range(99):
        deep_array_json += "]"
    deep_array_json += "]"

    var deep_array = loads(deep_array_json)
    # Verify it parsed successfully
    assert_true("42.0" in String(deep_array))

    # Create a deeply nested object structure
    var deep_object_json = String("{")
    for i in range(1, 50):
        deep_object_json += '"level' + String(i) + '": {'
    deep_object_json += '"value": 42'
    for _ in range(50):
        deep_object_json += "}"

    var deep_object = loads(deep_object_json)
    # Verify it parsed successfully
    assert_true("42.0" in String(deep_object))


def test_large_structures():
    """Test parsing of large JSON structures."""
    # Generate a large array with many elements
    var large_array_json = String("[")
    for i in range(1000):
        if i > 0:
            large_array_json += ", "
        large_array_json += String(i)
    large_array_json += "]"

    var large_array = loads(large_array_json)
    assert_true("0.0" in String(large_array))
    assert_true("999.0" in String(large_array))

    # Generate a large object with many key-value pairs
    var large_object_json = String("{")
    for i in range(1000):
        if i > 0:
            large_object_json += ", "
        large_object_json += '"key' + String(i) + '": ' + String(i)
    large_object_json += "}"

    var large_object = loads(large_object_json)
    assert_true('"key0": 0.0' in String(large_object))
    assert_true('"key999": 999.0' in String(large_object))


def main():
    test_loads()
    test_primitive_serialization()
    test_array_serialization()
    test_object_serialization()
    test_nested_structures()
    test_invalid_json()
    test_whitespace_handling()
    test_complex_json()
    test_object_and_arrays()
    test_real_world_json_formats()
    test_number_edge_cases()
    test_object_key_uniqueness()
    test_empty_structures()
    test_deeply_nested_structures()
    test_large_structures()
