# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
import json

from max.serve.router.json_utils import parse_json_from_text


def test_json_parsing():
    text = """
    {
        "name": "John",
        "age": 30,
        "nested_object": {
            "name": "John",
            "age": 30
        }
    }
    {
        "name": "Jane",
        "age": 25
    }
    ["foo", "bar"]
    """
    assert parse_json_from_text(text) == [
        {
            "name": "John",
            "age": 30,
            "nested_object": {"name": "John", "age": 30},
        },
        {"name": "Jane", "age": 25},
        ["foo", "bar"],
    ]


def test_large_json_object():
    """Test parsing a very large JSON object."""
    # Construct a large JSON object with 10,000 key-value pairs
    large_dict = {f"key_{i}": i for i in range(10000)}

    # Convert dict to JSON string (to ensure valid JSON formatting)
    json_text = json.dumps(large_dict)

    # The parser expects text, so we can add some whitespace and newlines
    test_text = f"\n{json_text}\nfoobar"
    result = parse_json_from_text(test_text)
    assert len(result) == 1
    assert result[0] == large_dict


def test_no_json_in_text():
    text = "foobar"
    assert parse_json_from_text(text) == []
