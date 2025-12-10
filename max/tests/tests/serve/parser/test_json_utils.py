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
import json

from max.serve.parser import parse_json_from_text


def test_json_parsing() -> None:
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


def test_large_json_object() -> None:
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


def test_no_json_in_text() -> None:
    text = "foobar"
    assert parse_json_from_text(text) == []
