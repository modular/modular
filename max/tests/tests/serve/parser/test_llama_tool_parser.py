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

import json

import pytest
from max.interfaces import ParsedToolCall, ParsedToolResponse
from max.serve.parser.llama_tool_parser import LlamaToolParser


def test_single_tool_call_parsing() -> None:
    """Test parsing a single tool call from JSON response."""
    parser = LlamaToolParser()

    response = """
    {
        "name": "get_weather",
        "parameters": {
            "location": "New York",
            "unit": "fahrenheit"
        }
    }
    """

    result = parser.parse_complete(response)

    assert isinstance(result, ParsedToolResponse)
    assert result.content is None
    assert len(result.tool_calls) == 1

    tool_call = result.tool_calls[0]
    assert isinstance(tool_call, ParsedToolCall)
    assert tool_call.id.startswith("call_")
    assert len(tool_call.id) == 21  # "call_" + 16 chars
    assert tool_call.name == "get_weather"
    assert json.loads(tool_call.arguments) == {
        "location": "New York",
        "unit": "fahrenheit",
    }


def test_multiple_tool_calls_parsing() -> None:
    """Test parsing multiple tool calls from JSON response."""
    parser = LlamaToolParser()

    response = """
    {
        "name": "get_weather",
        "parameters": {
            "location": "New York"
        }
    }
    {
        "name": "get_time",
        "parameters": {
            "timezone": "EST"
        }
    }
    """

    result = parser.parse_complete(response)

    assert len(result.tool_calls) == 2

    # Check first tool call
    tool_call1 = result.tool_calls[0]
    assert tool_call1.name == "get_weather"
    assert json.loads(tool_call1.arguments) == {"location": "New York"}

    # Check second tool call
    tool_call2 = result.tool_calls[1]
    assert tool_call2.name == "get_time"
    assert json.loads(tool_call2.arguments) == {"timezone": "EST"}

    # Ensure IDs are different
    assert tool_call1.id != tool_call2.id


def test_empty_response() -> None:
    """Test parsing an empty response."""
    parser = LlamaToolParser()

    response = ""

    result = parser.parse_complete(response)

    assert result.content is None
    assert len(result.tool_calls) == 0


def test_response_with_no_json() -> None:
    """Test parsing a response with no valid JSON."""
    parser = LlamaToolParser()

    response = "This is just plain text with no JSON objects."

    result = parser.parse_complete(response)

    assert result.content is None
    assert len(result.tool_calls) == 0


def test_response_with_text_and_json() -> None:
    """Test parsing a response that contains text and JSON."""
    parser = LlamaToolParser()

    response = """
    I need to get the weather for you.
    {
        "name": "get_weather",
        "parameters": {
            "location": "Boston"
        }
    }
    Let me check that for you.
    """

    result = parser.parse_complete(response)

    assert len(result.tool_calls) == 1
    tool_call = result.tool_calls[0]
    assert tool_call.name == "get_weather"
    assert json.loads(tool_call.arguments) == {"location": "Boston"}


def test_malformed_json_ignored() -> None:
    """Test that malformed JSON is ignored without causing errors."""
    parser = LlamaToolParser()

    response = """
    {
        "name": "get_weather",
        "parameters": {
            "location": "Seattle"
        }
    }
    { this is malformed json }
    {
        "name": "get_time",
        "parameters": {}
    }
    """

    result = parser.parse_complete(response)

    # Should only parse the valid JSON objects
    assert len(result.tool_calls) == 2
    assert result.tool_calls[0].name == "get_weather"
    assert result.tool_calls[1].name == "get_time"


def test_json_missing_name_raises_error() -> None:
    """Test that JSON missing 'name' field raises ValueError."""
    parser = LlamaToolParser()

    response = """
    {
        "parameters": {
            "location": "Miami"
        }
    }
    """

    with pytest.raises(
        ValueError,
        match=r"Both name and parameters not present in parsed JSON response.",
    ):
        parser.parse_complete(response)


def test_json_missing_parameters_raises_error() -> None:
    """Test that JSON missing 'parameters' field raises ValueError."""
    parser = LlamaToolParser()

    response = """
    {
        "name": "get_weather"
    }
    """

    with pytest.raises(
        ValueError,
        match=r"Both name and parameters not present in parsed JSON response.",
    ):
        parser.parse_complete(response)


def test_json_missing_both_fields_raises_error() -> None:
    """Test that JSON missing both 'name' and 'parameters' fields raises ValueError."""
    parser = LlamaToolParser()

    response = """
    {
        "description": "Some description"
    }
    """

    with pytest.raises(
        ValueError,
        match=r"Both name and parameters not present in parsed JSON response.",
    ):
        parser.parse_complete(response)


def test_mixed_valid_and_invalid_json() -> None:
    """Test that valid JSON is parsed while invalid JSON raises error."""
    parser = LlamaToolParser()

    response = """
    {
        "name": "get_weather",
        "parameters": {
            "location": "Portland"
        }
    }
    {
        "description": "Invalid entry missing name and parameters"
    }
    """

    with pytest.raises(
        ValueError,
        match=r"Both name and parameters not present in parsed JSON response.",
    ):
        parser.parse_complete(response)


def test_complex_parameters() -> None:
    """Test parsing tool call with complex nested parameters."""
    parser = LlamaToolParser()

    complex_params = {
        "query": "machine learning",
        "filters": {
            "date_range": {"start": "2023-01-01", "end": "2023-12-31"},
            "categories": ["ai", "tech"],
            "min_score": 0.8,
        },
        "options": {"limit": 10, "sort": "relevance", "include_metadata": True},
    }

    response = f"""
    {{
        "name": "search_articles",
        "parameters": {json.dumps(complex_params)}
    }}
    """

    result = parser.parse_complete(response)

    assert len(result.tool_calls) == 1
    tool_call = result.tool_calls[0]
    assert tool_call.name == "search_articles"
    parsed_args = json.loads(tool_call.arguments)
    assert parsed_args == complex_params


def test_empty_parameters() -> None:
    """Test parsing tool call with empty parameters."""
    parser = LlamaToolParser()

    response = """
    {
        "name": "get_random_fact",
        "parameters": {}
    }
    """

    result = parser.parse_complete(response)

    assert len(result.tool_calls) == 1
    tool_call = result.tool_calls[0]
    assert tool_call.name == "get_random_fact"
    assert json.loads(tool_call.arguments) == {}


def test_unique_tool_call_ids() -> None:
    """Test that each tool call gets a unique ID."""
    parser = LlamaToolParser()

    # Parse the same response multiple times
    response = """
    {
        "name": "test_function",
        "parameters": {"param": "value"}
    }
    """

    ids = set()
    for _ in range(10):
        result = parser.parse_complete(response)
        tool_call_id = result.tool_calls[0].id
        ids.add(tool_call_id)

    # All IDs should be unique
    assert len(ids) == 10

    # All IDs should follow the expected format
    for tool_id in ids:
        assert tool_id.startswith("call_")
        assert len(tool_id) == 21  # "call_" + 16 chars


def test_tool_call_id_format() -> None:
    """Test that tool call IDs have the correct format."""
    parser = LlamaToolParser()

    response = """
    {
        "name": "test_function",
        "parameters": {"param": "value"}
    }
    """

    result = parser.parse_complete(response)
    tool_call_id = result.tool_calls[0].id

    # Should start with "call_"
    assert tool_call_id.startswith("call_")

    # Should be exactly 21 characters long
    assert len(tool_call_id) == 21

    # The UUID part should be 16 characters (no hyphens)
    uuid_part = tool_call_id[5:]  # Remove "call_" prefix
    assert len(uuid_part) == 16
    assert "-" not in uuid_part


def test_response_structure() -> None:
    """Test that the response structure matches expected ParsedToolResponse format."""
    parser = LlamaToolParser()

    response = """
    {
        "name": "calculate",
        "parameters": {"expression": "2 + 2"}
    }
    """

    result = parser.parse_complete(response)

    # Should return a ParsedToolResponse object
    assert isinstance(result, ParsedToolResponse)
    assert result.content is None
    assert len(result.tool_calls) == 1

    tool_call = result.tool_calls[0]
    assert isinstance(tool_call, ParsedToolCall)
    assert tool_call.name == "calculate"
    assert tool_call.id.startswith("call_")


def test_reset_clears_buffer() -> None:
    """Test that reset() clears the internal buffer."""
    parser = LlamaToolParser()

    # Simulate accumulating some data
    parser._buffer = "some accumulated data"

    parser.reset()

    assert parser._buffer == ""


def test_parse_delta_accumulates() -> None:
    """Test that parse_delta accumulates tokens in buffer."""
    parser = LlamaToolParser()

    # Note: parse_delta is a stub that returns None
    # but should accumulate tokens
    result1 = parser.parse_delta('{"name":')
    result2 = parser.parse_delta('"test"}')

    assert result1 is None
    assert result2 is None
    assert parser._buffer == '{"name":"test"}'
