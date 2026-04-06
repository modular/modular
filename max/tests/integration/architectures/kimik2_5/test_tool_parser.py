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
from max.pipelines.architectures.kimik2_5.tool_parser import KimiToolParser


def test_single_tool_call_parsing() -> None:
    """Test parsing a single tool call with Kimi structural tags."""
    parser = KimiToolParser()

    response = """<|tool_calls_section_begin|>
<|tool_call_begin|>functions.get_weather:0<|tool_call_argument_begin|>
{"location": "New York", "unit": "fahrenheit"}
<|tool_call_end|>
<|tool_calls_section_end|>"""

    result = parser.parse_complete(response)

    assert isinstance(result, ParsedToolResponse)
    assert result.content is None
    assert len(result.tool_calls) == 1

    tool_call = result.tool_calls[0]
    assert isinstance(tool_call, ParsedToolCall)
    assert tool_call.id.startswith("call_")
    assert tool_call.name == "get_weather"
    assert json.loads(tool_call.arguments) == {
        "location": "New York",
        "unit": "fahrenheit",
    }


def test_multiple_tool_calls_parsing() -> None:
    """Test parsing multiple tool calls from Kimi response."""
    parser = KimiToolParser()

    response = """<|tool_calls_section_begin|>
<|tool_call_begin|>functions.get_weather:0<|tool_call_argument_begin|>
{"location": "New York"}
<|tool_call_end|>
<|tool_call_begin|>functions.get_time:1<|tool_call_argument_begin|>
{"timezone": "EST"}
<|tool_call_end|>
<|tool_calls_section_end|>"""

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


def test_response_without_tool_calls() -> None:
    """Test parsing a response without tool calls section."""
    parser = KimiToolParser()

    response = "This is just a regular response with no tool calls."

    result = parser.parse_complete(response)

    assert result.content == response
    assert len(result.tool_calls) == 0


def test_empty_response() -> None:
    """Test parsing an empty response."""
    parser = KimiToolParser()

    response = ""

    result = parser.parse_complete(response)

    assert result.content == ""
    assert len(result.tool_calls) == 0


def test_content_before_tool_calls() -> None:
    """Test parsing response with content before tool calls section."""
    parser = KimiToolParser()

    response = """I'll help you check the weather.

<|tool_calls_section_begin|>
<|tool_call_begin|>functions.get_weather:0<|tool_call_argument_begin|>
{"location": "Boston"}
<|tool_call_end|>
<|tool_calls_section_end|>"""

    result = parser.parse_complete(response)

    assert result.content == "I'll help you check the weather."
    assert len(result.tool_calls) == 1
    assert result.tool_calls[0].name == "get_weather"


def test_function_id_without_prefix() -> None:
    """Test parsing function ID without 'functions.' prefix (fallback format)."""
    parser = KimiToolParser()

    response = """<|tool_calls_section_begin|>
<|tool_call_begin|>search:2<|tool_call_argument_begin|>
{"query": "python tutorials"}
<|tool_call_end|>
<|tool_calls_section_end|>"""

    result = parser.parse_complete(response)

    assert len(result.tool_calls) == 1
    tool_call = result.tool_calls[0]
    assert tool_call.name == "search"
    assert "_2" in tool_call.id  # Index should be in the ID


def test_function_id_without_index() -> None:
    """Test parsing function ID without index suffix."""
    parser = KimiToolParser()

    response = """<|tool_calls_section_begin|>
<|tool_call_begin|>functions.calculate<|tool_call_argument_begin|>
{"expression": "2 + 2"}
<|tool_call_end|>
<|tool_calls_section_end|>"""

    result = parser.parse_complete(response)

    assert len(result.tool_calls) == 1
    tool_call = result.tool_calls[0]
    assert tool_call.name == "calculate"
    assert tool_call.id.startswith("call_")


def test_plain_function_name() -> None:
    """Test parsing plain function name without prefix or index."""
    parser = KimiToolParser()

    response = """<|tool_calls_section_begin|>
<|tool_call_begin|>get_random_fact<|tool_call_argument_begin|>
{}
<|tool_call_end|>
<|tool_calls_section_end|>"""

    result = parser.parse_complete(response)

    assert len(result.tool_calls) == 1
    tool_call = result.tool_calls[0]
    assert tool_call.name == "get_random_fact"


def test_complex_parameters() -> None:
    """Test parsing tool call with complex nested parameters."""
    parser = KimiToolParser()

    complex_params = {
        "query": "machine learning",
        "filters": {
            "date_range": {"start": "2023-01-01", "end": "2023-12-31"},
            "categories": ["ai", "tech"],
            "min_score": 0.8,
        },
        "options": {"limit": 10, "sort": "relevance", "include_metadata": True},
    }

    response = f"""<|tool_calls_section_begin|>
<|tool_call_begin|>functions.search_articles:0<|tool_call_argument_begin|>
{json.dumps(complex_params)}
<|tool_call_end|>
<|tool_calls_section_end|>"""

    result = parser.parse_complete(response)

    assert len(result.tool_calls) == 1
    tool_call = result.tool_calls[0]
    assert tool_call.name == "search_articles"
    parsed_args = json.loads(tool_call.arguments)
    assert parsed_args == complex_params


def test_empty_parameters() -> None:
    """Test parsing tool call with empty parameters."""
    parser = KimiToolParser()

    response = """<|tool_calls_section_begin|>
<|tool_call_begin|>functions.get_random_fact:0<|tool_call_argument_begin|>
{}
<|tool_call_end|>
<|tool_calls_section_end|>"""

    result = parser.parse_complete(response)

    assert len(result.tool_calls) == 1
    tool_call = result.tool_calls[0]
    assert tool_call.name == "get_random_fact"
    assert json.loads(tool_call.arguments) == {}


def test_tool_calls_section_without_end_tag() -> None:
    """Test parsing when end tag is missing (should still parse)."""
    parser = KimiToolParser()

    response = """<|tool_calls_section_begin|>
<|tool_call_begin|>functions.test:0<|tool_call_argument_begin|>
{"key": "value"}
<|tool_call_end|>"""

    result = parser.parse_complete(response)

    assert len(result.tool_calls) == 1
    assert result.tool_calls[0].name == "test"


def test_empty_tool_calls_section_raises_error() -> None:
    """Test that empty tool calls section raises ValueError."""
    parser = KimiToolParser()

    response = """<|tool_calls_section_begin|>
<|tool_calls_section_end|>"""

    with pytest.raises(ValueError, match=r"no valid tool calls parsed"):
        parser.parse_complete(response)


def test_unique_tool_call_ids() -> None:
    """Test that each tool call gets a unique ID."""
    parser = KimiToolParser()

    response = """<|tool_calls_section_begin|>
<|tool_call_begin|>functions.test:0<|tool_call_argument_begin|>
{"param": "value"}
<|tool_call_end|>
<|tool_calls_section_end|>"""

    ids = set()
    for _ in range(10):
        result = parser.parse_complete(response)
        tool_call_id = result.tool_calls[0].id
        ids.add(tool_call_id)

    # All IDs should be unique
    assert len(ids) == 10


def test_tool_call_id_format() -> None:
    """Test that tool call IDs have the correct format."""
    parser = KimiToolParser()

    response = """<|tool_calls_section_begin|>
<|tool_call_begin|>functions.test:5<|tool_call_argument_begin|>
{"param": "value"}
<|tool_call_end|>
<|tool_calls_section_end|>"""

    result = parser.parse_complete(response)
    tool_call_id = result.tool_calls[0].id

    # Should start with "call_"
    assert tool_call_id.startswith("call_")

    # Should contain the index
    assert "_5" in tool_call_id


def test_response_structure() -> None:
    """Test that the response structure matches expected ParsedToolResponse format."""
    parser = KimiToolParser()

    response = """<|tool_calls_section_begin|>
<|tool_call_begin|>functions.calculate:0<|tool_call_argument_begin|>
{"expression": "2 + 2"}
<|tool_call_end|>
<|tool_calls_section_end|>"""

    result = parser.parse_complete(response)

    # Should return a ParsedToolResponse object
    assert isinstance(result, ParsedToolResponse)
    assert result.content is None
    assert len(result.tool_calls) == 1

    tool_call = result.tool_calls[0]
    assert isinstance(tool_call, ParsedToolCall)
    assert tool_call.name == "calculate"
    assert tool_call.id.startswith("call_")


def test_whitespace_handling() -> None:
    """Test that whitespace in arguments is handled correctly."""
    parser = KimiToolParser()

    response = """<|tool_calls_section_begin|>
<|tool_call_begin|>functions.test:0<|tool_call_argument_begin|>

    {
        "key": "value with spaces",
        "nested": {
            "inner": "data"
        }
    }

<|tool_call_end|>
<|tool_calls_section_end|>"""

    result = parser.parse_complete(response)

    assert len(result.tool_calls) == 1
    args = json.loads(result.tool_calls[0].arguments)
    assert args == {"key": "value with spaces", "nested": {"inner": "data"}}


def test_reset_clears_buffer() -> None:
    """Test that reset() clears the internal buffer."""
    parser = KimiToolParser()

    # Simulate accumulating some data
    parser._buffer = "some accumulated data"

    parser.reset()

    assert parser._buffer == ""


def test_parse_delta_accumulates() -> None:
    """Test that parse_delta accumulates tokens in buffer."""
    parser = KimiToolParser()

    # Note: parse_delta is a stub that returns None
    # but should accumulate tokens
    result1 = parser.parse_delta("<|tool_calls")
    result2 = parser.parse_delta("_section_begin|>")

    assert result1 is None
    assert result2 is None
    assert parser._buffer == "<|tool_calls_section_begin|>"


def test_multiple_tool_calls_same_function() -> None:
    """Test parsing multiple calls to the same function."""
    parser = KimiToolParser()

    response = """<|tool_calls_section_begin|>
<|tool_call_begin|>functions.search:0<|tool_call_argument_begin|>
{"query": "first query"}
<|tool_call_end|>
<|tool_call_begin|>functions.search:1<|tool_call_argument_begin|>
{"query": "second query"}
<|tool_call_end|>
<|tool_call_begin|>functions.search:2<|tool_call_argument_begin|>
{"query": "third query"}
<|tool_call_end|>
<|tool_calls_section_end|>"""

    result = parser.parse_complete(response)

    assert len(result.tool_calls) == 3

    # All should have the same function name
    for tc in result.tool_calls:
        assert tc.name == "search"

    # But different IDs
    ids = [tc.id for tc in result.tool_calls]
    assert len(set(ids)) == 3

    # And different arguments
    queries = [json.loads(tc.arguments)["query"] for tc in result.tool_calls]
    assert queries == ["first query", "second query", "third query"]


def test_special_characters_in_arguments() -> None:
    """Test handling of special characters in tool arguments."""
    parser = KimiToolParser()

    special_args = {
        "code": 'print("Hello, World!")',
        "regex": r"\d+\.\d+",
        "unicode": "Hello \u4e16\u754c",
        "newlines": "line1\nline2\nline3",
    }

    response = f"""<|tool_calls_section_begin|>
<|tool_call_begin|>functions.execute:0<|tool_call_argument_begin|>
{json.dumps(special_args)}
<|tool_call_end|>
<|tool_calls_section_end|>"""

    result = parser.parse_complete(response)

    assert len(result.tool_calls) == 1
    parsed_args = json.loads(result.tool_calls[0].arguments)
    assert parsed_args == special_args
