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
from max.pipelines.architectures.gemma4.tool_parser import Gemma4ToolParser


def test_single_tool_call_parsing() -> None:
    """Test parsing a single tool call."""
    parser = Gemma4ToolParser()

    response = (
        '<|tool_call>call:get_weather{location:<|"|>Paris<|"|>}<tool_call|>'
    )

    result = parser.parse_complete(response)

    assert isinstance(result, ParsedToolResponse)
    assert result.content is None
    assert len(result.tool_calls) == 1

    tool_call = result.tool_calls[0]
    assert isinstance(tool_call, ParsedToolCall)
    assert tool_call.name == "get_weather"
    assert json.loads(tool_call.arguments) == {"location": "Paris"}


def test_multiple_tool_calls_parsing() -> None:
    """Test parsing multiple tool calls."""
    parser = Gemma4ToolParser()

    response = '<|tool_call>call:get_weather{location:<|"|>New York<|"|>}<tool_call|><|tool_call>call:get_time{timezone:<|"|>Asia/Tokyo<|"|>}<tool_call|>'

    result = parser.parse_complete(response)

    assert len(result.tool_calls) == 2

    tool_call1 = result.tool_calls[0]
    assert tool_call1.name == "get_weather"
    assert json.loads(tool_call1.arguments) == {"location": "New York"}

    tool_call2 = result.tool_calls[1]
    assert tool_call2.name == "get_time"
    assert json.loads(tool_call2.arguments) == {"timezone": "Asia/Tokyo"}

    assert tool_call1.id != tool_call2.id


def test_response_without_tool_calls() -> None:
    """Test parsing a response without tool calls."""
    parser = Gemma4ToolParser()

    response = "This is just a regular response with no tool calls."

    result = parser.parse_complete(response)

    assert result.content == response
    assert len(result.tool_calls) == 0


def test_empty_response() -> None:
    """Test parsing an empty response."""
    parser = Gemma4ToolParser()

    response = ""

    result = parser.parse_complete(response)

    assert result.content == ""
    assert len(result.tool_calls) == 0


def test_multiple_parameters() -> None:
    """Test parsing tool call with multiple parameters."""
    parser = Gemma4ToolParser()

    response = '<|tool_call>call:get_weather{location:<|"|>Boston<|"|>,unit:<|"|>fahrenheit<|"|>}<tool_call|>'

    result = parser.parse_complete(response)

    assert len(result.tool_calls) == 1
    tool_call = result.tool_calls[0]
    assert tool_call.name == "get_weather"
    assert json.loads(tool_call.arguments) == {
        "location": "Boston",
        "unit": "fahrenheit",
    }


def test_complex_nested_parameters() -> None:
    """Test parsing tool call with complex nested parameters."""
    parser = Gemma4ToolParser()

    response = '<|tool_call>call:search_articles{filters:{categories:[<|"|>AI<|"|>],date_range:{end:<|"|>2023-12-31<|"|>,start:<|"|>2023-01-01<|"|>}},options:{limit:10,sort:<|"|>relevance<|"|>},query:<|"|>machine learning<|"|>}<tool_call|>'

    result = parser.parse_complete(response)

    assert len(result.tool_calls) == 1
    tool_call = result.tool_calls[0]
    assert tool_call.name == "search_articles"
    parsed_args = json.loads(tool_call.arguments)
    assert parsed_args == {
        "query": "machine learning",
        "filters": {
            "date_range": {"start": "2023-01-01", "end": "2023-12-31"},
            "categories": ["AI"],
        },
        "options": {"limit": 10, "sort": "relevance"},
    }


def test_empty_parameters() -> None:
    """Test parsing tool call with empty parameters."""
    parser = Gemma4ToolParser()

    response = "<|tool_call>call:get_random_fact{}<tool_call|>"

    result = parser.parse_complete(response)

    assert len(result.tool_calls) == 1
    tool_call = result.tool_calls[0]
    assert tool_call.name == "get_random_fact"
    assert json.loads(tool_call.arguments) == {}


def test_multiple_calls_same_function() -> None:
    """Test parsing multiple calls to the same function."""
    parser = Gemma4ToolParser()

    response = '<|tool_call>call:get_weather{location:<|"|>London<|"|>}<tool_call|><|tool_call>call:get_weather{location:<|"|>Paris<|"|>}<tool_call|><|tool_call>call:get_weather{location:<|"|>Berlin<|"|>}<tool_call|>'

    result = parser.parse_complete(response)

    assert len(result.tool_calls) == 3

    for tc in result.tool_calls:
        assert tc.name == "get_weather"

    ids = [tc.id for tc in result.tool_calls]
    assert len(set(ids)) == 3

    locations = [
        json.loads(tc.arguments)["location"] for tc in result.tool_calls
    ]
    assert locations == ["London", "Paris", "Berlin"]


def test_special_characters_in_arguments() -> None:
    """Test handling of special characters in tool arguments."""
    parser = Gemma4ToolParser()

    response = '<|tool_call>call:execute_code{code:<|"|>print("Hello, World!")<|"|>,language:<|"|>python<|"|>}<tool_call|>'

    result = parser.parse_complete(response)

    assert len(result.tool_calls) == 1
    parsed_args = json.loads(result.tool_calls[0].arguments)
    assert parsed_args == {
        "code": 'print("Hello, World!")',
        "language": "python",
    }


def test_array_parameters() -> None:
    """Test parsing tool call with array parameters."""
    parser = Gemma4ToolParser()

    response = "<|tool_call>call:calculate_sum{numbers:[1,2,3,4,5]}<tool_call|>"

    result = parser.parse_complete(response)

    assert len(result.tool_calls) == 1
    tool_call = result.tool_calls[0]
    assert tool_call.name == "calculate_sum"
    assert json.loads(tool_call.arguments) == {"numbers": [1, 2, 3, 4, 5]}


def test_boolean_parameters() -> None:
    """Test parsing tool call with boolean parameters."""
    parser = Gemma4ToolParser()

    response = '<|tool_call>call:send_notification{message:<|"|>Notification message<|"|>,priority:true}<tool_call|>'

    result = parser.parse_complete(response)

    assert len(result.tool_calls) == 1
    tool_call = result.tool_calls[0]
    assert tool_call.name == "send_notification"
    parsed_args = json.loads(tool_call.arguments)
    assert parsed_args == {"message": "Notification message", "priority": True}


def test_tool_call_without_end_tag() -> None:
    """Test parsing when end tag is missing."""
    parser = Gemma4ToolParser()

    response = '<|tool_call>call:test{key:<|"|>value<|"|>}'

    with pytest.raises(ValueError, match=r"no valid tool calls parsed"):
        parser.parse_complete(response)


def test_empty_tool_calls_section_raises_error() -> None:
    """Test that tool call start without actual calls raises ValueError."""
    parser = Gemma4ToolParser()

    response = "<|tool_call><tool_call|>"

    with pytest.raises(ValueError, match=r"no valid tool calls parsed"):
        parser.parse_complete(response)


def test_unique_tool_call_ids() -> None:
    """Test that each tool call gets a unique ID."""
    parser = Gemma4ToolParser()

    response = '<|tool_call>call:test{param:<|"|>value<|"|>}<tool_call|>'

    ids = set()
    for _ in range(10):
        result = parser.parse_complete(response)
        tool_call_id = result.tool_calls[0].id
        ids.add(tool_call_id)

    assert len(ids) == 10


def test_tool_call_id_format() -> None:
    """Test that tool call IDs have the correct format."""
    parser = Gemma4ToolParser()

    response = '<|tool_call>call:test{param:<|"|>value<|"|>}<tool_call|>'

    result = parser.parse_complete(response)
    tool_call_id = result.tool_calls[0].id

    assert isinstance(tool_call_id, str)
    assert len(tool_call_id) == 8


def test_response_structure() -> None:
    """Test that the response structure matches expected format."""
    parser = Gemma4ToolParser()

    response = (
        '<|tool_call>call:calculate{expression:<|"|>2 + 2<|"|>}<tool_call|>'
    )

    result = parser.parse_complete(response)

    assert isinstance(result, ParsedToolResponse)
    assert result.content is None
    assert len(result.tool_calls) == 1

    tool_call = result.tool_calls[0]
    assert isinstance(tool_call, ParsedToolCall)
    assert tool_call.name == "calculate"
    assert isinstance(tool_call.id, str)


def test_reset_clears_buffer() -> None:
    """Test that reset() clears the internal buffer."""
    parser = Gemma4ToolParser()

    parser._buffer = "some accumulated data"

    parser.reset()

    assert parser._buffer == ""


def test_parse_delta_accumulates() -> None:
    """Test that parse_delta accumulates tokens in buffer."""
    parser = Gemma4ToolParser()

    result1 = parser.parse_delta("<|tool_call>")
    result2 = parser.parse_delta("call:test{")

    assert result1 is None
    assert result2 is None
    assert parser._buffer == "<|tool_call>call:test{"


def test_number_parameters() -> None:
    """Test parsing tool call with integer and float parameters."""
    parser = Gemma4ToolParser()

    response = (
        "<|tool_call>call:set_temperature{value:22,precision:0.5}<tool_call|>"
    )

    result = parser.parse_complete(response)

    assert len(result.tool_calls) == 1
    parsed_args = json.loads(result.tool_calls[0].arguments)
    assert parsed_args == {"value": 22, "precision": 0.5}


def test_false_boolean_parameter() -> None:
    """Test parsing tool call with false boolean value."""
    parser = Gemma4ToolParser()

    response = "<|tool_call>call:configure{enabled:false}<tool_call|>"

    result = parser.parse_complete(response)

    assert len(result.tool_calls) == 1
    parsed_args = json.loads(result.tool_calls[0].arguments)
    assert parsed_args == {"enabled": False}


def test_nested_arrays() -> None:
    """Test parsing tool call with nested arrays."""
    parser = Gemma4ToolParser()

    response = (
        "<|tool_call>call:process_matrix{matrix:[[1,2,3],[4,5,6]]}<tool_call|>"
    )

    result = parser.parse_complete(response)

    assert len(result.tool_calls) == 1
    parsed_args = json.loads(result.tool_calls[0].arguments)
    assert parsed_args == {"matrix": [[1, 2, 3], [4, 5, 6]]}


def test_function_name_with_underscores() -> None:
    """Test parsing function names with underscores."""
    parser = Gemma4ToolParser()

    response = "<|tool_call>call:get_current_time{}<tool_call|>"

    result = parser.parse_complete(response)

    assert len(result.tool_calls) == 1
    assert result.tool_calls[0].name == "get_current_time"


def test_function_name_with_dots() -> None:
    """Test parsing function names with dots."""
    parser = Gemma4ToolParser()

    response = "<|tool_call>call:api.v2.search{}<tool_call|>"

    result = parser.parse_complete(response)

    assert len(result.tool_calls) == 1
    assert result.tool_calls[0].name == "api.v2.search"


def test_function_name_with_hyphens() -> None:
    """Test parsing function names with hyphens."""
    parser = Gemma4ToolParser()

    response = "<|tool_call>call:get-user-info{}<tool_call|>"

    result = parser.parse_complete(response)

    assert len(result.tool_calls) == 1
    assert result.tool_calls[0].name == "get-user-info"
