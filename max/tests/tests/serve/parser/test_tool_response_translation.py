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

"""Tests for translating ParsedToolResponse to OpenAI chat completion choices."""

import json
from unittest.mock import MagicMock

from max.interfaces import ParsedToolCall, ParsedToolResponse
from max.serve.router.openai_routes import OpenAIChatResponseGenerator
from max.serve.schemas.openai import (
    ChatCompletionLogprobs,
    ChatCompletionMessageToolCall,
    ChatCompletionMessageToolCallFunction,
    ChatCompletionResponseChoice,
    ChatCompletionResponseMessage,
)


def create_response_generator() -> OpenAIChatResponseGenerator:
    """Create a minimal OpenAIChatResponseGenerator for testing."""
    mock_pipeline = MagicMock()
    return OpenAIChatResponseGenerator(
        pipeline=mock_pipeline,
    )


def test_single_tool_call_translation() -> None:
    """Test translating a single tool call."""
    generator = create_response_generator()

    parsed = ParsedToolResponse(
        content=None,
        tool_calls=[
            ParsedToolCall(
                id="call_abc123",
                name="get_weather",
                arguments='{"location": "New York", "unit": "fahrenheit"}',
            )
        ],
    )

    result = generator._tool_response_to_choices(parsed)

    assert len(result) == 1
    choice = result[0]
    assert isinstance(choice, ChatCompletionResponseChoice)
    assert choice.index == 0
    assert choice.finish_reason == "tool_calls"
    assert isinstance(choice.logprobs, ChatCompletionLogprobs)
    assert choice.logprobs.content == []
    assert choice.logprobs.refusal == []

    message = choice.message
    assert isinstance(message, ChatCompletionResponseMessage)
    assert message.role == "assistant"
    assert message.content == ""
    assert message.function_call is None
    assert message.refusal == ""

    tool_calls = message.tool_calls
    assert tool_calls is not None
    assert len(tool_calls) == 1

    tool_call = tool_calls[0]
    assert isinstance(tool_call, ChatCompletionMessageToolCall)
    assert tool_call.id == "call_abc123"
    assert tool_call.type == "function"

    function = tool_call.function
    assert isinstance(function, ChatCompletionMessageToolCallFunction)
    assert function.name == "get_weather"
    assert json.loads(function.arguments) == {
        "location": "New York",
        "unit": "fahrenheit",
    }


def test_multiple_tool_calls_translation() -> None:
    """Test translating multiple tool calls."""
    generator = create_response_generator()

    parsed = ParsedToolResponse(
        content=None,
        tool_calls=[
            ParsedToolCall(
                id="call_001",
                name="get_weather",
                arguments='{"location": "New York"}',
            ),
            ParsedToolCall(
                id="call_002",
                name="get_time",
                arguments='{"timezone": "EST"}',
            ),
        ],
    )

    result = generator._tool_response_to_choices(parsed)

    assert len(result) == 1
    choice = result[0]
    tool_calls = choice.message.tool_calls
    assert tool_calls is not None
    assert len(tool_calls) == 2

    tool_call1 = tool_calls[0]
    assert isinstance(tool_call1, ChatCompletionMessageToolCall)
    assert tool_call1.id == "call_001"
    assert tool_call1.function.name == "get_weather"
    assert json.loads(tool_call1.function.arguments) == {"location": "New York"}

    tool_call2 = tool_calls[1]
    assert isinstance(tool_call2, ChatCompletionMessageToolCall)
    assert tool_call2.id == "call_002"
    assert tool_call2.function.name == "get_time"
    assert json.loads(tool_call2.function.arguments) == {"timezone": "EST"}


def test_tool_call_with_content_translation() -> None:
    """Test translating tool call with preceding content."""
    generator = create_response_generator()

    parsed = ParsedToolResponse(
        content="I'll help you check the weather.",
        tool_calls=[
            ParsedToolCall(
                id="call_weather",
                name="get_weather",
                arguments='{"location": "Boston"}',
            )
        ],
    )

    result = generator._tool_response_to_choices(parsed)

    assert len(result) == 1
    choice = result[0]
    assert choice.message.content == "I'll help you check the weather."
    assert choice.finish_reason == "tool_calls"

    tool_calls = choice.message.tool_calls
    assert tool_calls is not None
    assert len(tool_calls) == 1
    assert isinstance(tool_calls[0], ChatCompletionMessageToolCall)
    assert tool_calls[0].function.name == "get_weather"


def test_empty_tool_calls_translation() -> None:
    """Test translating response with no tool calls."""
    generator = create_response_generator()

    parsed = ParsedToolResponse(
        content=None,
        tool_calls=[],
    )

    result = generator._tool_response_to_choices(parsed)

    assert len(result) == 1
    choice = result[0]
    assert choice.message.tool_calls is None
    assert choice.message.content == ""


def test_complex_parameters_translation() -> None:
    """Test translating tool call with complex nested parameters."""
    generator = create_response_generator()

    complex_params = {
        "query": "machine learning",
        "filters": {
            "date_range": {"start": "2023-01-01", "end": "2023-12-31"},
            "categories": ["ai", "tech"],
            "min_score": 0.8,
        },
        "options": {"limit": 10, "sort": "relevance", "include_metadata": True},
    }

    parsed = ParsedToolResponse(
        content=None,
        tool_calls=[
            ParsedToolCall(
                id="call_search",
                name="search_articles",
                arguments=json.dumps(complex_params),
            )
        ],
    )

    result = generator._tool_response_to_choices(parsed)

    assert len(result) == 1
    tool_calls = result[0].message.tool_calls
    assert tool_calls is not None
    assert len(tool_calls) == 1

    tool_call = tool_calls[0]
    assert isinstance(tool_call, ChatCompletionMessageToolCall)
    assert tool_call.function.name == "search_articles"
    parsed_args = json.loads(tool_call.function.arguments)
    assert parsed_args == complex_params


def test_custom_logprobs_translation() -> None:
    """Test that custom logprobs are passed through correctly."""
    generator = create_response_generator()

    parsed = ParsedToolResponse(
        content=None,
        tool_calls=[
            ParsedToolCall(
                id="call_test",
                name="test_function",
                arguments="{}",
            )
        ],
    )

    custom_logprobs = ChatCompletionLogprobs(
        content=[],
        refusal=[],
    )

    result = generator._tool_response_to_choices(
        parsed, logprobs=custom_logprobs
    )

    assert len(result) == 1
    assert result[0].logprobs is custom_logprobs


def test_tool_call_type_is_function() -> None:
    """Test that tool call type is always 'function'."""
    generator = create_response_generator()

    parsed = ParsedToolResponse(
        content=None,
        tool_calls=[
            ParsedToolCall(
                id="call_1",
                name="func1",
                arguments="{}",
            ),
            ParsedToolCall(
                id="call_2",
                name="func2",
                arguments="{}",
            ),
        ],
    )

    result = generator._tool_response_to_choices(parsed)

    tool_calls = result[0].message.tool_calls
    assert tool_calls is not None
    for tc in tool_calls:
        assert tc.type == "function"


def test_response_structure_matches_openai_spec() -> None:
    """Test that the response structure matches OpenAI spec."""
    generator = create_response_generator()

    parsed = ParsedToolResponse(
        content=None,
        tool_calls=[
            ParsedToolCall(
                id="call_verify",
                name="verify_structure",
                arguments='{"test": true}',
            )
        ],
    )

    result = generator._tool_response_to_choices(parsed)

    assert isinstance(result, list)
    assert len(result) == 1

    choice = result[0]
    assert isinstance(choice, ChatCompletionResponseChoice)
    assert choice.index == 0
    assert choice.finish_reason == "tool_calls"
    assert isinstance(choice.logprobs, ChatCompletionLogprobs)

    message = choice.message
    assert isinstance(message, ChatCompletionResponseMessage)
    assert message.role == "assistant"
    assert message.function_call is None
    assert message.refusal == ""
    assert message.tool_calls is not None
    assert len(message.tool_calls) == 1

    tool_call = message.tool_calls[0]
    assert isinstance(tool_call, ChatCompletionMessageToolCall)
    assert isinstance(tool_call.function, ChatCompletionMessageToolCallFunction)
