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
import uuid
from unittest.mock import patch

import pytest
from llguidance import LLMatcher, LLTokenizer
from llguidance._tokenizer import TokenizerWrapper
from max.interfaces import (
    ParsedToolCall,
    ParsedToolCallDelta,
    ParsedToolResponse,
)
from max.pipelines.architectures.kimik2_5.tool_parser import KimiToolParser
from max.pipelines.lib.tool_parsing import StreamingToolCallState


class _MinimalTokenizer:
    """Minimal byte tokenizer for grammar compilation validation tests.

    Maps each byte value to a token ID, providing a 256-token vocabulary
    sufficient for testing grammar compilation without loading a real model.
    """

    eos_token_id: int = 0
    bos_token_id: int | None = None
    tokens: list[bytes] = [bytes([i]) for i in range(256)]

    def __call__(self, s: bytes | str) -> list[int]:
        if isinstance(s, str):
            s = s.encode("utf-8")
        return list(s)


@pytest.fixture(scope="module")
def ll_tokenizer() -> LLTokenizer:
    """Create a minimal LLTokenizer for grammar validation tests."""
    wrapper = TokenizerWrapper(_MinimalTokenizer())
    return LLTokenizer(wrapper, n_vocab=256)


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
    assert tool_call.id.startswith("get_weather:")
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
    assert tool_call.id.startswith("search:")


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
    assert tool_call.id.startswith("calculate:")


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

    assert tool_call_id.startswith("test:")


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
    assert tool_call.id.startswith("calculate:")


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
    """Test that reset() clears the internal buffer and streaming state."""

    parser = KimiToolParser()

    # Simulate accumulating some data
    parser._buffer = "some accumulated data"
    parser._state.sent_content_idx = 10
    parser._state.tool_calls.append(StreamingToolCallState())

    parser.reset()

    assert parser._buffer == ""
    assert parser._state.sent_content_idx == 0
    assert len(parser._state.tool_calls) == 0


def test_parse_delta_accumulates() -> None:
    """Test that parse_delta accumulates tokens in buffer."""
    parser = KimiToolParser()

    # parse_delta should accumulate tokens; before any section marker lands,
    # result is None
    result1 = parser.parse_delta("<|tool_calls")
    assert result1 is None

    # Once the section-begin marker is complete, returns [] (not None) so the
    # streaming path knows to suppress structural tokens even with no deltas yet
    result2 = parser.parse_delta("_section_begin|>")
    assert result2 == []
    assert parser._buffer == "<|tool_calls_section_begin|>"


def test_parse_delta_returns_empty_list_inside_tool_section() -> None:
    """Test that parse_delta returns [] (not None) inside the tool-calls section.

    [] signals the caller to suppress raw structural tokens from being emitted
    as content, even when there are no tool-call deltas ready to stream yet.
    """
    parser = KimiToolParser()

    # Tokens that don't start/complete a section marker and have no sendable
    # content return None (more context needed before anything can be emitted)
    result_pre = parser.parse_delta("<|tool_calls")
    assert result_pre is None

    # Once the section-begin marker completes, returns [] even with no deltas
    result_in_section = parser.parse_delta("_section_begin|>")
    assert result_in_section == []

    # Structural tokens mid-section also return [] while parsing
    result_mid = parser.parse_delta("<|tool_call_begin|>")
    assert result_mid == []


def test_parse_delta_single_tool_call_streaming() -> None:
    """Test streaming a single tool call token by token."""
    fixed_uuid = uuid.UUID("12345678-1234-5678-9abc-def012345678")
    with patch(
        "max.pipelines.architectures.kimik2_5.tool_parser.uuid.uuid4",
        return_value=fixed_uuid,
    ):
        parser = KimiToolParser()

        # Simulate streaming a complete tool call
        chunks = [
            "<|tool_calls_section_begin|>",
            "<|tool_call_begin|>",
            "functions.get_weather:0",
            "<|tool_call_argument_begin|>",
            '{"loc',
            'ation": "',
            'New York"}',
            "<|tool_call_end|>",
            "<|tool_calls_section_end|>",
        ]

        all_deltas: list[ParsedToolCallDelta] = []
        for chunk in chunks:
            result = parser.parse_delta(chunk)
            if result:
                all_deltas.extend(result)

        assert all_deltas == [
            ParsedToolCallDelta(
                index=0,
                id="get_weather:12345678",
                name="get_weather",
            ),
            ParsedToolCallDelta(index=0, arguments='{"loc'),
            ParsedToolCallDelta(index=0, arguments='ation": "'),
            ParsedToolCallDelta(index=0, arguments='New York"}'),
        ]


def test_parse_delta_multiple_tool_calls_streaming() -> None:
    """Test streaming multiple tool calls."""

    uuid_first = uuid.UUID("11111111-1111-1111-1111-111111111111")
    uuid_second = uuid.UUID("22222222-2222-2222-2222-222222222222")
    with patch(
        "max.pipelines.architectures.kimik2_5.tool_parser.uuid.uuid4",
        side_effect=[uuid_first, uuid_second],
    ):
        parser = KimiToolParser()

        # Complete response with two tool calls
        response = """<|tool_calls_section_begin|>
<|tool_call_begin|>functions.get_weather:0<|tool_call_argument_begin|>
{"location": "NYC"}
<|tool_call_end|>
<|tool_call_begin|>functions.get_time:1<|tool_call_argument_begin|>
{"zone": "EST"}
<|tool_call_end|>
<|tool_calls_section_end|>"""

        result = parser.parse_delta(response)

        assert result == [
            ParsedToolCallDelta(
                index=0,
                id="get_weather:11111111",
                name="get_weather",
            ),
            ParsedToolCallDelta(
                index=0,
                arguments='\n{"location": "NYC"}\n',
            ),
            ParsedToolCallDelta(
                index=1,
                id="get_time:22222222",
                name="get_time",
            ),
            ParsedToolCallDelta(
                index=1,
                arguments='\n{"zone": "EST"}\n',
            ),
        ]


def test_parse_delta_with_content_before_tools() -> None:
    """Test streaming when there's content before tool calls section."""
    fixed_uuid = uuid.UUID("12345678-1234-5678-9abc-def012345678")
    with patch(
        "max.pipelines.architectures.kimik2_5.tool_parser.uuid.uuid4",
        return_value=fixed_uuid,
    ):
        parser = KimiToolParser()

        # Stream content then tool call
        chunks = [
            "I'll check the weather for you.\n\n",
            "<|tool_calls_section_begin|>",
            "<|tool_call_begin|>functions.get_weather:0<|tool_call_argument_begin|>",
            '{"location": "Boston"}',
            "<|tool_call_end|>",
            "<|tool_calls_section_end|>",
        ]

        all_deltas: list[ParsedToolCallDelta] = []
        for chunk in chunks:
            result = parser.parse_delta(chunk)
            if result:
                all_deltas.extend(result)

        assert all_deltas == [
            ParsedToolCallDelta(
                index=0,
                content="I'll check the weather for you.\n\n",
            ),
            ParsedToolCallDelta(
                index=0,
                id="get_weather:12345678",
                name="get_weather",
            ),
            ParsedToolCallDelta(
                index=0,
                arguments='{"location": "Boston"}',
            ),
        ]


def test_parse_delta_argument_diffing() -> None:
    """Test that argument deltas are properly diffed."""

    parser = KimiToolParser()

    # Start the tool call
    parser.parse_delta("<|tool_calls_section_begin|>")
    parser.parse_delta(
        "<|tool_call_begin|>functions.test:0<|tool_call_argument_begin|>"
    )

    # Send arguments in small chunks
    result1 = parser.parse_delta('{"key')
    result2 = parser.parse_delta('": "val')
    result3 = parser.parse_delta('ue"}')

    # Each result should only contain the new portion
    all_args = []
    for r in [result1, result2, result3]:
        if r:
            for delta in r:
                if delta.arguments:
                    all_args.append(delta.arguments)

    # Concatenated should form the full arguments
    full = "".join(all_args)
    assert '{"key": "value"}' in full or full == '{"key": "value"}'


def test_parse_delta_reset_clears_state() -> None:
    """Test that reset() clears all streaming state."""
    parser = KimiToolParser()

    # Accumulate some state
    parser.parse_delta("<|tool_calls_section_begin|>")
    parser.parse_delta(
        "<|tool_call_begin|>functions.test:0<|tool_call_argument_begin|>"
    )
    parser.parse_delta('{"key": "value"}')

    # Verify state exists
    assert parser._buffer != ""
    assert len(parser._state.tool_calls) > 0

    # Reset
    parser.reset()

    # Verify state is cleared
    assert parser._buffer == ""
    assert len(parser._state.tool_calls) == 0
    assert parser._state.sent_content_idx == 0


def test_parse_delta_partial_marker_handling() -> None:
    """Test that partial markers at buffer end are held back."""
    parser = KimiToolParser()

    # Send content that ends with partial marker
    result1 = parser.parse_delta("Hello world<|tool")

    # Should not emit the partial marker as content
    if result1:
        for delta in result1:
            if delta.content:
                assert "<|tool" not in delta.content

    # Complete the marker
    parser.parse_delta("_calls_section_begin|>")

    # Buffer should now contain the complete marker
    assert "<|tool_calls_section_begin|>" in parser._buffer


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


def test_generate_tool_call_grammar_with_tool_names(
    ll_tokenizer: LLTokenizer,
) -> None:
    """Test generating a regex grammar for constrained decoding with specific tools."""
    grammar = KimiToolParser.generate_tool_call_grammar(
        tool_names=["get_weather", "search"]
    )

    # Verify the grammar is a non-empty string
    assert isinstance(grammar, str)
    assert len(grammar) > 0

    # Verify LLMatcher can compile the grammar (will raise if invalid)
    matcher = LLMatcher(ll_tokenizer, grammar)
    assert matcher is not None


def test_generate_tool_call_grammar_without_tool_names(
    ll_tokenizer: LLTokenizer,
) -> None:
    """Test generating a regex grammar that accepts any valid identifier."""
    grammar = KimiToolParser.generate_tool_call_grammar(tool_names=None)

    # Verify the grammar is a non-empty string
    assert isinstance(grammar, str)
    assert len(grammar) > 0

    # Verify LLMatcher can compile the grammar (will raise if invalid)
    matcher = LLMatcher(ll_tokenizer, grammar)
    assert matcher is not None


def test_generate_tool_call_grammar_escapes_special_chars(
    ll_tokenizer: LLTokenizer,
) -> None:
    """Test that special regex characters in tool names are escaped."""
    # Tool names with regex special characters
    grammar = KimiToolParser.generate_tool_call_grammar(
        tool_names=["get_weather.v2", "search+plus", "tool[0]"]
    )

    # Should not raise and should produce valid grammar
    assert isinstance(grammar, str)
    assert len(grammar) > 0

    # Verify LLMatcher can compile the grammar (will raise if invalid)
    matcher = LLMatcher(ll_tokenizer, grammar)
    assert matcher is not None


def test_generate_tool_call_grammar_with_response_format_schema(
    ll_tokenizer: LLTokenizer,
) -> None:
    """Test generating a combined grammar with tools and response_format_schema."""
    response_format_schema = {
        "type": "object",
        "properties": {
            "answer": {"type": "string"},
            "confidence": {"type": "number"},
        },
        "required": ["answer"],
    }

    grammar = KimiToolParser.generate_tool_call_grammar(
        tool_names=["get_weather", "search"],
        response_format_schema=response_format_schema,
    )

    # Verify the grammar is a non-empty string
    assert isinstance(grammar, str)
    assert len(grammar) > 0

    # Combined grammar should contain alternation syntax (Lark format)
    # It should reference both tool_calls and json_response
    assert "tool_calls" in grammar
    assert "json_response" in grammar
    assert "%json" in grammar  # JSON schema embedding

    # Verify LLMatcher can compile the grammar (will raise if invalid)
    matcher = LLMatcher(ll_tokenizer, grammar)
    assert matcher is not None


def test_generate_tool_call_grammar_combined_accepts_json_object_type(
    ll_tokenizer: LLTokenizer,
) -> None:
    """Test combined grammar with json_object type (any valid JSON)."""
    # json_object mode uses a permissive schema
    response_format_schema = {"type": "object"}

    grammar = KimiToolParser.generate_tool_call_grammar(
        tool_names=["calculate"],
        response_format_schema=response_format_schema,
    )

    assert isinstance(grammar, str)
    assert len(grammar) > 0
    assert "json_response" in grammar

    # Verify LLMatcher can compile the grammar (will raise if invalid)
    matcher = LLMatcher(ll_tokenizer, grammar)
    assert matcher is not None


def test_generate_tool_call_grammar_no_schema_returns_regex_grammar(
    ll_tokenizer: LLTokenizer,
) -> None:
    """Test that without response_format_schema, we get regex-only grammar."""
    grammar = KimiToolParser.generate_tool_call_grammar(
        tool_names=["get_weather"],
        response_format_schema=None,
    )

    # Without schema, should return grammar from grammar_from_regex()
    assert isinstance(grammar, str)
    assert len(grammar) > 0
    # The regex grammar should NOT contain JSON schema embedding
    assert "%json" not in grammar
    # Should contain the tool call pattern
    assert "tool_calls_section_begin" in grammar

    # Verify LLMatcher can compile the grammar (will raise if invalid)
    matcher = LLMatcher(ll_tokenizer, grammar)
    assert matcher is not None


def test_parser_handles_json_content_when_no_tool_calls() -> None:
    """Test that parser returns content as-is when no tool call markers present."""
    parser = KimiToolParser()

    # This is what the model would output when choosing JSON content
    # instead of tool calls (with combined grammar)
    json_response = '{"answer": "The weather is sunny", "confidence": 0.95}'

    result = parser.parse_complete(json_response)

    # No tool calls should be parsed
    assert len(result.tool_calls) == 0
    # Content should be returned as-is
    assert result.content == json_response
