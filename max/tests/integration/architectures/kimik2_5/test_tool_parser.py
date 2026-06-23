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
from typing import Any, cast
from unittest.mock import MagicMock, patch

import llguidance.numpy
import numpy as np
import pytest
from llguidance import LLMatcher, LLTokenizer
from llguidance._tokenizer import TokenizerWrapper
from max.pipelines.architectures.kimik2_5.tool_parser import (
    _MAX_TOOL_CALL_SECTIONS,
    IM_END,
    THINK_END,
    THINK_START,
    TOOL_CALL_ARGUMENT_BEGIN,
    TOOL_CALL_BEGIN,
    TOOL_CALL_END,
    TOOL_CALLS_SECTION_BEGIN,
    TOOL_CALLS_SECTION_END,
    KimiToolParser,
)
from max.pipelines.context import (
    GrammarEnforcementState,
    StructuredOutputRegionDelimiters,
    TextContext,
    TokenBuffer,
)
from max.pipelines.lib.pipeline_variants.overlap_text_generation import (
    MAGIC_DRAFT_TOKEN_ID,
)
from max.pipelines.lib.pipeline_variants.utils import StructuredOutputHelper
from max.pipelines.lib.tool_parsing import StreamingToolCallState
from max.pipelines.modeling.types import (
    ParsedToolCall,
    ParsedToolCallDelta,
    ParsedToolResponse,
    PipelineTokenizer,
)


class _MinimalTokenizer:
    """Byte tokenizer extended with Kimi K2.5 special tokens.

    Maps byte values 0-255 to token IDs 0-255, then assigns dedicated IDs to
    each Kimi structural / reasoning / turn-terminator token so that grammar
    generation can resolve them via ``convert_tokens_to_ids`` and the
    ``LLMatcher`` sees them as single tokens — matching how the real Kimi
    tokenizer encodes these markers. This is what lets the grammar's
    ``/[\\s\\S]*/`` bodies terminate atomically at the closing marker.
    """

    _SPECIAL_TOKENS: dict[str, int] = {
        TOOL_CALLS_SECTION_BEGIN: 256,
        TOOL_CALLS_SECTION_END: 257,
        TOOL_CALL_BEGIN: 258,
        TOOL_CALL_END: 259,
        TOOL_CALL_ARGUMENT_BEGIN: 260,
        THINK_START: 261,
        THINK_END: 262,
        IM_END: 263,
    }
    _N_VOCAB: int = 264

    eos_token_id: int = 0
    bos_token_id: int | None = None
    unk_token_id: int | None = None

    def __init__(self) -> None:
        self.tokens: list[bytes] = [bytes([i]) for i in range(256)]
        self.tokens.extend(t.encode("utf-8") for t in self._SPECIAL_TOKENS)

    def convert_tokens_to_ids(self, token: str) -> int | None:
        return self._SPECIAL_TOKENS.get(token)

    def __call__(self, s: bytes | str) -> list[int]:
        if isinstance(s, str):
            s = s.encode("utf-8")
        result: list[int] = []
        i = 0
        while i < len(s):
            for text, tid in sorted(
                self._SPECIAL_TOKENS.items(), key=lambda x: -len(x[0])
            ):
                encoded = text.encode("utf-8")
                if s[i : i + len(encoded)] == encoded:
                    result.append(tid)
                    i += len(encoded)
                    break
            else:
                result.append(s[i])
                i += 1
        return result


@pytest.fixture(scope="module")
def minimal_tokenizer() -> _MinimalTokenizer:
    """Raw byte+special-token tokenizer for grammar validation tests."""
    return _MinimalTokenizer()


@pytest.fixture(scope="module")
def mock_tokenizer(
    minimal_tokenizer: _MinimalTokenizer,
) -> PipelineTokenizer[Any, Any, Any]:
    """PipelineTokenizer stub whose ``.delegate`` is the minimal tokenizer."""
    stub = cast(PipelineTokenizer[Any, Any, Any], MagicMock())
    stub.delegate = minimal_tokenizer  # type: ignore[attr-defined]
    return stub


@pytest.fixture(scope="module")
def ll_tokenizer(minimal_tokenizer: _MinimalTokenizer) -> LLTokenizer:
    """Create a minimal LLTokenizer for grammar validation tests."""
    wrapper = TokenizerWrapper(minimal_tokenizer)
    return LLTokenizer(wrapper, n_vocab=_MinimalTokenizer._N_VOCAB)


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

    # parse_delta should accumulate tokens; return [] to indicate parser is actively buffering and raw tokens shouldn't be used yet.
    result1 = parser.parse_delta("<|tool_calls")
    assert result1 == []

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

    result_pre = parser.parse_delta("<|tool_calls")
    assert result_pre == []

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


def _tools(*names: str) -> list[dict[str, Any]]:
    """Build a minimal OpenAI-style tools list from function names."""
    return [{"type": "function", "function": {"name": n}} for n in names]


def _section(name: str, idx: int, args: str) -> str:
    """Builds one ``<|tool_calls_section_begin|>...end|>`` block string."""
    return (
        f"{TOOL_CALLS_SECTION_BEGIN}"
        f"{TOOL_CALL_BEGIN}functions.{name}:{idx}{TOOL_CALL_ARGUMENT_BEGIN}"
        f"{args}{TOOL_CALL_END}"
        f"{TOOL_CALLS_SECTION_END}"
    )


def test_generate_tool_call_grammar_with_tool_names(
    ll_tokenizer: LLTokenizer,
    mock_tokenizer: PipelineTokenizer[Any, Any, Any],
) -> None:
    """Test generating a Lark grammar for constrained decoding with tools."""
    grammar = KimiToolParser.generate_tool_call_grammar(
        tools=_tools("get_weather", "search"), tokenizer=mock_tokenizer
    )

    assert isinstance(grammar, str)
    assert len(grammar) > 0

    # Verify LLMatcher can compile the grammar (will raise if invalid)
    matcher = LLMatcher(ll_tokenizer, grammar)
    assert matcher is not None


def test_generate_tool_call_grammar_without_tool_names(
    ll_tokenizer: LLTokenizer,
    mock_tokenizer: PipelineTokenizer[Any, Any, Any],
) -> None:
    """Test generating a grammar that accepts any valid identifier."""
    grammar = KimiToolParser.generate_tool_call_grammar(
        tools=None, tokenizer=mock_tokenizer
    )

    assert isinstance(grammar, str)
    assert len(grammar) > 0

    matcher = LLMatcher(ll_tokenizer, grammar)
    assert matcher is not None


def test_generate_tool_call_grammar_escapes_special_chars(
    ll_tokenizer: LLTokenizer,
    mock_tokenizer: PipelineTokenizer[Any, Any, Any],
) -> None:
    """Test that special characters in tool names are escaped for Lark."""
    grammar = KimiToolParser.generate_tool_call_grammar(
        tools=_tools("get_weather.v2", "search+plus", "tool[0]"),
        tokenizer=mock_tokenizer,
    )

    assert isinstance(grammar, str)
    assert len(grammar) > 0

    matcher = LLMatcher(ll_tokenizer, grammar)
    assert matcher is not None


def test_generate_tool_call_grammar_requires_tokenizer() -> None:
    """Grammar generation must raise without a tokenizer to resolve IDs."""
    with pytest.raises(ValueError, match=r"tokenizer is required"):
        KimiToolParser.generate_tool_call_grammar(tools=_tools("get_weather"))


def test_generate_tool_call_grammar_uses_single_token_refs(
    mock_tokenizer: PipelineTokenizer[Any, Any, Any],
) -> None:
    """Structural markers are emitted as ``<[id]>`` single-token references.

    Literal ``<|...|>`` marker text must NOT appear in the grammar — the
    ``|`` would collide with Lark's alternation operator, and byte-literal
    matching is what the migration away from ``grammar_from_regex`` removed.
    """
    grammar = KimiToolParser.generate_tool_call_grammar(
        tools=_tools("get_weather"), tokenizer=mock_tokenizer
    )

    assert "<[256]>" in grammar  # section-begin id from _MinimalTokenizer
    assert TOOL_CALLS_SECTION_BEGIN not in grammar
    assert "%json" not in grammar


def test_generate_tool_call_grammar_with_response_format_schema(
    ll_tokenizer: LLTokenizer,
    mock_tokenizer: PipelineTokenizer[Any, Any, Any],
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
        tools=_tools("get_weather", "search"),
        response_format_schema=response_format_schema,
        tokenizer=mock_tokenizer,
    )

    assert isinstance(grammar, str)
    assert len(grammar) > 0

    # Combined grammar references both the tool_calls branch and json_response.
    assert "tool_calls" in grammar
    assert "json_response" in grammar
    assert "%json" in grammar  # JSON schema embedding

    matcher = LLMatcher(ll_tokenizer, grammar)
    assert matcher is not None


def test_generate_tool_call_grammar_combined_accepts_json_object_type(
    ll_tokenizer: LLTokenizer,
    mock_tokenizer: PipelineTokenizer[Any, Any, Any],
) -> None:
    """Test combined grammar with json_object type (any valid JSON)."""
    response_format_schema = {"type": "object"}

    grammar = KimiToolParser.generate_tool_call_grammar(
        tools=_tools("calculate"),
        response_format_schema=response_format_schema,
        tokenizer=mock_tokenizer,
    )

    assert isinstance(grammar, str)
    assert len(grammar) > 0
    assert "json_response" in grammar

    matcher = LLMatcher(ll_tokenizer, grammar)
    assert matcher is not None


def test_grammar_accepts_up_to_max_sections(
    ll_tokenizer: LLTokenizer,
    minimal_tokenizer: _MinimalTokenizer,
    mock_tokenizer: PipelineTokenizer[Any, Any, Any],
) -> None:
    """Matcher accepts up to ``_MAX_TOOL_CALL_SECTIONS`` back-to-back.

    Multiple back-to-back sections are what ``tool_choice=auto`` produces
    when the model re-enters tool calling after content; the matcher must
    accept the re-entry rather than rejecting the second section-begin (the
    prod desync this change fixes). Derives the count from the constant so
    the bound and the test stay in lockstep.
    """
    grammar = KimiToolParser.generate_tool_call_grammar(
        tools=_tools("get_weather"), tokenizer=mock_tokenizer
    )
    matcher = LLMatcher(ll_tokenizer, grammar)

    max_sections = "".join(
        _section("get_weather", i, '{"location": "NYC"}')
        for i in range(_MAX_TOOL_CALL_SECTIONS)
    )
    tokens = minimal_tokenizer(max_sections)
    consumed = matcher.try_consume_tokens(tokens)
    assert consumed == len(tokens), (
        f"matcher should accept {_MAX_TOOL_CALL_SECTIONS} sections but "
        f"rejected at offset {consumed} of {len(tokens)}; "
        f"error: {matcher.get_error()}"
    )


def test_grammar_caps_at_max_sections(
    ll_tokenizer: LLTokenizer,
    minimal_tokenizer: _MinimalTokenizer,
    mock_tokenizer: PipelineTokenizer[Any, Any, Any],
) -> None:
    """The section past ``_MAX_TOOL_CALL_SECTIONS`` must be refused.

    Guards against silently lifting the cap; the cap is a secondary backstop
    (``max_tokens`` is the primary ceiling) that keeps a stuck model from
    holding a GPU slot. Derives the count from the constant.
    """
    grammar = KimiToolParser.generate_tool_call_grammar(
        tools=_tools("get_weather"), tokenizer=mock_tokenizer
    )
    matcher = LLMatcher(ll_tokenizer, grammar)

    max_sections = "".join(
        _section("get_weather", i, '{"location": "NYC"}')
        for i in range(_MAX_TOOL_CALL_SECTIONS)
    )
    consumed = matcher.try_consume_tokens(minimal_tokenizer(max_sections))
    assert consumed == len(minimal_tokenizer(max_sections))

    over_cap_begin = minimal_tokenizer(TOOL_CALLS_SECTION_BEGIN)
    consumed_over = matcher.try_consume_tokens(over_cap_begin)
    assert consumed_over == 0, (
        f"matcher should reject section-begin #{_MAX_TOOL_CALL_SECTIONS + 1} "
        f"past the cap but accepted {consumed_over} of "
        f"{len(over_cap_begin)} tokens"
    )


def test_grammar_accepts_interleaved_thinking(
    ll_tokenizer: LLTokenizer,
    minimal_tokenizer: _MinimalTokenizer,
    mock_tokenizer: PipelineTokenizer[Any, Any, Any],
) -> None:
    """Matcher accepts ``<think>...</think>`` blocks between tool sections.

    This is Kimi's interleaved thinking: reason, call, reason, call. The
    reasoning body is byte-level freeform text terminated atomically by
    ``</think>`` and may contain ``<`` and markup.
    """
    grammar = KimiToolParser.generate_tool_call_grammar(
        tools=_tools("get_weather", "search"), tokenizer=mock_tokenizer
    )
    matcher = LLMatcher(ll_tokenizer, grammar)

    interleaved = (
        _section("get_weather", 0, '{"location": "NYC"}')
        + f"{THINK_START}Now let me search for x < y markup</p>{THINK_END}"
        + _section("search", 1, '{"q": "if (a < b)"}')
        + IM_END
    )
    tokens = minimal_tokenizer(interleaved)
    consumed = matcher.try_consume_tokens(tokens)
    assert consumed == len(tokens), (
        f"matcher should accept interleaved think+sections+im_end but "
        f"rejected at offset {consumed} of {len(tokens)}; "
        f"error: {matcher.get_error()}"
    )


def test_grammar_rejects_im_end_mid_section(
    ll_tokenizer: LLTokenizer,
    minimal_tokenizer: _MinimalTokenizer,
    mock_tokenizer: PipelineTokenizer[Any, Any, Any],
) -> None:
    """``<|im_end|>`` is only allowed at accepting states, not mid-section."""
    grammar = KimiToolParser.generate_tool_call_grammar(
        tools=_tools("get_weather"), tokenizer=mock_tokenizer
    )
    matcher = LLMatcher(ll_tokenizer, grammar)

    # Open a section, then try to terminate before closing it.
    prefix = (
        f"{TOOL_CALLS_SECTION_BEGIN}{TOOL_CALL_BEGIN}functions.get_weather:0"
    )
    consumed_prefix = matcher.try_consume_tokens(minimal_tokenizer(prefix))
    assert consumed_prefix == len(minimal_tokenizer(prefix))

    im_end_tokens = minimal_tokenizer(IM_END)
    consumed = matcher.try_consume_tokens(im_end_tokens)
    assert consumed == 0, (
        "matcher should reject <|im_end|> in the middle of an open section"
    )


def test_grammar_accepts_unbounded_argument_body(
    ll_tokenizer: LLTokenizer,
    minimal_tokenizer: _MinimalTokenizer,
    mock_tokenizer: PipelineTokenizer[Any, Any, Any],
) -> None:
    """Argument body has no length cap; matcher must accept >8192 chars.

    The argument body is intentionally unbounded — only ``max_tokens`` /
    context bounds it — so legitimate large arguments (file blobs, embedded
    documents, search-result payloads) are not silently truncated. Feeds a
    ~10 KB argument and verifies every token is consumed.
    """
    grammar = KimiToolParser.generate_tool_call_grammar(
        tools=_tools("echo_document"), tokenizer=mock_tokenizer
    )
    matcher = LLMatcher(ll_tokenizer, grammar)

    filler = ("abcdefghijklmnopqrstuvwxyz0123456789 " * 300)[:10_000]
    assert len(filler) > 8192

    tool_call = _section("echo_document", 0, f'{{"content": "{filler}"}}')
    tokens = minimal_tokenizer(tool_call)
    # Feed in chunks so a partial reject is localisable to its context.
    chunk = 64
    consumed = 0
    for start in range(0, len(tokens), chunk):
        batch = tokens[start : start + chunk]
        n = matcher.try_consume_tokens(batch)
        if n != len(batch):
            raise AssertionError(
                f"matcher rejected token at offset {start + n} of "
                f"{len(tokens)} (consumed {consumed + n} so far); "
                f"context={tool_call[max(0, start + n - 20) : start + n + 20]!r}"
            )
        consumed += n

    assert consumed == len(tokens)


def test_grammar_accepts_less_than_in_arguments(
    ll_tokenizer: LLTokenizer,
    minimal_tokenizer: _MinimalTokenizer,
    mock_tokenizer: PipelineTokenizer[Any, Any, Any],
) -> None:
    """Grammar accepts ``<`` anywhere in the argument body.

    Because ``<|tool_call_end|>`` is referenced as an atomic single token
    (not a byte literal), the argument body terminates cleanly at the
    closing marker and may contain ``<`` freely — code comparisons,
    HTML/XML, JSX, and git-diff conflict markers — without triggering
    premature tag detection. This is a deliberate relaxation from the old
    byte-level regex grammar, which had to reject ``<`` outside JSON strings.
    """
    grammar = KimiToolParser.generate_tool_call_grammar(
        tools=_tools("write_file"), tokenizer=mock_tokenizer
    )

    test_payloads = [
        '{"content": "if (x < y) { return x; }"}',
        '{"content": "<html><body><p>Hello</p></body></html>"}',
        '{"content": "const App = () => <div><span>Hi</span></div>;"}',
        '{"content": "<<<<<<< HEAD\\nold\\n=======\\nnew\\n>>>>>>> branch"}',
        '{"code": "a < b", "html": "<p>text</p>", "note": "x<y<z"}',
        # < even outside JSON strings is fine: the atomic end marker frames
        # the body, so the grammar no longer needs to detect tags via "<".
        '{"done": true} < extra',
    ]

    for payload in test_payloads:
        matcher = LLMatcher(ll_tokenizer, grammar)
        tokens = minimal_tokenizer(_section("write_file", 0, payload))
        consumed = matcher.try_consume_tokens(tokens)
        assert consumed == len(tokens), (
            f"matcher rejected payload at offset {consumed} of "
            f"{len(tokens)}; payload={payload!r}"
        )


def _simulate_auto_enforcement(
    state: GrammarEnforcementState,
    matcher: LLMatcher,
    tokenizer: _MinimalTokenizer,
    text: str,
) -> tuple[bool, int | None]:
    """Drives ``tool_choice=auto`` enforcement token-by-token over ``text``.

    Mirrors the committed-token path of
    ``StructuredOutputHelper.advance_fsm_and_compute_bitmasks``: each token
    advances the enforcement state machine, and tokens are fed to the matcher
    only while ``update_enforcement_state`` says to. Returns
    ``(rejected, token)`` where ``rejected`` is True if the matcher refused a
    token the state machine fed it (the prod desync fingerprint).
    """
    for token in tokenizer(text):
        if state.update_enforcement_state(token):
            if matcher.try_consume_tokens([token]) != 1:
                return True, token
    return False, None


def test_auto_mode_multi_section_no_matcher_desync(
    ll_tokenizer: LLTokenizer,
    minimal_tokenizer: _MinimalTokenizer,
    mock_tokenizer: PipelineTokenizer[Any, Any, Any],
) -> None:
    """Regression for the prod ``Async matcher rejected token`` desync.

    In ``tool_choice=auto`` the enforcement state machine flips enforcement
    off at ``<|tool_calls_section_end|>`` and back on at the next
    ``<|tool_calls_section_begin|>``. With the old single-section grammar the
    matcher was terminal after the first section, so re-feeding a second
    section-begin was rejected (``role=bonus``) and enforcement was disabled
    for the rest of the request. With the multi-section grammar the matcher
    accepts re-entry; this drives two sections separated by free content
    (as auto mode produces) and asserts no rejection.
    """
    grammar = KimiToolParser.generate_tool_call_grammar(
        tools=_tools("get_weather", "search"), tokenizer=mock_tokenizer
    )
    matcher = LLMatcher(ll_tokenizer, grammar)

    state = GrammarEnforcementState(
        grammar_enforced=False,
        tools_forced=False,
        tool_region=StructuredOutputRegionDelimiters(
            start_token_ids=minimal_tokenizer(TOOL_CALLS_SECTION_BEGIN),
            end_token_ids=minimal_tokenizer(TOOL_CALLS_SECTION_END),
        ),
    )

    stream = (
        _section("get_weather", 0, '{"location": "NYC"}')
        # Free content between sections — NOT fed to the matcher because
        # enforcement is off here; this is the gap the old grammar tripped on.
        + "Let me also check the time."
        + _section("search", 1, '{"q": "time in NYC"}')
    )

    rejected, token = _simulate_auto_enforcement(
        state, matcher, minimal_tokenizer, stream
    )
    assert not rejected, (
        f"matcher rejected token {token} mid-stream — the multi-section "
        f"re-entry desync regressed; matcher error: {matcher.get_error()}"
    )
    # Enforcement ends off: the final section-end flipped it back off.
    assert state.grammar_enforced is False


def test_parse_complete_multiple_sections() -> None:
    """parse_complete aggregates tool calls across multiple sections.

    Kimi emits multiple ``<|tool_calls_section_begin|>...end|>`` blocks per
    turn. The parser must return every call across all sections and must not
    leak inter-section text (here a reasoning block) into a tool call.
    """
    parser = KimiToolParser()

    response = (
        _section("get_weather", 0, '{"location": "NYC"}')
        + f"{THINK_START}now the time{THINK_END}"
        + _section("get_time", 1, '{"zone": "EST"}')
    )

    result = parser.parse_complete(response)

    assert result.content is None
    assert [tc.name for tc in result.tool_calls] == ["get_weather", "get_time"]
    assert json.loads(result.tool_calls[0].arguments) == {"location": "NYC"}
    assert json.loads(result.tool_calls[1].arguments) == {"zone": "EST"}
    # The inter-section reasoning must not have leaked into either call.
    for tc in result.tool_calls:
        assert "now the time" not in tc.arguments
        assert "think" not in tc.arguments


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


def _unpack_bitmask(packed: np.ndarray, vocab_size: int) -> np.ndarray:
    """Unpack a packed int32 bitmask ``[..., ceil(vocab/32)]`` to bool
    ``[..., vocab]`` so tests can index by token id and use ``.all()`` to mean
    "fully unconstrained".

    Mirrors the GPU ``apply_packed_bitmask`` layout: bit ``t`` lives in word
    ``t >> 5`` at position ``t & 31``.
    """
    masks = np.int32(1) << np.arange(32, dtype=np.int32)
    bits = (packed[..., np.newaxis] & masks) != 0
    bits = bits.reshape(*packed.shape[:-1], -1)
    return bits[..., :vocab_size]


def test_sync_fill_constrains_name_only_after_section_consumed(
    ll_tokenizer: LLTokenizer,
    mock_tokenizer: PipelineTokenizer[Any, Any, Any],
) -> None:
    """Slot 0 is constrained only once the FSM has consumed the section opener.

    With ``grammar_enforced`` still False (section opener not yet consumed)
    the first sampled slot is unconstrained; after the
    FSM advances through the section / call / "functions." prefix it is
    constrained to the menu. This is why the sync path must wait for the
    callback's FSM advance.
    """
    section_begin = 256  # _MinimalTokenizer special-token id
    section_end = 257
    call_begin = 258
    grammar = KimiToolParser.generate_tool_call_grammar(
        tools=_tools("get_weather"), tokenizer=mock_tokenizer
    )
    matcher = LLMatcher(ll_tokenizer, grammar)

    helper = StructuredOutputHelper(
        enabled=True,
        vocab_size=_MinimalTokenizer._N_VOCAB,
        tool_call_region_delimiters=StructuredOutputRegionDelimiters(
            start_token_ids=[section_begin], end_token_ids=[section_end]
        ),
    )

    ctx = TextContext(max_length=4096, tokens=TokenBuffer(np.array([1])))
    ctx._matcher = matcher
    ctx.set_tool_region(
        start_token_ids=[section_begin], end_token_ids=[section_end]
    )
    assert not ctx.grammar_enforced  # auto mode, before the section opener

    num_positions = 2
    drafts = np.zeros((1, num_positions - 1), dtype=np.int64)

    # Stale FSM: section opener not yet consumed -> the name slot (slot 0) is
    # left fully unconstrained, so the model could sample any name.
    stale = _unpack_bitmask(
        helper.compute_speculative_bitmasks([ctx], drafts, num_positions),
        _MinimalTokenizer._N_VOCAB,
    )
    assert stale[0, 0].all()

    # Advance the FSM through the section/call/"functions." prefix, exactly as
    # the async callback's Part 1 would before the sync fill reads it.
    prefix = [section_begin, call_begin] + list(b"functions.")
    for tok in prefix:
        if ctx.update_enforcement_state(tok):
            assert matcher.try_consume_tokens([tok]) == 1
    assert ctx.grammar_enforced

    # Current FSM: the name slot is now constrained to the menu. The first
    # byte of "get_weather" is allowed; a byte no menu name starts with is not.
    current = _unpack_bitmask(
        helper.compute_speculative_bitmasks([ctx], drafts, num_positions),
        _MinimalTokenizer._N_VOCAB,
    )
    assert not current[0, 0].all()
    assert current[0, 0, ord("g")]  # tool: get_weather
    assert not current[0, 0, ord("z")]


def test_sync_fill_with_placeholder_drafts_leaves_bonus_slot_unconstrained(
    ll_tokenizer: LLTokenizer,
    mock_tokenizer: PipelineTokenizer[Any, Any, Any],
) -> None:
    """Reproduces the residual composition-change desync.

    On the sync-prime path the draft tokens are MAGIC placeholders (the real
    drafts live only on-device, scattered in by ``realize_future_tokens``). The
    speculative fill breaks at the first invalid placeholder, leaving every
    slot after slot 0 -- including the bonus position -- unconstrained. With
    ``num_accepted >= 1`` the bonus is then sampled freely and the matcher
    rejects it on replay. A real draft constrains that slot, which is what
    gather-by-rid restores by reusing the callback's real-draft precompute.
    """
    section_begin, section_end, call_begin = 256, 257, 258
    grammar = KimiToolParser.generate_tool_call_grammar(
        tools=_tools("get_weather"), tokenizer=mock_tokenizer
    )
    matcher = LLMatcher(ll_tokenizer, grammar)
    helper = StructuredOutputHelper(
        enabled=True,
        vocab_size=_MinimalTokenizer._N_VOCAB,
        tool_call_region_delimiters=StructuredOutputRegionDelimiters(
            start_token_ids=[section_begin], end_token_ids=[section_end]
        ),
    )
    ctx = TextContext(max_length=4096, tokens=TokenBuffer(np.array([1])))
    ctx._matcher = matcher
    ctx.set_tool_region(
        start_token_ids=[section_begin], end_token_ids=[section_end]
    )
    # Advance into the tool-call header so the name slot is constrained.
    for tok in [section_begin, call_begin] + list(b"functions."):
        if ctx.update_enforcement_state(tok):
            assert matcher.try_consume_tokens([tok]) == 1
    assert ctx.grammar_enforced

    num_positions = 2  # K=1 -> slot 0 (first sampled) + slot 1 (bonus)

    # Placeholder draft (what the sync path actually has): the fill breaks at
    # the invalid placeholder, leaving the bonus slot unconstrained.
    magic = np.full((1, 1), MAGIC_DRAFT_TOKEN_ID, dtype=np.int64)
    bad = _unpack_bitmask(
        helper.compute_speculative_bitmasks([ctx], magic, num_positions),
        _MinimalTokenizer._N_VOCAB,
    )
    assert not bad[0, 0].all()  # slot 0 still constrained
    assert bad[0, 1].all()  # bonus slot UNCONSTRAINED -- the bug

    # Real draft (what adopt / gather provides): the bonus slot is constrained.
    real = np.array([[ord("g")]], dtype=np.int64)  # first byte of get_weather
    good = _unpack_bitmask(
        helper.compute_speculative_bitmasks([ctx], real, num_positions),
        _MinimalTokenizer._N_VOCAB,
    )
    assert not good[0, 1].all()  # bonus slot constrained
    assert good[0, 1, ord("e")]  # "get_weather" continues with 'e'
    assert not good[0, 1, ord("z")]  # an out-of-name byte is forbidden


def _build_auto_ctx(
    ll_tokenizer: LLTokenizer,
    mock_tokenizer: PipelineTokenizer[Any, Any, Any],
) -> tuple[StructuredOutputHelper, Any, LLMatcher]:
    """A ``tool_choice=auto`` helper + context wired to the real Kimi grammar."""
    section_begin, section_end = 256, 257
    grammar = KimiToolParser.generate_tool_call_grammar(
        tools=_tools("get_weather"), tokenizer=mock_tokenizer
    )
    matcher = LLMatcher(ll_tokenizer, grammar)
    helper = StructuredOutputHelper(
        enabled=True,
        vocab_size=_MinimalTokenizer._N_VOCAB,
        tool_call_region_delimiters=StructuredOutputRegionDelimiters(
            start_token_ids=[section_begin], end_token_ids=[section_end]
        ),
    )
    ctx = TextContext(max_length=4096, tokens=TokenBuffer(np.array([1])))
    ctx._matcher = matcher
    ctx.set_tool_region(
        start_token_ids=[section_begin], end_token_ids=[section_end]
    )
    return helper, ctx, matcher


def _commit(ctx: Any, matcher: LLMatcher, tokens: list[int]) -> None:
    """Advance enforcement + matcher over committed tokens (Part-1 mirror)."""
    for tok in tokens:
        if ctx.update_enforcement_state(tok):
            assert matcher.try_consume_tokens([tok]) == 1, (
                f"prefix token {tok} unexpectedly rejected"
            )


@pytest.mark.parametrize(
    "prefix, structural_tag, tail, label",
    [
        # Matcher sits right after ``functions.get_weather:0``; the next
        # token is ``<|tool_call_argument_begin|>`` (260).
        (
            [256, 258] + list(b"functions.get_weather:0"),
            260,
            list(b'{"'),
            "arg_begin",
        ),
        # Matcher sits right after the section opener; the next token is
        # ``<|tool_call_begin|>`` (258).
        ([256], 258, list(b"functions."), "call_begin"),
    ],
)
def test_spec_decode_walk_preserves_matcher_state_across_structural_tag(
    ll_tokenizer: LLTokenizer,
    mock_tokenizer: PipelineTokenizer[Any, Any, Any],
    prefix: list[int],
    structural_tag: int,
    tail: list[int],
    label: str,
) -> None:
    """Speculative walk must preserve matcher state across structural tags.

    The walk consumes draft tokens that cross a tool-call structural tag
    (``<|tool_call_begin|>`` / ``<|tool_call_argument_begin|>``) and must leave
    the matcher unchanged, so the next batch can still consume that same tag.
    Guards the ``deep_copy`` fix against the ``rollback()`` desync.
    """
    helper, ctx, matcher = _build_auto_ctx(ll_tokenizer, mock_tokenizer)

    # Commit up to just before the structural tag.
    _commit(ctx, matcher, prefix)
    assert ctx.grammar_enforced, f"[{label}] enforcement should be on"

    # Sanity: the structural tag is legal at the current (pre-walk) state.
    pre = _unpack_bitmask(
        helper.compute_speculative_bitmasks(
            [ctx], np.zeros((1, 0), dtype=np.int64), 1
        ),
        _MinimalTokenizer._N_VOCAB,
    )
    assert pre[0, 0, structural_tag], (
        f"[{label}] structural tag {structural_tag} not legal pre-walk"
    )

    # Speculative walk over real drafts that cross the structural tag.
    drafts = np.array([[structural_tag, *tail]], dtype=np.int64)
    helper.compute_speculative_bitmasks(
        [ctx], drafts, num_positions=drafts.shape[1] + 1
    )

    # The walk must leave the matcher unchanged. The next committed token is
    # the same structural tag; it must still be consumable.
    assert ctx.update_enforcement_state(structural_tag), (
        f"[{label}] tag should still be grammar content after walk"
    )
    assert matcher.try_consume_tokens([structural_tag]) == 1, (
        f"[{label}] matcher drifted past structural tag {structural_tag} "
        f"-- speculative walk did not preserve matcher state"
    )


@pytest.mark.parametrize(
    "prefix, walk, retry, rollback_is_inverse, label",
    [
        # Cross the ``(tool_call){0,N}`` repetition boundary: CALL_BEGIN is the
        # entry token of the repeated ``tool_call`` non-terminal. rollback is
        # NOT a perfect inverse here -- this is the defect that motivates the
        # deep_copy fix in ``_speculatively_fill_bitmask_window``.
        ([256], [258] + list(b"functions."), 258, False, "call_begin_entry"),
        # Stay inside the single ``tool_call`` rule body: ARG_BEGIN is mid-rule.
        # rollback IS a perfect inverse here.
        (
            [256, 258] + list(b"functions.get_weather:0"),
            [260] + list(b'{"'),
            260,
            True,
            "arg_begin_mid_rule",
        ),
    ],
)
def test_llguidance_rollback_inverse_behavior_across_rule_boundary(
    ll_tokenizer: LLTokenizer,
    mock_tokenizer: PipelineTokenizer[Any, Any, Any],
    prefix: list[int],
    walk: list[int],
    retry: int,
    rollback_is_inverse: bool,
    label: str,
) -> None:
    """Characterizes ``LLMatcher.rollback``: inverse mid-rule, NOT across a
    rule/repetition boundary.

    Bypasses all MAX code (raw llguidance matcher only). This is a tripwire: it
    pins the upstream behavior that ``_speculatively_fill_bitmask_window`` works
    around by speculating on a ``deep_copy`` instead of consume+rollback. If a
    future llguidance bump makes rollback a perfect inverse across rule
    boundaries, the ``call_begin_entry`` case will flip and this test will fail
    -- signalling that the deep_copy workaround could be simplified back to
    rollback.
    """
    grammar = KimiToolParser.generate_tool_call_grammar(
        tools=_tools("get_weather"), tokenizer=mock_tokenizer
    )
    matcher = LLMatcher(ll_tokenizer, grammar)
    for tok in prefix:
        assert matcher.try_consume_tokens([tok]) == 1

    before = llguidance.numpy.allocate_token_bitmask(
        1, _MinimalTokenizer._N_VOCAB
    )
    llguidance.numpy.fill_next_token_bitmask(matcher, before, index=0)

    consumed = 0
    for tok in walk:
        if matcher.try_consume_tokens([tok]) == 1:
            consumed += 1
        else:
            break
    matcher.rollback(consumed)

    after = llguidance.numpy.allocate_token_bitmask(
        1, _MinimalTokenizer._N_VOCAB
    )
    llguidance.numpy.fill_next_token_bitmask(matcher, after, index=0)

    restored = bool(np.array_equal(before, after)) and (
        matcher.try_consume_tokens([retry]) == 1
    )
    assert restored == rollback_is_inverse, (
        f"[{label}] rollback({consumed}) inverse={restored}, "
        f"expected {rollback_is_inverse} -- upstream llguidance rollback "
        f"behavior across this boundary changed"
    )


def test_deep_copy_walk_avoids_rule_boundary_rollback_desync(
    ll_tokenizer: LLTokenizer,
    mock_tokenizer: PipelineTokenizer[Any, Any, Any],
) -> None:
    """Validates the fix direction: speculate on a ``deep_copy``, never rollback.

    Walking draft tokens on ``matcher.deep_copy()`` leaves the real matcher
    completely untouched, so the rule-boundary ``rollback`` defect cannot
    desync it. The original matcher must still accept ``<|tool_call_begin|>``
    (258) as the next committed token -- the exact case ``rollback`` breaks.
    """
    grammar = KimiToolParser.generate_tool_call_grammar(
        tools=_tools("get_weather"), tokenizer=mock_tokenizer
    )
    matcher = LLMatcher(ll_tokenizer, grammar)
    assert matcher.try_consume_tokens([256]) == 1  # section_begin

    before = llguidance.numpy.allocate_token_bitmask(
        1, _MinimalTokenizer._N_VOCAB
    )
    llguidance.numpy.fill_next_token_bitmask(matcher, before, index=0)

    # Speculative walk on a throwaway copy across the rule-entry boundary.
    scratch = matcher.deep_copy()
    for tok in [258, *list(b"functions.")]:
        if scratch.try_consume_tokens([tok]) != 1:
            break

    after = llguidance.numpy.allocate_token_bitmask(
        1, _MinimalTokenizer._N_VOCAB
    )
    llguidance.numpy.fill_next_token_bitmask(matcher, after, index=0)

    assert np.array_equal(before, after), (
        "real matcher mask changed despite walking only the deep_copy"
    )
    assert matcher.try_consume_tokens([258]) == 1, (
        "real matcher should still accept <|tool_call_begin|> -- deep_copy "
        "speculation leaves it untouched (the rollback desync is avoided)"
    )
