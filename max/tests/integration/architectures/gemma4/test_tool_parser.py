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
from typing import Any, cast
from unittest.mock import MagicMock

import pytest
from llguidance import LLMatcher, LLTokenizer
from llguidance._tokenizer import TokenizerWrapper
from max.pipelines.architectures.gemma4.tool_parser import Gemma4ToolParser
from max.pipelines.modeling.types import (
    ParsedToolCall,
    ParsedToolResponse,
    PipelineTokenizer,
)


def _tools(*names: str) -> list[dict[str, Any]]:
    """Build a minimal OpenAI-style tools list from function names."""
    return [{"type": "function", "function": {"name": n}} for n in names]


def _tools_with_schemas(
    schemas: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    """Build tools list with parameter schemas attached."""
    return [
        {
            "type": "function",
            "function": {"name": n, "parameters": s},
        }
        for n, s in schemas.items()
    ]


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


def test_tool_call_without_end_tag_raises_error() -> None:
    """Test that a tool call missing its close tag raises ValueError."""
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
    assert tool_call_id.startswith("call_")
    assert len(tool_call_id) == 29  # "call_" + 24 hex chars


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
    """Test that parse_delta suppresses partial tool-call content."""
    parser = Gemma4ToolParser()

    # First chunk opens the tool call — no name yet, so suppression only.
    result1 = parser.parse_delta("<|tool_call>")
    assert result1 == []

    # Second chunk delivers the header (call:test{). The base class emits
    # the name as soon as the header is parseable.
    result2 = parser.parse_delta("call:test{")
    assert result2 is not None
    assert len(result2) == 1
    assert result2[0].name == "test"
    assert result2[0].id is not None
    assert result2[0].arguments is None

    assert parser._buffer == "<|tool_call>call:test{"


def test_parse_delta_returns_none_outside_tool_section() -> None:
    """parse_delta returns None for plain content with no tool markers."""
    parser = Gemma4ToolParser()

    result = parser.parse_delta("Just some text.")

    # No tool-call markers anywhere — the chunk is plain content. We emit
    # it via a ParsedToolCallDelta(content=...) so the streaming layer
    # routes it to the assistant ``content`` field.
    assert result is not None
    assert len(result) == 1
    assert result[0].content == "Just some text."
    assert result[0].id is None
    assert result[0].name is None


def test_parse_delta_emits_complete_tool_call() -> None:
    """parse_delta emits name early, then arguments on close marker."""
    parser = Gemma4ToolParser()

    assert parser.parse_delta("<|tool_call>") == []

    # Name is emitted as soon as the header (call:NAME{) is parseable.
    name_result = parser.parse_delta('call:get_weather{location:<|"|>Paris')
    assert name_result is not None
    assert len(name_result) == 1
    assert name_result[0].name == "get_weather"
    assert name_result[0].id is not None

    # Arguments are emitted atomically when the close marker arrives.
    args_result = parser.parse_delta('<|"|>}<tool_call|>')
    assert args_result is not None
    assert len(args_result) == 1
    assert args_result[0].index == 0
    assert args_result[0].arguments is not None
    assert json.loads(args_result[0].arguments) == {"location": "Paris"}
    # Name/id not re-emitted on the args delta.
    assert args_result[0].name is None
    assert args_result[0].id is None


def test_parse_delta_emits_content_before_tool_call() -> None:
    """parse_delta emits leading plain content then enters tool mode."""
    parser = Gemma4ToolParser()

    result = parser.parse_delta("preamble<|tool_call>")

    assert result is not None
    assert len(result) == 1
    assert result[0].content == "preamble"
    # Next chunk delivers the header — name is emitted early.
    name_result = parser.parse_delta("call:f{}")
    assert name_result is not None
    assert len(name_result) == 1
    assert name_result[0].name == "f"


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


# ---------------------------------------------------------------------------
# Grammar generation tests
# ---------------------------------------------------------------------------


class _MinimalTokenizer:
    """Byte tokenizer extended with Gemma4 special tokens.

    Maps byte values 0-255 to token IDs 0-255, then assigns dedicated
    IDs to each Gemma4 special token so that grammar generation can
    resolve them via ``convert_tokens_to_ids``.
    """

    _SPECIAL_TOKENS: dict[str, int] = {
        "<|tool_call>": 256,
        "<tool_call|>": 257,
        "<|tool>": 258,
        "<tool|>": 259,
        "<|tool_response>": 260,
        "<tool_response|>": 261,
        '<|"|>': 262,
        "<turn|>": 263,
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


def test_generate_tool_call_grammar_with_tool_names(
    ll_tokenizer: LLTokenizer,
    mock_tokenizer: PipelineTokenizer[Any, Any, Any],
) -> None:
    """Test generating a Lark grammar for constrained decoding with specific tools."""
    grammar = Gemma4ToolParser.generate_tool_call_grammar(
        tools=_tools("get_weather", "search"),
        tokenizer=mock_tokenizer,
    )

    assert isinstance(grammar, str)
    assert len(grammar) > 0
    assert "tool_calls" in grammar

    matcher = LLMatcher(ll_tokenizer, grammar)
    assert matcher is not None


def test_generate_tool_call_grammar_without_tool_names(
    ll_tokenizer: LLTokenizer,
    mock_tokenizer: PipelineTokenizer[Any, Any, Any],
) -> None:
    """Test generating a grammar that accepts any valid identifier."""
    grammar = Gemma4ToolParser.generate_tool_call_grammar(
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
    """Test that special regex characters in tool names are escaped."""
    grammar = Gemma4ToolParser.generate_tool_call_grammar(
        tools=_tools("get_weather.v2", "search+plus", "tool[0]"),
        tokenizer=mock_tokenizer,
    )

    assert isinstance(grammar, str)
    assert len(grammar) > 0

    matcher = LLMatcher(ll_tokenizer, grammar)
    assert matcher is not None


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

    grammar = Gemma4ToolParser.generate_tool_call_grammar(
        tools=_tools("get_weather", "search"),
        response_format_schema=response_format_schema,
        tokenizer=mock_tokenizer,
    )

    assert isinstance(grammar, str)
    assert len(grammar) > 0

    assert "tool_calls" in grammar
    assert "json_response" in grammar
    assert "%json" in grammar

    matcher = LLMatcher(ll_tokenizer, grammar)
    assert matcher is not None


def test_generate_tool_call_grammar_combined_accepts_json_object_type(
    ll_tokenizer: LLTokenizer,
    mock_tokenizer: PipelineTokenizer[Any, Any, Any],
) -> None:
    """Test combined grammar with json_object type (any valid JSON)."""
    response_format_schema = {"type": "object"}

    grammar = Gemma4ToolParser.generate_tool_call_grammar(
        tools=_tools("calculate"),
        response_format_schema=response_format_schema,
        tokenizer=mock_tokenizer,
    )

    assert isinstance(grammar, str)
    assert len(grammar) > 0
    assert "json_response" in grammar

    matcher = LLMatcher(ll_tokenizer, grammar)
    assert matcher is not None


def test_generate_tool_call_grammar_no_schema_is_tool_only(
    ll_tokenizer: LLTokenizer,
    mock_tokenizer: PipelineTokenizer[Any, Any, Any],
) -> None:
    """Test that without response_format_schema, grammar is tool-calls only."""
    grammar = Gemma4ToolParser.generate_tool_call_grammar(
        tools=_tools("get_weather"),
        response_format_schema=None,
        tokenizer=mock_tokenizer,
    )

    assert isinstance(grammar, str)
    assert len(grammar) > 0
    assert "%json" not in grammar
    assert "tool_call" in grammar

    matcher = LLMatcher(ll_tokenizer, grammar)
    assert matcher is not None


def test_generate_tool_call_grammar_accepts_real_tool_call(
    ll_tokenizer: LLTokenizer,
    mock_tokenizer: PipelineTokenizer[Any, Any, Any],
    minimal_tokenizer: _MinimalTokenizer,
) -> None:
    """The grammar must accept a realistic Gemma4 tool call wire string."""
    grammar = Gemma4ToolParser.generate_tool_call_grammar(
        tools=_tools("get_weather"),
        tokenizer=mock_tokenizer,
    )
    matcher = LLMatcher(ll_tokenizer, grammar)

    response = (
        "<|tool_call>call:get_weather"
        '{location:<|"|>San Francisco<|"|>,unit:<|"|>celsius<|"|>}'
        "<tool_call|><turn|>"
    )
    tokens = minimal_tokenizer(response)
    accepted = matcher.validate_tokens(tokens)
    assert accepted == len(tokens), (
        f"Grammar rejected real Gemma4 tool call after {accepted}/{len(tokens)} tokens"
    )


def test_generate_tool_call_grammar_accepts_multiple_calls(
    ll_tokenizer: LLTokenizer,
    mock_tokenizer: PipelineTokenizer[Any, Any, Any],
    minimal_tokenizer: _MinimalTokenizer,
) -> None:
    """The grammar must accept a sequence of Gemma4 tool calls."""
    grammar = Gemma4ToolParser.generate_tool_call_grammar(
        tools=_tools("get_weather", "get_time"),
        tokenizer=mock_tokenizer,
    )
    matcher = LLMatcher(ll_tokenizer, grammar)

    response = (
        '<|tool_call>call:get_weather{location:<|"|>NYC<|"|>}<tool_call|>'
        '<|tool_call>call:get_time{tz:<|"|>UTC<|"|>}<tool_call|>'
        "<turn|>"
    )
    tokens = minimal_tokenizer(response)
    accepted = matcher.validate_tokens(tokens)
    assert accepted == len(tokens)


def test_generate_tool_call_grammar_accepts_nested_object_args(
    ll_tokenizer: LLTokenizer,
    mock_tokenizer: PipelineTokenizer[Any, Any, Any],
    minimal_tokenizer: _MinimalTokenizer,
) -> None:
    """The grammar must accept tool calls with nested ``{...}`` arg values."""
    grammar = Gemma4ToolParser.generate_tool_call_grammar(
        tools=_tools("configure"),
        tokenizer=mock_tokenizer,
    )
    matcher = LLMatcher(ll_tokenizer, grammar)

    response = (
        "<|tool_call>call:configure"
        '{settings:{theme:<|"|>dark<|"|>,size:12},flags:[true,false]}'
        "<tool_call|><turn|>"
    )
    tokens = minimal_tokenizer(response)
    accepted = matcher.validate_tokens(tokens)
    assert accepted == len(tokens), (
        f"Grammar rejected nested-object tool call after {accepted}/{len(tokens)} tokens"
    )


def test_generate_tool_call_grammar_rejects_single_quoted_strings(
    ll_tokenizer: LLTokenizer,
    mock_tokenizer: PipelineTokenizer[Any, Any, Any],
    minimal_tokenizer: _MinimalTokenizer,
) -> None:
    """The grammar must reject tool calls using single quotes instead of <|"|>.

    Regression guard: a permissive earlier grammar allowed any non-'<'
    byte in args, which let the model use Python-style single quotes
    and drift into freeform text.
    """
    grammar = Gemma4ToolParser.generate_tool_call_grammar(
        tools=_tools("get_weather"),
        tokenizer=mock_tokenizer,
    )
    matcher = LLMatcher(ll_tokenizer, grammar)

    bad_response = (
        "<|tool_call>call:get_weather{location: 'Coquitlam, BC'}<tool_call|>"
    )
    tokens = minimal_tokenizer(bad_response)
    accepted = matcher.validate_tokens(tokens)
    assert accepted < len(tokens), (
        "Grammar should reject single-quoted strings but accepted all tokens"
    )


def test_generate_tool_call_grammar_accepts_empty_args(
    ll_tokenizer: LLTokenizer,
    mock_tokenizer: PipelineTokenizer[Any, Any, Any],
    minimal_tokenizer: _MinimalTokenizer,
) -> None:
    """The grammar must accept tool calls with no arguments."""
    grammar = Gemma4ToolParser.generate_tool_call_grammar(
        tools=_tools("get_time"),
        tokenizer=mock_tokenizer,
    )
    matcher = LLMatcher(ll_tokenizer, grammar)

    response = "<|tool_call>call:get_time{}<tool_call|><turn|>"
    tokens = minimal_tokenizer(response)
    accepted = matcher.validate_tokens(tokens)
    assert accepted == len(tokens)


def test_generate_tool_call_grammar_accepts_number_args(
    ll_tokenizer: LLTokenizer,
    mock_tokenizer: PipelineTokenizer[Any, Any, Any],
    minimal_tokenizer: _MinimalTokenizer,
) -> None:
    """The grammar must accept numeric argument values."""
    grammar = Gemma4ToolParser.generate_tool_call_grammar(
        tools=_tools("set_temp"),
        tokenizer=mock_tokenizer,
    )
    matcher = LLMatcher(ll_tokenizer, grammar)

    response = (
        "<|tool_call>call:set_temp{value:22,precision:0.5}<tool_call|><turn|>"
    )
    tokens = minimal_tokenizer(response)
    accepted = matcher.validate_tokens(tokens)
    assert accepted == len(tokens)


def test_generate_tool_call_grammar_accepts_boolean_args(
    ll_tokenizer: LLTokenizer,
    mock_tokenizer: PipelineTokenizer[Any, Any, Any],
    minimal_tokenizer: _MinimalTokenizer,
) -> None:
    """The grammar must accept boolean argument values."""
    grammar = Gemma4ToolParser.generate_tool_call_grammar(
        tools=_tools("configure"),
        tokenizer=mock_tokenizer,
    )
    matcher = LLMatcher(ll_tokenizer, grammar)

    response = (
        "<|tool_call>call:configure{enabled:true,debug:false}<tool_call|>"
        "<turn|>"
    )
    tokens = minimal_tokenizer(response)
    accepted = matcher.validate_tokens(tokens)
    assert accepted == len(tokens)


def test_generate_tool_call_grammar_accepts_array_args(
    ll_tokenizer: LLTokenizer,
    mock_tokenizer: PipelineTokenizer[Any, Any, Any],
    minimal_tokenizer: _MinimalTokenizer,
) -> None:
    """The grammar must accept array argument values."""
    grammar = Gemma4ToolParser.generate_tool_call_grammar(
        tools=_tools("process"),
        tokenizer=mock_tokenizer,
    )
    matcher = LLMatcher(ll_tokenizer, grammar)

    response = (
        "<|tool_call>call:process"
        '{items:[<|"|>a<|"|>,<|"|>b<|"|>],counts:[1,2,3]}'
        "<tool_call|><turn|>"
    )
    tokens = minimal_tokenizer(response)
    accepted = matcher.validate_tokens(tokens)
    assert accepted == len(tokens)


def test_generate_tool_call_grammar_combined_accepts_json_branch(
    ll_tokenizer: LLTokenizer,
    mock_tokenizer: PipelineTokenizer[Any, Any, Any],
    minimal_tokenizer: _MinimalTokenizer,
) -> None:
    """The combined grammar must accept valid JSON objects on the JSON branch."""
    response_format_schema = {
        "type": "object",
        "properties": {"answer": {"type": "string"}},
        "required": ["answer"],
        "additionalProperties": False,
    }
    grammar = Gemma4ToolParser.generate_tool_call_grammar(
        tools=_tools("get_weather"),
        response_format_schema=response_format_schema,
        tokenizer=mock_tokenizer,
    )
    matcher = LLMatcher(ll_tokenizer, grammar)

    json_response = '{"answer":"hello"}'
    tokens = minimal_tokenizer(json_response)
    accepted = matcher.validate_tokens(tokens)
    assert accepted == len(tokens)


def test_generate_tool_call_grammar_combined_rejects_whitespace_json(
    ll_tokenizer: LLTokenizer,
    mock_tokenizer: PipelineTokenizer[Any, Any, Any],
    minimal_tokenizer: _MinimalTokenizer,
) -> None:
    """The combined grammar rejects structural whitespace (whitespace_pattern='')."""
    response_format_schema = {
        "type": "object",
        "properties": {"answer": {"type": "string"}},
        "required": ["answer"],
        "additionalProperties": False,
    }
    grammar = Gemma4ToolParser.generate_tool_call_grammar(
        tools=_tools("get_weather"),
        response_format_schema=response_format_schema,
        tokenizer=mock_tokenizer,
    )
    matcher = LLMatcher(ll_tokenizer, grammar)

    padded = '{\n  "answer": "hello"\n}'
    tokens = minimal_tokenizer(padded)
    accepted = matcher.validate_tokens(tokens)
    assert accepted < len(tokens), (
        "Grammar should reject whitespace-padded JSON"
    )


# ---------------------------------------------------------------------------
# Schema-aware grammar tests
# ---------------------------------------------------------------------------


def test_generate_tool_call_grammar_schema_aware_accepts_correct_types(
    ll_tokenizer: LLTokenizer,
    mock_tokenizer: PipelineTokenizer[Any, Any, Any],
    minimal_tokenizer: _MinimalTokenizer,
) -> None:
    """Schema-aware grammar accepts string values for string-typed properties."""
    tool_schemas = {
        "get_weather": {
            "type": "object",
            "properties": {
                "location": {"type": "string"},
                "unit": {"type": "string"},
            },
        },
    }
    grammar = Gemma4ToolParser.generate_tool_call_grammar(
        tools=_tools_with_schemas(tool_schemas),
        tokenizer=mock_tokenizer,
    )
    matcher = LLMatcher(ll_tokenizer, grammar)

    good = (
        "<|tool_call>call:get_weather"
        '{location:<|"|>San Francisco<|"|>,unit:<|"|>celsius<|"|>}'
        "<tool_call|><turn|>"
    )
    tokens = minimal_tokenizer(good)
    accepted = matcher.validate_tokens(tokens)
    assert accepted == len(tokens), (
        "Schema-aware grammar should accept correct string types"
    )


def test_generate_tool_call_grammar_schema_aware_rejects_wrong_type(
    ll_tokenizer: LLTokenizer,
    mock_tokenizer: PipelineTokenizer[Any, Any, Any],
    minimal_tokenizer: _MinimalTokenizer,
) -> None:
    """Schema-aware grammar rejects numeric values for string-typed properties."""
    tool_schemas = {
        "get_weather": {
            "type": "object",
            "properties": {
                "location": {"type": "string"},
            },
        },
    }
    grammar = Gemma4ToolParser.generate_tool_call_grammar(
        tools=_tools_with_schemas(tool_schemas),
        tokenizer=mock_tokenizer,
    )
    matcher = LLMatcher(ll_tokenizer, grammar)

    bad = "<|tool_call>call:get_weather{location:-83.2}<tool_call|>"
    tokens = minimal_tokenizer(bad)
    accepted = matcher.validate_tokens(tokens)
    assert accepted < len(tokens), (
        "Schema-aware grammar should reject numeric value for string property"
    )


def test_generate_tool_call_grammar_schema_aware_mixed_types(
    ll_tokenizer: LLTokenizer,
    mock_tokenizer: PipelineTokenizer[Any, Any, Any],
    minimal_tokenizer: _MinimalTokenizer,
) -> None:
    """Schema-aware grammar constrains each property to its declared type."""
    tool_schemas = {
        "search": {
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "limit": {"type": "number"},
                "verbose": {"type": "boolean"},
            },
        },
    }
    grammar = Gemma4ToolParser.generate_tool_call_grammar(
        tools=_tools_with_schemas(tool_schemas),
        tokenizer=mock_tokenizer,
    )
    matcher = LLMatcher(ll_tokenizer, grammar)

    good = (
        "<|tool_call>call:search"
        '{query:<|"|>machine learning<|"|>,limit:10,verbose:true}'
        "<tool_call|><turn|>"
    )
    tokens = minimal_tokenizer(good)
    accepted = matcher.validate_tokens(tokens)
    assert accepted == len(tokens)


def test_generate_tool_call_grammar_schema_aware_falls_back_without_properties(
    ll_tokenizer: LLTokenizer,
    mock_tokenizer: PipelineTokenizer[Any, Any, Any],
    minimal_tokenizer: _MinimalTokenizer,
) -> None:
    """Tools without property schemas fall back to generic args."""
    tool_schemas = {
        "get_time": {"type": "object"},
    }
    grammar = Gemma4ToolParser.generate_tool_call_grammar(
        tools=_tools_with_schemas(tool_schemas),
        tokenizer=mock_tokenizer,
    )
    matcher = LLMatcher(ll_tokenizer, grammar)

    response = '<|tool_call>call:get_time{tz:<|"|>UTC<|"|>}<tool_call|><turn|>'
    tokens = minimal_tokenizer(response)
    accepted = matcher.validate_tokens(tokens)
    assert accepted == len(tokens)


def test_generate_tool_call_grammar_schema_aware_multiple_tools(
    ll_tokenizer: LLTokenizer,
    mock_tokenizer: PipelineTokenizer[Any, Any, Any],
    minimal_tokenizer: _MinimalTokenizer,
) -> None:
    """Schema-aware grammar handles multiple tools with different schemas."""
    tool_schemas = {
        "get_weather": {
            "type": "object",
            "properties": {
                "location": {"type": "string"},
            },
        },
        "calculate": {
            "type": "object",
            "properties": {
                "expression": {"type": "string"},
                "precision": {"type": "number"},
            },
        },
    }
    grammar = Gemma4ToolParser.generate_tool_call_grammar(
        tools=_tools_with_schemas(tool_schemas),
        tokenizer=mock_tokenizer,
    )
    matcher = LLMatcher(ll_tokenizer, grammar)

    response = (
        "<|tool_call>call:get_weather"
        '{location:<|"|>Tokyo<|"|>}'
        "<tool_call|>"
        "<|tool_call>call:calculate"
        '{expression:<|"|>2+2<|"|>,precision:4}'
        "<tool_call|><turn|>"
    )
    tokens = minimal_tokenizer(response)
    accepted = matcher.validate_tokens(tokens)
    assert accepted == len(tokens)


# ---------------------------------------------------------------------------
# Null, scientific notation, and integer parsing tests
# ---------------------------------------------------------------------------


def test_null_parameter() -> None:
    """Test parsing tool call with null parameter value."""
    parser = Gemma4ToolParser()

    response = "<|tool_call>call:check{value:null}<tool_call|>"

    result = parser.parse_complete(response)

    assert len(result.tool_calls) == 1
    assert json.loads(result.tool_calls[0].arguments) == {"value": None}


def test_scientific_notation_parameter() -> None:
    """Test parsing tool call with scientific notation number."""
    parser = Gemma4ToolParser()

    response = "<|tool_call>call:calc{x:1.5e10,y:2E3}<tool_call|>"

    result = parser.parse_complete(response)

    assert len(result.tool_calls) == 1
    parsed = json.loads(result.tool_calls[0].arguments)
    assert parsed == {"x": 1.5e10, "y": 2e3}


def test_grammar_accepts_null_value(
    ll_tokenizer: LLTokenizer,
    mock_tokenizer: PipelineTokenizer[Any, Any, Any],
    minimal_tokenizer: _MinimalTokenizer,
) -> None:
    """The grammar must accept null as a bare value."""
    grammar = Gemma4ToolParser.generate_tool_call_grammar(
        tools=_tools("check"),
        tokenizer=mock_tokenizer,
    )
    matcher = LLMatcher(ll_tokenizer, grammar)

    response = "<|tool_call>call:check{value:null}<tool_call|><turn|>"
    tokens = minimal_tokenizer(response)
    accepted = matcher.validate_tokens(tokens)
    assert accepted == len(tokens)


def test_grammar_accepts_scientific_notation(
    ll_tokenizer: LLTokenizer,
    mock_tokenizer: PipelineTokenizer[Any, Any, Any],
    minimal_tokenizer: _MinimalTokenizer,
) -> None:
    """The grammar must accept scientific notation numbers."""
    grammar = Gemma4ToolParser.generate_tool_call_grammar(
        tools=_tools("calc"),
        tokenizer=mock_tokenizer,
    )
    matcher = LLMatcher(ll_tokenizer, grammar)

    for wire in [
        "<|tool_call>call:calc{x:1.5e10}<tool_call|><turn|>",
        "<|tool_call>call:calc{x:2E3}<tool_call|><turn|>",
        "<|tool_call>call:calc{x:-1.0e-5}<tool_call|><turn|>",
    ]:
        matcher = LLMatcher(ll_tokenizer, grammar)
        tokens = minimal_tokenizer(wire)
        accepted = matcher.validate_tokens(tokens)
        assert accepted == len(tokens), f"Rejected: {wire}"


# ---------------------------------------------------------------------------
# Integer vs number type enforcement
# ---------------------------------------------------------------------------


def test_schema_aware_integer_accepts_integer(
    ll_tokenizer: LLTokenizer,
    mock_tokenizer: PipelineTokenizer[Any, Any, Any],
    minimal_tokenizer: _MinimalTokenizer,
) -> None:
    """Schema-aware grammar accepts integer values for integer-typed properties."""
    tool_schemas = {
        "set_count": {
            "type": "object",
            "properties": {"count": {"type": "integer"}},
        },
    }
    grammar = Gemma4ToolParser.generate_tool_call_grammar(
        tools=_tools_with_schemas(tool_schemas),
        tokenizer=mock_tokenizer,
    )
    matcher = LLMatcher(ll_tokenizer, grammar)

    good = "<|tool_call>call:set_count{count:42}<tool_call|><turn|>"
    tokens = minimal_tokenizer(good)
    accepted = matcher.validate_tokens(tokens)
    assert accepted == len(tokens)


def test_schema_aware_integer_rejects_decimal(
    ll_tokenizer: LLTokenizer,
    mock_tokenizer: PipelineTokenizer[Any, Any, Any],
    minimal_tokenizer: _MinimalTokenizer,
) -> None:
    """Schema-aware grammar rejects decimal values for integer-typed properties."""
    tool_schemas = {
        "set_count": {
            "type": "object",
            "properties": {"count": {"type": "integer"}},
        },
    }
    grammar = Gemma4ToolParser.generate_tool_call_grammar(
        tools=_tools_with_schemas(tool_schemas),
        tokenizer=mock_tokenizer,
    )
    matcher = LLMatcher(ll_tokenizer, grammar)

    bad = "<|tool_call>call:set_count{count:1.5}<tool_call|>"
    tokens = minimal_tokenizer(bad)
    accepted = matcher.validate_tokens(tokens)
    assert accepted < len(tokens)


def test_schema_aware_number_accepts_integer_and_float(
    ll_tokenizer: LLTokenizer,
    mock_tokenizer: PipelineTokenizer[Any, Any, Any],
    minimal_tokenizer: _MinimalTokenizer,
) -> None:
    """Schema-aware grammar accepts both integer and float for number-typed properties."""
    tool_schemas = {
        "calc": {
            "type": "object",
            "properties": {"value": {"type": "number"}},
        },
    }
    grammar = Gemma4ToolParser.generate_tool_call_grammar(
        tools=_tools_with_schemas(tool_schemas),
        tokenizer=mock_tokenizer,
    )

    for wire in [
        "<|tool_call>call:calc{value:42}<tool_call|><turn|>",
        "<|tool_call>call:calc{value:3.14}<tool_call|><turn|>",
    ]:
        matcher = LLMatcher(ll_tokenizer, grammar)
        tokens = minimal_tokenizer(wire)
        accepted = matcher.validate_tokens(tokens)
        assert accepted == len(tokens), f"Rejected: {wire}"


# ---------------------------------------------------------------------------
# Enum enforcement tests
# ---------------------------------------------------------------------------


def test_schema_aware_enum_accepts_valid_value(
    ll_tokenizer: LLTokenizer,
    mock_tokenizer: PipelineTokenizer[Any, Any, Any],
    minimal_tokenizer: _MinimalTokenizer,
) -> None:
    """Schema-aware grammar accepts a value that is in the enum list."""
    tool_schemas = {
        "get_weather": {
            "type": "object",
            "properties": {
                "unit": {
                    "type": "string",
                    "enum": ["celsius", "fahrenheit"],
                },
            },
        },
    }
    grammar = Gemma4ToolParser.generate_tool_call_grammar(
        tools=_tools_with_schemas(tool_schemas),
        tokenizer=mock_tokenizer,
    )
    matcher = LLMatcher(ll_tokenizer, grammar)

    good = (
        '<|tool_call>call:get_weather{unit:<|"|>celsius<|"|>}'
        "<tool_call|><turn|>"
    )
    tokens = minimal_tokenizer(good)
    accepted = matcher.validate_tokens(tokens)
    assert accepted == len(tokens)


def test_schema_aware_enum_rejects_invalid_value(
    ll_tokenizer: LLTokenizer,
    mock_tokenizer: PipelineTokenizer[Any, Any, Any],
    minimal_tokenizer: _MinimalTokenizer,
) -> None:
    """Schema-aware grammar rejects a value not in the enum list."""
    tool_schemas = {
        "get_weather": {
            "type": "object",
            "properties": {
                "unit": {
                    "type": "string",
                    "enum": ["celsius", "fahrenheit"],
                },
            },
        },
    }
    grammar = Gemma4ToolParser.generate_tool_call_grammar(
        tools=_tools_with_schemas(tool_schemas),
        tokenizer=mock_tokenizer,
    )
    matcher = LLMatcher(ll_tokenizer, grammar)

    bad = '<|tool_call>call:get_weather{unit:<|"|>kelvin<|"|>}<tool_call|>'
    tokens = minimal_tokenizer(bad)
    accepted = matcher.validate_tokens(tokens)
    assert accepted < len(tokens)


def test_schema_aware_enum_with_null(
    ll_tokenizer: LLTokenizer,
    mock_tokenizer: PipelineTokenizer[Any, Any, Any],
    minimal_tokenizer: _MinimalTokenizer,
) -> None:
    """Schema-aware grammar accepts null when it is in the enum list."""
    tool_schemas = {
        "check": {
            "type": "object",
            "properties": {
                "status": {"enum": ["active", "inactive", None]},
            },
        },
    }
    grammar = Gemma4ToolParser.generate_tool_call_grammar(
        tools=_tools_with_schemas(tool_schemas),
        tokenizer=mock_tokenizer,
    )
    matcher = LLMatcher(ll_tokenizer, grammar)

    good = "<|tool_call>call:check{status:null}<tool_call|><turn|>"
    tokens = minimal_tokenizer(good)
    accepted = matcher.validate_tokens(tokens)
    assert accepted == len(tokens)


# ---------------------------------------------------------------------------
# Nested object and array type enforcement
# ---------------------------------------------------------------------------


def test_schema_aware_nested_object_typed(
    ll_tokenizer: LLTokenizer,
    mock_tokenizer: PipelineTokenizer[Any, Any, Any],
    minimal_tokenizer: _MinimalTokenizer,
) -> None:
    """Schema-aware grammar enforces types inside nested objects."""
    tool_schemas = {
        "send": {
            "type": "object",
            "properties": {
                "address": {
                    "type": "object",
                    "properties": {
                        "city": {"type": "string"},
                        "zip": {"type": "integer"},
                    },
                },
            },
        },
    }
    grammar = Gemma4ToolParser.generate_tool_call_grammar(
        tools=_tools_with_schemas(tool_schemas),
        tokenizer=mock_tokenizer,
    )
    matcher = LLMatcher(ll_tokenizer, grammar)

    good = (
        "<|tool_call>call:send"
        '{address:{city:<|"|>NYC<|"|>,zip:10001}}'
        "<tool_call|><turn|>"
    )
    tokens = minimal_tokenizer(good)
    accepted = matcher.validate_tokens(tokens)
    assert accepted == len(tokens)


def test_schema_aware_nested_array_typed_items(
    ll_tokenizer: LLTokenizer,
    mock_tokenizer: PipelineTokenizer[Any, Any, Any],
    minimal_tokenizer: _MinimalTokenizer,
) -> None:
    """Schema-aware grammar enforces item types inside arrays."""
    tool_schemas = {
        "process": {
            "type": "object",
            "properties": {
                "tags": {
                    "type": "array",
                    "items": {"type": "string"},
                },
            },
        },
    }
    grammar = Gemma4ToolParser.generate_tool_call_grammar(
        tools=_tools_with_schemas(tool_schemas),
        tokenizer=mock_tokenizer,
    )
    matcher = LLMatcher(ll_tokenizer, grammar)

    good = (
        "<|tool_call>call:process"
        '{tags:[<|"|>a<|"|>,<|"|>b<|"|>]}'
        "<tool_call|><turn|>"
    )
    tokens = minimal_tokenizer(good)
    accepted = matcher.validate_tokens(tokens)
    assert accepted == len(tokens)


def test_schema_aware_nested_array_rejects_wrong_item_type(
    ll_tokenizer: LLTokenizer,
    mock_tokenizer: PipelineTokenizer[Any, Any, Any],
    minimal_tokenizer: _MinimalTokenizer,
) -> None:
    """Schema-aware grammar rejects wrong item types inside arrays."""
    tool_schemas = {
        "process": {
            "type": "object",
            "properties": {
                "tags": {
                    "type": "array",
                    "items": {"type": "string"},
                },
            },
        },
    }
    grammar = Gemma4ToolParser.generate_tool_call_grammar(
        tools=_tools_with_schemas(tool_schemas),
        tokenizer=mock_tokenizer,
    )
    matcher = LLMatcher(ll_tokenizer, grammar)

    bad = "<|tool_call>call:process{tags:[42,true]}<tool_call|>"
    tokens = minimal_tokenizer(bad)
    accepted = matcher.validate_tokens(tokens)
    assert accepted < len(tokens)


# ---------------------------------------------------------------------------
# Required / optional / ordering / duplicate enforcement
# ---------------------------------------------------------------------------


def test_schema_aware_required_rejects_missing(
    ll_tokenizer: LLTokenizer,
    mock_tokenizer: PipelineTokenizer[Any, Any, Any],
    minimal_tokenizer: _MinimalTokenizer,
) -> None:
    """Schema-aware grammar rejects when a required property is missing."""
    tool_schemas = {
        "get_weather": {
            "type": "object",
            "properties": {
                "location": {"type": "string"},
                "unit": {"type": "string"},
            },
            "required": ["location"],
        },
    }
    grammar = Gemma4ToolParser.generate_tool_call_grammar(
        tools=_tools_with_schemas(tool_schemas),
        tokenizer=mock_tokenizer,
    )
    matcher = LLMatcher(ll_tokenizer, grammar)

    bad = '<|tool_call>call:get_weather{unit:<|"|>celsius<|"|>}<tool_call|>'
    tokens = minimal_tokenizer(bad)
    accepted = matcher.validate_tokens(tokens)
    assert accepted < len(tokens)


def test_schema_aware_required_accepts_all_present(
    ll_tokenizer: LLTokenizer,
    mock_tokenizer: PipelineTokenizer[Any, Any, Any],
    minimal_tokenizer: _MinimalTokenizer,
) -> None:
    """Schema-aware grammar accepts when all required properties are present."""
    tool_schemas = {
        "get_weather": {
            "type": "object",
            "properties": {
                "location": {"type": "string"},
                "unit": {"type": "string"},
            },
            "required": ["location"],
        },
    }
    grammar = Gemma4ToolParser.generate_tool_call_grammar(
        tools=_tools_with_schemas(tool_schemas),
        tokenizer=mock_tokenizer,
    )
    matcher = LLMatcher(ll_tokenizer, grammar)

    good = (
        "<|tool_call>call:get_weather"
        '{location:<|"|>Tokyo<|"|>,unit:<|"|>celsius<|"|>}'
        "<tool_call|><turn|>"
    )
    tokens = minimal_tokenizer(good)
    accepted = matcher.validate_tokens(tokens)
    assert accepted == len(tokens)


def test_schema_aware_rejects_wrong_order(
    ll_tokenizer: LLTokenizer,
    mock_tokenizer: PipelineTokenizer[Any, Any, Any],
    minimal_tokenizer: _MinimalTokenizer,
) -> None:
    """Schema-aware grammar rejects properties in non-schema order."""
    tool_schemas = {
        "get_weather": {
            "type": "object",
            "properties": {
                "location": {"type": "string"},
                "unit": {"type": "string"},
            },
        },
    }
    grammar = Gemma4ToolParser.generate_tool_call_grammar(
        tools=_tools_with_schemas(tool_schemas),
        tokenizer=mock_tokenizer,
    )
    matcher = LLMatcher(ll_tokenizer, grammar)

    bad = (
        "<|tool_call>call:get_weather"
        '{unit:<|"|>celsius<|"|>,location:<|"|>Tokyo<|"|>}'
        "<tool_call|>"
    )
    tokens = minimal_tokenizer(bad)
    accepted = matcher.validate_tokens(tokens)
    assert accepted < len(tokens)


def test_schema_aware_rejects_duplicate_property(
    ll_tokenizer: LLTokenizer,
    mock_tokenizer: PipelineTokenizer[Any, Any, Any],
    minimal_tokenizer: _MinimalTokenizer,
) -> None:
    """Schema-aware grammar rejects duplicate properties."""
    tool_schemas = {
        "get_weather": {
            "type": "object",
            "properties": {
                "location": {"type": "string"},
            },
        },
    }
    grammar = Gemma4ToolParser.generate_tool_call_grammar(
        tools=_tools_with_schemas(tool_schemas),
        tokenizer=mock_tokenizer,
    )
    matcher = LLMatcher(ll_tokenizer, grammar)

    bad = (
        "<|tool_call>call:get_weather"
        '{location:<|"|>NYC<|"|>,location:<|"|>LA<|"|>}'
        "<tool_call|>"
    )
    tokens = minimal_tokenizer(bad)
    accepted = matcher.validate_tokens(tokens)
    assert accepted < len(tokens)


def test_schema_aware_optional_can_be_skipped(
    ll_tokenizer: LLTokenizer,
    mock_tokenizer: PipelineTokenizer[Any, Any, Any],
    minimal_tokenizer: _MinimalTokenizer,
) -> None:
    """Schema-aware grammar allows skipping optional properties."""
    tool_schemas = {
        "get_weather": {
            "type": "object",
            "properties": {
                "location": {"type": "string"},
                "unit": {"type": "string"},
            },
            "required": ["location"],
        },
    }
    grammar = Gemma4ToolParser.generate_tool_call_grammar(
        tools=_tools_with_schemas(tool_schemas),
        tokenizer=mock_tokenizer,
    )
    matcher = LLMatcher(ll_tokenizer, grammar)

    good = (
        "<|tool_call>call:get_weather"
        '{location:<|"|>Tokyo<|"|>}'
        "<tool_call|><turn|>"
    )
    tokens = minimal_tokenizer(good)
    accepted = matcher.validate_tokens(tokens)
    assert accepted == len(tokens)


def test_schema_aware_all_optional_empty_args(
    ll_tokenizer: LLTokenizer,
    mock_tokenizer: PipelineTokenizer[Any, Any, Any],
    minimal_tokenizer: _MinimalTokenizer,
) -> None:
    """Schema-aware grammar accepts empty args when all properties are optional."""
    tool_schemas = {
        "get_weather": {
            "type": "object",
            "properties": {
                "location": {"type": "string"},
                "unit": {"type": "string"},
            },
        },
    }
    grammar = Gemma4ToolParser.generate_tool_call_grammar(
        tools=_tools_with_schemas(tool_schemas),
        tokenizer=mock_tokenizer,
    )
    matcher = LLMatcher(ll_tokenizer, grammar)

    good = "<|tool_call>call:get_weather{}<tool_call|><turn|>"
    tokens = minimal_tokenizer(good)
    accepted = matcher.validate_tokens(tokens)
    assert accepted == len(tokens)


def test_schema_aware_required_only_without_optional(
    ll_tokenizer: LLTokenizer,
    mock_tokenizer: PipelineTokenizer[Any, Any, Any],
    minimal_tokenizer: _MinimalTokenizer,
) -> None:
    """Schema-aware grammar accepts just the required property without optional."""
    tool_schemas = {
        "get_weather": {
            "type": "object",
            "properties": {
                "location": {"type": "string"},
                "unit": {"type": "string"},
            },
            "required": ["location"],
        },
    }
    grammar = Gemma4ToolParser.generate_tool_call_grammar(
        tools=_tools_with_schemas(tool_schemas),
        tokenizer=mock_tokenizer,
    )
    matcher = LLMatcher(ll_tokenizer, grammar)

    good = (
        "<|tool_call>call:get_weather"
        '{location:<|"|>Paris<|"|>}'
        "<tool_call|><turn|>"
    )
    tokens = minimal_tokenizer(good)
    accepted = matcher.validate_tokens(tokens)
    assert accepted == len(tokens)


def test_schema_aware_deep_nesting_fallback(
    ll_tokenizer: LLTokenizer,
    mock_tokenizer: PipelineTokenizer[Any, Any, Any],
    minimal_tokenizer: _MinimalTokenizer,
) -> None:
    """Nesting beyond max_depth falls back to generic value rule."""
    # Build a schema nested 7 levels deep (exceeds max_depth=5).
    schema: dict[str, Any] = {"type": "string"}
    for i in range(7):
        schema = {
            "type": "object",
            "properties": {f"level{i}": schema},
            "required": [f"level{i}"],
        }
    tool_schemas = {"deep_fn": schema}
    grammar = Gemma4ToolParser.generate_tool_call_grammar(
        tools=_tools_with_schemas(tool_schemas),
        tokenizer=mock_tokenizer,
    )
    matcher = LLMatcher(ll_tokenizer, grammar)

    # At depth > 5 the grammar falls back to generic `value`, so any
    # well-formed JSON value should be accepted in the innermost slot.
    # Build a valid nested call with strings all the way down.
    inner = '<|"|>hi<|"|>'
    for i in range(7):
        inner = "{" + f"level{i}:" + inner + "}"
    wire = f"<|tool_call>call:deep_fn{inner}<tool_call|><turn|>"
    tokens = minimal_tokenizer(wire)
    accepted = matcher.validate_tokens(tokens)
    assert accepted == len(tokens)


def test_schema_aware_many_properties(
    ll_tokenizer: LLTokenizer,
    mock_tokenizer: PipelineTokenizer[Any, Any, Any],
    minimal_tokenizer: _MinimalTokenizer,
) -> None:
    """Fixed-order suffix rules work for a large number of properties."""
    n = 20
    props = {f"p{i}": {"type": "string"} for i in range(n)}
    tool_schemas = {
        "big_fn": {
            "type": "object",
            "properties": props,
            "required": [f"p{i}" for i in range(n)],
        },
    }
    grammar = Gemma4ToolParser.generate_tool_call_grammar(
        tools=_tools_with_schemas(tool_schemas),
        tokenizer=mock_tokenizer,
    )
    matcher = LLMatcher(ll_tokenizer, grammar)

    # All 20 properties in schema order
    args = ",".join(f'p{i}:<|"|>v{i}<|"|>' for i in range(n))
    wire = f"<|tool_call>call:big_fn{{{args}}}<tool_call|><turn|>"
    tokens = minimal_tokenizer(wire)
    accepted = matcher.validate_tokens(tokens)
    assert accepted == len(tokens)


# ---------------------------------------------------------------------------
# Fail-closed fixes: newlines, hyphenated keys, enum objects, integer sci-not
# ---------------------------------------------------------------------------


def test_string_content_accepts_newlines(
    ll_tokenizer: LLTokenizer,
    mock_tokenizer: PipelineTokenizer[Any, Any, Any],
    minimal_tokenizer: _MinimalTokenizer,
) -> None:
    """STRING_CONTENT must match newline characters."""
    grammar = Gemma4ToolParser.generate_tool_call_grammar(
        tools=_tools("note"),
        tokenizer=mock_tokenizer,
    )
    matcher = LLMatcher(ll_tokenizer, grammar)

    sd = '<|"|>'
    wire = f"<|tool_call>call:note{{text:{sd}line1\nline2\nline3{sd}}}<tool_call|><turn|>"
    tokens = minimal_tokenizer(wire)
    accepted = matcher.validate_tokens(tokens)
    assert accepted == len(tokens)


def test_key_terminal_accepts_hyphens_and_dots(
    ll_tokenizer: LLTokenizer,
    mock_tokenizer: PipelineTokenizer[Any, Any, Any],
    minimal_tokenizer: _MinimalTokenizer,
) -> None:
    """KEY terminal must allow hyphens and dots in property names."""
    grammar = Gemma4ToolParser.generate_tool_call_grammar(
        tools=_tools("req"),
        tokenizer=mock_tokenizer,
    )
    matcher = LLMatcher(ll_tokenizer, grammar)

    for wire in [
        '<|tool_call>call:req{Content-Type:<|"|>text/html<|"|>}<tool_call|><turn|>',
        '<|tool_call>call:req{x.custom:<|"|>val<|"|>}<tool_call|><turn|>',
        '<|tool_call>call:req{a-b.c_d:<|"|>ok<|"|>}<tool_call|><turn|>',
    ]:
        matcher = LLMatcher(ll_tokenizer, grammar)
        tokens = minimal_tokenizer(wire)
        accepted = matcher.validate_tokens(tokens)
        assert accepted == len(tokens), f"Rejected: {wire}"


def test_enum_with_object_value_accepted(
    ll_tokenizer: LLTokenizer,
    mock_tokenizer: PipelineTokenizer[Any, Any, Any],
    minimal_tokenizer: _MinimalTokenizer,
) -> None:
    """Enum containing object/array values must not silently drop them."""
    tool_schemas = {
        "apply": {
            "type": "object",
            "properties": {
                "config": {
                    "enum": [
                        "default",
                        {"mode": "advanced", "level": 3},
                    ],
                },
            },
            "required": ["config"],
        },
    }
    grammar = Gemma4ToolParser.generate_tool_call_grammar(
        tools=_tools_with_schemas(tool_schemas),
        tokenizer=mock_tokenizer,
    )

    # String alternative still works
    matcher = LLMatcher(ll_tokenizer, grammar)
    wire_str = (
        '<|tool_call>call:apply{config:<|"|>default<|"|>}<tool_call|><turn|>'
    )
    tokens = minimal_tokenizer(wire_str)
    accepted = matcher.validate_tokens(tokens)
    assert accepted == len(tokens), "String enum value rejected"

    # Object alternative must also be accepted (via object_val fallback)
    matcher = LLMatcher(ll_tokenizer, grammar)
    sd = '<|"|>'
    wire_obj = f"<|tool_call>call:apply{{config:{{mode:{sd}advanced{sd},level:3}}}}<tool_call|><turn|>"
    tokens = minimal_tokenizer(wire_obj)
    accepted = matcher.validate_tokens(tokens)
    assert accepted == len(tokens), "Object enum value rejected"


def test_integer_terminal_accepts_scientific_notation(
    ll_tokenizer: LLTokenizer,
    mock_tokenizer: PipelineTokenizer[Any, Any, Any],
    minimal_tokenizer: _MinimalTokenizer,
) -> None:
    """INTEGER terminal must accept scientific notation like 1e9."""
    tool_schemas = {
        "alloc": {
            "type": "object",
            "properties": {
                "count": {"type": "integer"},
            },
            "required": ["count"],
        },
    }
    grammar = Gemma4ToolParser.generate_tool_call_grammar(
        tools=_tools_with_schemas(tool_schemas),
        tokenizer=mock_tokenizer,
    )

    for wire in [
        "<|tool_call>call:alloc{count:1e9}<tool_call|><turn|>",
        "<|tool_call>call:alloc{count:6E23}<tool_call|><turn|>",
        "<|tool_call>call:alloc{count:1e+5}<tool_call|><turn|>",
        "<|tool_call>call:alloc{count:42}<tool_call|><turn|>",
    ]:
        matcher = LLMatcher(ll_tokenizer, grammar)
        tokens = minimal_tokenizer(wire)
        accepted = matcher.validate_tokens(tokens)
        assert accepted == len(tokens), f"Rejected: {wire}"


# ---------------------------------------------------------------------------
# Union type lists and null type handling
# ---------------------------------------------------------------------------


def test_nullable_type_list_accepts_string(
    ll_tokenizer: LLTokenizer,
    mock_tokenizer: PipelineTokenizer[Any, Any, Any],
    minimal_tokenizer: _MinimalTokenizer,
) -> None:
    """type: ["string", "null"] must accept a string value."""
    tool_schemas = {
        "greet": {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "nickname": {"type": ["string", "null"]},
            },
            "required": ["name", "nickname"],
        },
    }
    grammar = Gemma4ToolParser.generate_tool_call_grammar(
        tools=_tools_with_schemas(tool_schemas),
        tokenizer=mock_tokenizer,
    )
    matcher = LLMatcher(ll_tokenizer, grammar)
    sd = '<|"|>'
    wire = f"<|tool_call>call:greet{{name:{sd}Alice{sd},nickname:{sd}Ali{sd}}}<tool_call|><turn|>"
    tokens = minimal_tokenizer(wire)
    accepted = matcher.validate_tokens(tokens)
    assert accepted == len(tokens)


def test_nullable_type_list_accepts_null(
    ll_tokenizer: LLTokenizer,
    mock_tokenizer: PipelineTokenizer[Any, Any, Any],
    minimal_tokenizer: _MinimalTokenizer,
) -> None:
    """type: ["string", "null"] must accept a null value."""
    tool_schemas = {
        "greet": {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "nickname": {"type": ["string", "null"]},
            },
            "required": ["name", "nickname"],
        },
    }
    grammar = Gemma4ToolParser.generate_tool_call_grammar(
        tools=_tools_with_schemas(tool_schemas),
        tokenizer=mock_tokenizer,
    )
    matcher = LLMatcher(ll_tokenizer, grammar)
    sd = '<|"|>'
    wire = f"<|tool_call>call:greet{{name:{sd}Alice{sd},nickname:null}}<tool_call|><turn|>"
    tokens = minimal_tokenizer(wire)
    accepted = matcher.validate_tokens(tokens)
    assert accepted == len(tokens)


def test_nullable_type_list_rejects_wrong_type(
    ll_tokenizer: LLTokenizer,
    mock_tokenizer: PipelineTokenizer[Any, Any, Any],
    minimal_tokenizer: _MinimalTokenizer,
) -> None:
    """type: ["string", "null"] must reject an integer value."""
    tool_schemas = {
        "greet": {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "nickname": {"type": ["string", "null"]},
            },
            "required": ["name", "nickname"],
        },
    }
    grammar = Gemma4ToolParser.generate_tool_call_grammar(
        tools=_tools_with_schemas(tool_schemas),
        tokenizer=mock_tokenizer,
    )
    matcher = LLMatcher(ll_tokenizer, grammar)
    sd = '<|"|>'
    wire = f"<|tool_call>call:greet{{name:{sd}Alice{sd},nickname:42}}<tool_call|><turn|>"
    tokens = minimal_tokenizer(wire)
    accepted = matcher.validate_tokens(tokens)
    assert accepted < len(tokens), (
        "Should reject integer for ['string', 'null']"
    )


def test_null_type_standalone_accepts_null(
    ll_tokenizer: LLTokenizer,
    mock_tokenizer: PipelineTokenizer[Any, Any, Any],
    minimal_tokenizer: _MinimalTokenizer,
) -> None:
    """type: "null" must accept null and reject other values."""
    tool_schemas = {
        "cancel": {
            "type": "object",
            "properties": {
                "action": {"type": "string"},
                "reason": {"type": "null"},
            },
            "required": ["action", "reason"],
        },
    }
    grammar = Gemma4ToolParser.generate_tool_call_grammar(
        tools=_tools_with_schemas(tool_schemas),
        tokenizer=mock_tokenizer,
    )
    sd = '<|"|>'

    # null value accepted
    matcher = LLMatcher(ll_tokenizer, grammar)
    wire = f"<|tool_call>call:cancel{{action:{sd}cancel{sd},reason:null}}<tool_call|><turn|>"
    tokens = minimal_tokenizer(wire)
    accepted = matcher.validate_tokens(tokens)
    assert accepted == len(tokens), "null value rejected for type 'null'"

    # string value rejected
    matcher = LLMatcher(ll_tokenizer, grammar)
    wire_str = f"<|tool_call>call:cancel{{action:{sd}cancel{sd},reason:{sd}none{sd}}}<tool_call|><turn|>"
    tokens = minimal_tokenizer(wire_str)
    accepted = matcher.validate_tokens(tokens)
    assert accepted < len(tokens), "string should be rejected for type 'null'"
