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

"""Tests for the GLM-4.5+ tool-call parser and constrained-decoding grammar."""

from __future__ import annotations

import json
from typing import Any, cast
from unittest.mock import MagicMock

import pytest
from llguidance import LLMatcher, LLTokenizer
from llguidance._tokenizer import TokenizerWrapper
from max.pipelines.architectures.glm5_1.tool_parser import GlmToolParser
from max.pipelines.modeling.types import (
    ParsedToolResponse,
    PipelineTokenizer,
)

# ---------------------------------------------------------------------------
# Complete parsing
# ---------------------------------------------------------------------------


def test_single_tool_call() -> None:
    parser = GlmToolParser()
    response = (
        "<tool_call>get_weather"
        "<arg_key>location</arg_key><arg_value>San Francisco</arg_value>"
        "<arg_key>unit</arg_key><arg_value>celsius</arg_value>"
        "</tool_call>"
    )
    result = parser.parse_complete(response)
    assert isinstance(result, ParsedToolResponse)
    assert result.content is None
    assert len(result.tool_calls) == 1
    tc = result.tool_calls[0]
    assert tc.name == "get_weather"
    assert tc.id.startswith("call_")
    # Bare strings stay strings.
    assert json.loads(tc.arguments) == {
        "location": "San Francisco",
        "unit": "celsius",
    }


def test_non_string_values_decode_as_json() -> None:
    parser = GlmToolParser()
    response = (
        "<tool_call>calc"
        "<arg_key>n</arg_key><arg_value>42</arg_value>"
        "<arg_key>flag</arg_key><arg_value>true</arg_value>"
        "<arg_key>items</arg_key><arg_value>[1, 2, 3]</arg_value>"
        "</tool_call>"
    )
    result = parser.parse_complete(response)
    assert json.loads(result.tool_calls[0].arguments) == {
        "n": 42,
        "flag": True,
        "items": [1, 2, 3],
    }


def test_content_before_tool_call_is_preserved() -> None:
    parser = GlmToolParser()
    response = "Let me check.<tool_call>ping</tool_call>"
    result = parser.parse_complete(response)
    assert result.content == "Let me check."
    assert len(result.tool_calls) == 1
    assert result.tool_calls[0].name == "ping"
    assert json.loads(result.tool_calls[0].arguments) == {}


def test_multiple_tool_calls() -> None:
    parser = GlmToolParser()
    response = (
        "<tool_call>a<arg_key>x</arg_key><arg_value>1</arg_value></tool_call>\n"
        "<tool_call>b<arg_key>y</arg_key><arg_value>2</arg_value></tool_call>"
    )
    result = parser.parse_complete(response)
    assert [tc.name for tc in result.tool_calls] == ["a", "b"]
    assert json.loads(result.tool_calls[0].arguments) == {"x": 1}
    assert json.loads(result.tool_calls[1].arguments) == {"y": 2}


def test_plain_text_has_no_tool_calls() -> None:
    parser = GlmToolParser()
    result = parser.parse_complete("just a normal answer")
    assert result.content == "just a normal answer"
    assert result.tool_calls == []


# ---------------------------------------------------------------------------
# Streaming
# ---------------------------------------------------------------------------


def _collect_stream(
    parser: GlmToolParser, chunks: list[str]
) -> tuple[str, str, str]:
    content = ""
    name = ""
    args = ""
    for chunk in chunks:
        deltas = parser.parse_delta(chunk)
        if not deltas:
            continue
        for d in deltas:
            if d.content:
                content += d.content
            if d.name:
                name = d.name
            if d.arguments:
                args += d.arguments
    return content, name, args


def test_streaming_reassembles_call() -> None:
    parser = GlmToolParser()
    # Split mid-marker to exercise partial-token holdback.
    chunks = [
        "hi <tool_",
        "call>get_weather<arg_key>loc",
        "ation</arg_key><arg_value>Paris",
        "</arg_value></tool_call>",
    ]
    content, name, args = _collect_stream(parser, chunks)
    assert content.strip() == "hi"
    assert name == "get_weather"
    assert json.loads(args) == {"location": "Paris"}


# ---------------------------------------------------------------------------
# Constrained-decoding grammar
# ---------------------------------------------------------------------------

_GLM_SPECIAL_TOKENS: dict[str, int] = {
    "<tool_call>": 256,
    "</tool_call>": 257,
    "<arg_key>": 258,
    "</arg_key>": 259,
    "<arg_value>": 260,
    "</arg_value>": 261,
    # Turn-ender tokens the grammar requires after the calls.
    "<|observation|>": 262,
    "<|user|>": 263,
    "<|endoftext|>": 264,
}

# A GLM tool-calling turn ends with a turn-ender token; append it to wire
# strings fed to the grammar matcher.
_TURN_END = "<|observation|>"


class _MinimalTokenizer:
    """Byte tokenizer extended with the GLM tool-call special tokens."""

    _N_VOCAB = 265
    eos_token_id = 0
    bos_token_id: int | None = None
    unk_token_id: int | None = None

    def __init__(self) -> None:
        self.tokens: list[bytes] = [bytes([i]) for i in range(256)]
        self.tokens.extend(t.encode("utf-8") for t in _GLM_SPECIAL_TOKENS)

    def convert_tokens_to_ids(self, token: str) -> int | None:
        return _GLM_SPECIAL_TOKENS.get(token)

    def __call__(self, s: bytes | str) -> list[int]:
        if isinstance(s, str):
            s = s.encode("utf-8")
        result: list[int] = []
        i = 0
        while i < len(s):
            for text, tid in sorted(
                _GLM_SPECIAL_TOKENS.items(), key=lambda x: -len(x[0])
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
    return _MinimalTokenizer()


@pytest.fixture(scope="module")
def mock_tokenizer(
    minimal_tokenizer: _MinimalTokenizer,
) -> PipelineTokenizer[Any, Any, Any]:
    stub = cast(PipelineTokenizer[Any, Any, Any], MagicMock())
    stub.delegate = minimal_tokenizer  # type: ignore[attr-defined]
    return stub


@pytest.fixture(scope="module")
def ll_tokenizer(minimal_tokenizer: _MinimalTokenizer) -> LLTokenizer:
    wrapper = TokenizerWrapper(minimal_tokenizer)
    return LLTokenizer(wrapper, n_vocab=_MinimalTokenizer._N_VOCAB)


def _tools(*names: str) -> list[dict[str, Any]]:
    return [
        {
            "type": "function",
            "function": {
                "name": n,
                "parameters": {
                    "type": "object",
                    "properties": {"location": {"type": "string"}},
                },
            },
        }
        for n in names
    ]


def test_grammar_accepts_real_tool_call(
    ll_tokenizer: LLTokenizer,
    mock_tokenizer: PipelineTokenizer[Any, Any, Any],
    minimal_tokenizer: _MinimalTokenizer,
) -> None:
    grammar = GlmToolParser.generate_tool_call_grammar(
        tools=_tools("get_weather"),
        tokenizer=mock_tokenizer,
    )
    matcher = LLMatcher(ll_tokenizer, grammar)
    response = (
        "<tool_call>get_weather"
        "<arg_key>location</arg_key><arg_value>San Francisco</arg_value>"
        "</tool_call>" + _TURN_END
    )
    tokens = minimal_tokenizer(response)
    accepted = matcher.validate_tokens(tokens)
    assert accepted == len(tokens), (
        f"Grammar rejected GLM tool call after {accepted}/{len(tokens)} tokens"
    )


def test_grammar_without_tool_names(
    ll_tokenizer: LLTokenizer,
    mock_tokenizer: PipelineTokenizer[Any, Any, Any],
    minimal_tokenizer: _MinimalTokenizer,
) -> None:
    grammar = GlmToolParser.generate_tool_call_grammar(tokenizer=mock_tokenizer)
    matcher = LLMatcher(ll_tokenizer, grammar)
    response = (
        "<tool_call>anything<arg_key>k</arg_key><arg_value>v</arg_value>"
        "</tool_call>" + _TURN_END
    )
    tokens = minimal_tokenizer(response)
    assert matcher.validate_tokens(tokens) == len(tokens)


def test_grammar_accepts_multiple_calls(
    ll_tokenizer: LLTokenizer,
    mock_tokenizer: PipelineTokenizer[Any, Any, Any],
    minimal_tokenizer: _MinimalTokenizer,
) -> None:
    grammar = GlmToolParser.generate_tool_call_grammar(
        tools=_tools("get_weather", "get_time"),
        tokenizer=mock_tokenizer,
    )
    matcher = LLMatcher(ll_tokenizer, grammar)
    response = (
        "<tool_call>get_weather<arg_key>location</arg_key>"
        "<arg_value>NYC</arg_value></tool_call>"
        "<tool_call>get_time</tool_call>" + _TURN_END
    )
    tokens = minimal_tokenizer(response)
    assert matcher.validate_tokens(tokens) == len(tokens)


def test_grammar_with_response_format_schema(
    ll_tokenizer: LLTokenizer,
    mock_tokenizer: PipelineTokenizer[Any, Any, Any],
    minimal_tokenizer: _MinimalTokenizer,
) -> None:
    schema = {
        "type": "object",
        "properties": {"answer": {"type": "string"}},
        "required": ["answer"],
    }
    grammar = GlmToolParser.generate_tool_call_grammar(
        response_format_schema=schema,
        tools=_tools("get_weather"),
        tokenizer=mock_tokenizer,
    )
    matcher = LLMatcher(ll_tokenizer, grammar)
    # The JSON branch must accept a schema-conformant object.
    tokens = minimal_tokenizer('{"answer":"hi"}')
    assert matcher.validate_tokens(tokens) == len(tokens)


def test_grammar_requires_tokenizer() -> None:
    with pytest.raises(ValueError, match="tokenizer is required"):
        GlmToolParser.generate_tool_call_grammar(tools=_tools("x"))


# ---------------------------------------------------------------------------
# Schema-aware argument constraints
# ---------------------------------------------------------------------------

_SCHEMA_TOOL = [
    {
        "type": "function",
        "function": {
            "name": "book",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {"type": "string"},
                    "nights": {"type": "integer"},
                    "klass": {"type": "string", "enum": ["economy", "first"]},
                },
                "required": ["city", "nights"],
                "additionalProperties": False,
            },
        },
    }
]


def _validate(
    grammar: str,
    ll_tokenizer: LLTokenizer,
    minimal_tokenizer: _MinimalTokenizer,
    wire: str,
) -> bool:
    matcher = LLMatcher(ll_tokenizer, grammar)
    toks = minimal_tokenizer(wire)
    return matcher.validate_tokens(toks) == len(toks)


def _call(args_xml: str, name: str = "book") -> str:
    return f"<tool_call>{name}{args_xml}</tool_call>{_TURN_END}"


def test_schema_aware_accepts_valid_typed_args(
    ll_tokenizer: LLTokenizer,
    mock_tokenizer: PipelineTokenizer[Any, Any, Any],
    minimal_tokenizer: _MinimalTokenizer,
) -> None:
    g = GlmToolParser.generate_tool_call_grammar(
        tools=_SCHEMA_TOOL, tokenizer=mock_tokenizer
    )
    # string bare, integer JSON, enum value — all valid.
    wire = _call(
        "<arg_key>city</arg_key><arg_value>Paris</arg_value>"
        "<arg_key>nights</arg_key><arg_value>3</arg_value>"
        "<arg_key>klass</arg_key><arg_value>first</arg_value>"
    )
    assert _validate(g, ll_tokenizer, minimal_tokenizer, wire)


def test_schema_aware_rejects_wrong_type(
    ll_tokenizer: LLTokenizer,
    mock_tokenizer: PipelineTokenizer[Any, Any, Any],
    minimal_tokenizer: _MinimalTokenizer,
) -> None:
    g = GlmToolParser.generate_tool_call_grammar(
        tools=_SCHEMA_TOOL, tokenizer=mock_tokenizer
    )
    # nights must be an integer; a bare string is not accepted.
    wire = _call(
        "<arg_key>city</arg_key><arg_value>Paris</arg_value>"
        "<arg_key>nights</arg_key><arg_value>many</arg_value>"
    )
    assert not _validate(g, ll_tokenizer, minimal_tokenizer, wire)


def test_schema_aware_rejects_bad_enum(
    ll_tokenizer: LLTokenizer,
    mock_tokenizer: PipelineTokenizer[Any, Any, Any],
    minimal_tokenizer: _MinimalTokenizer,
) -> None:
    g = GlmToolParser.generate_tool_call_grammar(
        tools=_SCHEMA_TOOL, tokenizer=mock_tokenizer
    )
    wire = _call(
        "<arg_key>city</arg_key><arg_value>Paris</arg_value>"
        "<arg_key>nights</arg_key><arg_value>2</arg_value>"
        "<arg_key>klass</arg_key><arg_value>business</arg_value>"
    )
    assert not _validate(g, ll_tokenizer, minimal_tokenizer, wire)


def test_schema_aware_rejects_unknown_key(
    ll_tokenizer: LLTokenizer,
    mock_tokenizer: PipelineTokenizer[Any, Any, Any],
    minimal_tokenizer: _MinimalTokenizer,
) -> None:
    g = GlmToolParser.generate_tool_call_grammar(
        tools=_SCHEMA_TOOL, tokenizer=mock_tokenizer
    )
    # additionalProperties: false -> an undeclared key is rejected.
    wire = _call(
        "<arg_key>city</arg_key><arg_value>Paris</arg_value>"
        "<arg_key>nights</arg_key><arg_value>2</arg_value>"
        "<arg_key>bogus</arg_key><arg_value>x</arg_value>"
    )
    assert not _validate(g, ll_tokenizer, minimal_tokenizer, wire)


def test_schema_aware_rejects_missing_required(
    ll_tokenizer: LLTokenizer,
    mock_tokenizer: PipelineTokenizer[Any, Any, Any],
    minimal_tokenizer: _MinimalTokenizer,
) -> None:
    g = GlmToolParser.generate_tool_call_grammar(
        tools=_SCHEMA_TOOL, tokenizer=mock_tokenizer
    )
    # 'nights' (required) omitted.
    wire = _call("<arg_key>city</arg_key><arg_value>Paris</arg_value>")
    assert not _validate(g, ll_tokenizer, minimal_tokenizer, wire)


def test_parsers_registered_under_glm45() -> None:
    """The arch wires ``tool_parser="glm45"`` / ``reasoning_parser="glm45"``."""
    # Importing the arch triggers the @register side effects.
    import max.pipelines.architectures.glm5_1.arch  # noqa: F401
    from max.pipelines.architectures.glm5_1.reasoning import GlmReasoningParser
    from max.pipelines.lib.reasoning import (
        get_parser_cls as get_reasoning_cls,
    )
    from max.pipelines.lib.tool_parsing import get_parser_cls as get_tool_cls

    assert get_tool_cls("glm45") is GlmToolParser
    assert get_reasoning_cls("glm45") is GlmReasoningParser
