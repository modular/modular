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
import os
import struct
import sys
import uuid
from typing import Any
from unittest.mock import patch

import llguidance.hf
import pytest
from llguidance import LLMatcher, LLTokenizer
from llguidance._tokenizer import TokenizerWrapper
from max.pipelines.architectures.minimax_m2.tool_parser import (
    MinimaxM2ToolParser,
)
from max.pipelines.lib.tool_parsing import (
    _TOOL_CALL_ID_LENGTH,
    StreamingToolCallState,
)
from max.pipelines.modeling.types import (
    ParsedToolCall,
    ParsedToolCallDelta,
    ParsedToolResponse,
)
from transformers import AutoTokenizer, PreTrainedTokenizerFast


def test_single_tool_call_parsing() -> None:
    """Test parsing a single tool call with MiniMax M2 XML tags."""
    parser = MinimaxM2ToolParser()

    response = """<minimax:tool_call>
<invoke name="get_weather">
<parameter name="location">New York</parameter>
<parameter name="unit">fahrenheit</parameter>
</invoke>
</minimax:tool_call>"""

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
    """Test parsing multiple tool calls from MiniMax M2 response."""
    parser = MinimaxM2ToolParser()

    response = """<minimax:tool_call>
<invoke name="get_weather">
<parameter name="location">New York</parameter>
</invoke>
<invoke name="get_time">
<parameter name="timezone">EST</parameter>
</invoke>
</minimax:tool_call>"""

    result = parser.parse_complete(response)

    assert len(result.tool_calls) == 2

    tool_call1 = result.tool_calls[0]
    assert tool_call1.name == "get_weather"
    assert json.loads(tool_call1.arguments) == {"location": "New York"}

    tool_call2 = result.tool_calls[1]
    assert tool_call2.name == "get_time"
    assert json.loads(tool_call2.arguments) == {"timezone": "EST"}

    assert tool_call1.id != tool_call2.id


def test_response_without_tool_calls() -> None:
    """Test parsing a response without tool calls section."""
    parser = MinimaxM2ToolParser()

    response = "This is just a regular response with no tool calls."

    result = parser.parse_complete(response)

    assert result.content == response
    assert len(result.tool_calls) == 0


def test_empty_response() -> None:
    """Test parsing an empty response."""
    parser = MinimaxM2ToolParser()

    result = parser.parse_complete("")

    assert result.content == ""
    assert len(result.tool_calls) == 0


def test_content_before_tool_calls() -> None:
    """Test parsing response with content before tool calls section."""
    parser = MinimaxM2ToolParser()

    response = """I'll help you check the weather.

<minimax:tool_call>
<invoke name="get_weather">
<parameter name="location">Boston</parameter>
</invoke>
</minimax:tool_call>"""

    result = parser.parse_complete(response)

    assert result.content == "I'll help you check the weather."
    assert len(result.tool_calls) == 1
    assert result.tool_calls[0].name == "get_weather"


def test_quoted_function_name() -> None:
    """Test parsing invoke with double-quoted function name."""
    parser = MinimaxM2ToolParser()

    response = """<minimax:tool_call>
<invoke name="search_articles">
<parameter name="query">python tutorials</parameter>
</invoke>
</minimax:tool_call>"""

    result = parser.parse_complete(response)

    assert len(result.tool_calls) == 1
    assert result.tool_calls[0].name == "search_articles"


def test_single_quoted_function_name() -> None:
    """Test parsing invoke with single-quoted function name."""
    parser = MinimaxM2ToolParser()

    response = """<minimax:tool_call>
<invoke name='calculate'>
<parameter name='expression'>2 + 2</parameter>
</invoke>
</minimax:tool_call>"""

    result = parser.parse_complete(response)

    assert len(result.tool_calls) == 1
    assert result.tool_calls[0].name == "calculate"


def test_unquoted_function_name() -> None:
    """Test parsing invoke with unquoted function name."""
    parser = MinimaxM2ToolParser()

    response = """<minimax:tool_call>
<invoke name=get_random_fact>
<parameter name=category>science</parameter>
</invoke>
</minimax:tool_call>"""

    result = parser.parse_complete(response)

    assert len(result.tool_calls) == 1
    assert result.tool_calls[0].name == "get_random_fact"


def test_complex_parameters() -> None:
    """Test parsing tool call with complex nested parameters."""
    parser = MinimaxM2ToolParser()

    filters_obj = {
        "date_range": {"start": "2023-01-01", "end": "2023-12-31"},
        "categories": ["ai", "tech"],
        "min_score": 0.8,
    }
    options_obj = {"limit": 10, "sort": "relevance", "include_metadata": True}

    response = f"""<minimax:tool_call>
<invoke name="search_articles">
<parameter name="query">machine learning</parameter>
<parameter name="filters">{json.dumps(filters_obj)}</parameter>
<parameter name="options">{json.dumps(options_obj)}</parameter>
</invoke>
</minimax:tool_call>"""

    result = parser.parse_complete(response)

    assert len(result.tool_calls) == 1
    tool_call = result.tool_calls[0]
    assert tool_call.name == "search_articles"
    parsed_args = json.loads(tool_call.arguments)
    assert parsed_args["query"] == "machine learning"
    assert parsed_args["filters"] == filters_obj
    assert parsed_args["options"] == options_obj


def test_empty_parameters() -> None:
    """Test parsing tool call with no parameters."""
    parser = MinimaxM2ToolParser()

    response = """<minimax:tool_call>
<invoke name="get_random_fact">
</invoke>
</minimax:tool_call>"""

    result = parser.parse_complete(response)

    assert len(result.tool_calls) == 1
    tool_call = result.tool_calls[0]
    assert tool_call.name == "get_random_fact"
    assert json.loads(tool_call.arguments) == {}


def test_tool_calls_section_without_end_tag() -> None:
    """Test parsing when </minimax:tool_call> end tag is missing."""
    parser = MinimaxM2ToolParser()

    response = """<minimax:tool_call>
<invoke name="test">
<parameter name="key">value</parameter>
</invoke>"""

    result = parser.parse_complete(response)

    assert len(result.tool_calls) == 1
    assert result.tool_calls[0].name == "test"


def test_empty_tool_calls_section_raises_error() -> None:
    """Test that empty tool calls section raises ValueError."""
    parser = MinimaxM2ToolParser()

    response = """<minimax:tool_call>
</minimax:tool_call>"""

    with pytest.raises(ValueError, match=r"no valid tool calls parsed"):
        parser.parse_complete(response)


def test_unique_tool_call_ids() -> None:
    """Test that each tool call gets a unique ID."""
    parser = MinimaxM2ToolParser()

    response = """<minimax:tool_call>
<invoke name="test">
<parameter name="param">value</parameter>
</invoke>
</minimax:tool_call>"""

    ids = set()
    for _ in range(10):
        result = parser.parse_complete(response)
        ids.add(result.tool_calls[0].id)

    assert len(ids) == 10


def test_tool_call_id_format() -> None:
    """Test that tool call IDs have the correct format."""
    parser = MinimaxM2ToolParser()

    response = """<minimax:tool_call>
<invoke name="test">
<parameter name="param">value</parameter>
</invoke>
</minimax:tool_call>"""

    result = parser.parse_complete(response)
    tool_call_id = result.tool_calls[0].id

    assert tool_call_id.startswith("call_")
    suffix = tool_call_id[len("call_") :]
    assert len(suffix) == _TOOL_CALL_ID_LENGTH
    int(suffix, 16)  # Verify it's valid hex


def test_response_structure() -> None:
    """Test that the response structure matches expected format."""
    parser = MinimaxM2ToolParser()

    response = """<minimax:tool_call>
<invoke name="calculate">
<parameter name="expression">2 + 2</parameter>
</invoke>
</minimax:tool_call>"""

    result = parser.parse_complete(response)

    assert isinstance(result, ParsedToolResponse)
    assert result.content is None
    assert len(result.tool_calls) == 1

    tool_call = result.tool_calls[0]
    assert isinstance(tool_call, ParsedToolCall)
    assert tool_call.name == "calculate"
    assert tool_call.id.startswith("call_")


def test_whitespace_handling() -> None:
    """Test that whitespace in arguments is handled correctly."""
    parser = MinimaxM2ToolParser()

    response = """<minimax:tool_call>
<invoke name="test">
<parameter name="key">   value with spaces   </parameter>
<parameter name="nested">{"inner": "data"}</parameter>
</invoke>
</minimax:tool_call>"""

    result = parser.parse_complete(response)

    assert len(result.tool_calls) == 1
    args = json.loads(result.tool_calls[0].arguments)
    assert args["key"] == "value with spaces"
    assert args["nested"] == {"inner": "data"}


def test_special_characters_in_arguments() -> None:
    """Test handling of special characters in tool arguments."""
    parser = MinimaxM2ToolParser()

    response = """<minimax:tool_call>
<invoke name="execute">
<parameter name="code">print("Hello, World!")</parameter>
<parameter name="regex">\\d+\\.\\d+</parameter>
<parameter name="unicode">\u4e16\u754c</parameter>
</invoke>
</minimax:tool_call>"""

    result = parser.parse_complete(response)

    assert len(result.tool_calls) == 1
    parsed_args = json.loads(result.tool_calls[0].arguments)
    assert parsed_args["code"] == 'print("Hello, World!")'
    assert parsed_args["regex"] == r"\d+\.\d+"
    assert parsed_args["unicode"] == "\u4e16\u754c"


def test_multiple_tool_calls_same_function() -> None:
    """Test parsing multiple calls to the same function."""
    parser = MinimaxM2ToolParser()

    response = """<minimax:tool_call>
<invoke name="search">
<parameter name="query">first query</parameter>
</invoke>
<invoke name="search">
<parameter name="query">second query</parameter>
</invoke>
<invoke name="search">
<parameter name="query">third query</parameter>
</invoke>
</minimax:tool_call>"""

    result = parser.parse_complete(response)

    assert len(result.tool_calls) == 3

    for tc in result.tool_calls:
        assert tc.name == "search"

    ids = [tc.id for tc in result.tool_calls]
    assert len(set(ids)) == 3

    queries = [json.loads(tc.arguments)["query"] for tc in result.tool_calls]
    assert queries == ["first query", "second query", "third query"]


def test_numeric_parameter_values() -> None:
    """Test that numeric parameter values are parsed correctly."""
    parser = MinimaxM2ToolParser()

    response = """<minimax:tool_call>
<invoke name="configure">
<parameter name="count">5</parameter>
<parameter name="threshold">0.75</parameter>
<parameter name="enabled">true</parameter>
<parameter name="label">text_value</parameter>
</invoke>
</minimax:tool_call>"""

    result = parser.parse_complete(response)

    args = json.loads(result.tool_calls[0].arguments)
    assert args["count"] == 5
    assert args["threshold"] == 0.75
    assert args["enabled"] is True
    assert args["label"] == "text_value"


# --- Streaming tests ---


def test_reset_clears_buffer() -> None:
    """Test that reset() clears the internal buffer and streaming state."""
    parser = MinimaxM2ToolParser()

    parser._buffer = "some accumulated data"
    parser._state.sent_content_idx = 10
    parser._state.tool_calls.append(StreamingToolCallState())

    parser.reset()

    assert parser._buffer == ""
    assert parser._state.sent_content_idx == 0
    assert len(parser._state.tool_calls) == 0


def test_parse_delta_accumulates() -> None:
    """Test that parse_delta accumulates tokens in buffer."""
    parser = MinimaxM2ToolParser()

    # parse_delta should accumulate tokens; return [] to indicate parser is actively buffering and raw tokens shouldn't be used yet.
    result1 = parser.parse_delta("<minimax:")
    assert result1 == []

    # Once the marker completes, returns [] (not None) so the streaming
    # path knows to suppress raw structural tokens even with no deltas yet.
    result2 = parser.parse_delta("tool_call>")
    assert result2 == []
    assert parser._buffer == "<minimax:tool_call>"


def test_parse_delta_single_tool_call_streaming() -> None:
    """Test streaming a single tool call token by token."""
    fixed_uuid = uuid.UUID("12345678-1234-5678-9abc-def012345678")
    with patch(
        "max.pipelines.lib.tool_parsing.uuid.uuid4",
        return_value=fixed_uuid,
    ):
        parser = MinimaxM2ToolParser()

        chunks = [
            "<minimax:tool_call>",
            '<invoke name="get_weather">',
            '<parameter name="location">New York</parameter>',
            "</invoke>",
            "</minimax:tool_call>",
        ]

        all_deltas: list[ParsedToolCallDelta] = []
        for chunk in chunks:
            result = parser.parse_delta(chunk)
            if result:
                all_deltas.extend(result)

        expected_id = f"call_{fixed_uuid.hex[:_TOOL_CALL_ID_LENGTH]}"
        assert all_deltas == [
            ParsedToolCallDelta(
                index=0,
                id=expected_id,
                name="get_weather",
            ),
            ParsedToolCallDelta(index=0, arguments='{"location": "New York"'),
            ParsedToolCallDelta(index=0, arguments="}"),
        ]


def test_parse_delta_multiple_tool_calls_streaming() -> None:
    """Test streaming multiple tool calls."""
    uuid_first = uuid.UUID("11111111-1111-1111-1111-111111111111")
    uuid_second = uuid.UUID("22222222-2222-2222-2222-222222222222")
    with patch(
        "max.pipelines.lib.tool_parsing.uuid.uuid4",
        side_effect=[uuid_first, uuid_second],
    ):
        parser = MinimaxM2ToolParser()

        response = """<minimax:tool_call>
<invoke name="get_weather">
<parameter name="location">NYC</parameter>
</invoke>
<invoke name="get_time">
<parameter name="zone">EST</parameter>
</invoke>
</minimax:tool_call>"""

        result = parser.parse_delta(response)

        id_first = f"call_{uuid_first.hex[:_TOOL_CALL_ID_LENGTH]}"
        id_second = f"call_{uuid_second.hex[:_TOOL_CALL_ID_LENGTH]}"
        assert result == [
            ParsedToolCallDelta(
                index=0,
                id=id_first,
                name="get_weather",
            ),
            ParsedToolCallDelta(
                index=0,
                arguments='{"location": "NYC"}',
            ),
            ParsedToolCallDelta(
                index=1,
                id=id_second,
                name="get_time",
            ),
            ParsedToolCallDelta(
                index=1,
                arguments='{"zone": "EST"}',
            ),
        ]


def test_parse_delta_with_content_before_tools() -> None:
    """Test streaming when there's content before tool calls section."""
    fixed_uuid = uuid.UUID("12345678-1234-5678-9abc-def012345678")
    with patch(
        "max.pipelines.lib.tool_parsing.uuid.uuid4",
        return_value=fixed_uuid,
    ):
        parser = MinimaxM2ToolParser()

        chunks = [
            "I'll check the weather for you.\n\n",
            "<minimax:tool_call>",
            '<invoke name="get_weather"><parameter name="location">Boston</parameter></invoke>',
            "</minimax:tool_call>",
        ]

        all_deltas: list[ParsedToolCallDelta] = []
        for chunk in chunks:
            result = parser.parse_delta(chunk)
            if result:
                all_deltas.extend(result)

        expected_id = f"call_{fixed_uuid.hex[:_TOOL_CALL_ID_LENGTH]}"
        assert all_deltas == [
            ParsedToolCallDelta(
                index=0,
                content="I'll check the weather for you.\n\n",
            ),
            ParsedToolCallDelta(
                index=0,
                id=expected_id,
                name="get_weather",
            ),
            ParsedToolCallDelta(
                index=0,
                arguments='{"location": "Boston"}',
            ),
        ]


def test_parse_delta_argument_diffing() -> None:
    """Test that argument deltas are properly diffed across parameters."""
    parser = MinimaxM2ToolParser()

    parser.parse_delta("<minimax:tool_call>")
    parser.parse_delta('<invoke name="test">')

    result1 = parser.parse_delta('<parameter name="a">1</parameter>')
    result2 = parser.parse_delta('<parameter name="b">2</parameter>')
    result3 = parser.parse_delta("</invoke>")

    all_args = []
    for r in [result1, result2, result3]:
        if r:
            for delta in r:
                if delta.arguments is not None:
                    all_args.append(delta.arguments)

    full = "".join(all_args)
    assert json.loads(full) == {"a": 1, "b": 2}


def test_parse_delta_reset_clears_state() -> None:
    """Test that reset() clears all streaming state."""
    parser = MinimaxM2ToolParser()

    parser.parse_delta("<minimax:tool_call>")
    parser.parse_delta(
        '<invoke name="test"><parameter name="key">value</parameter>'
    )

    assert parser._buffer != ""
    assert len(parser._state.tool_calls) > 0

    parser.reset()

    assert parser._buffer == ""
    assert len(parser._state.tool_calls) == 0
    assert parser._state.sent_content_idx == 0


def test_parse_delta_partial_marker_handling() -> None:
    """Test that partial markers at buffer end are held back."""
    parser = MinimaxM2ToolParser()

    result1 = parser.parse_delta("Hello world<minimax:")

    assert result1 is not None
    assert len(result1) == 1
    assert result1[0].content == "Hello world"

    result2 = parser.parse_delta("tool_call>")

    # Once the section marker is complete, the parser is inside the
    # tool-calls section and returns [] to signal suppression.
    assert result2 == []
    assert "<minimax:tool_call>" in parser._buffer


def test_parse_delta_mid_token_splits() -> None:
    """Test streaming with tokens that split mid-tag."""
    fixed_uuid = uuid.UUID("12345678-1234-5678-9abc-def012345678")
    with patch(
        "max.pipelines.lib.tool_parsing.uuid.uuid4",
        return_value=fixed_uuid,
    ):
        parser = MinimaxM2ToolParser()

        chunks = [
            "<minimax:tool_",
            "call>",
            "<invoke na",
            'me="get_we',
            'ather">',  # spellchecker:disable-line
            "<parameter",
            ' name="loc',
            'ation">New',
            " York</para",
            "meter>",
            "</inv",
            "oke>",
            "</minimax:tool_call>",
        ]

        all_deltas: list[ParsedToolCallDelta] = []
        for chunk in chunks:
            result = parser.parse_delta(chunk)
            if result:
                all_deltas.extend(result)

        expected_id = f"call_{fixed_uuid.hex[:_TOOL_CALL_ID_LENGTH]}"

        name_deltas = [d for d in all_deltas if d.name]
        assert len(name_deltas) == 1
        assert name_deltas[0].name == "get_weather"
        assert name_deltas[0].id == expected_id

        arg_deltas = [d for d in all_deltas if d.arguments is not None]
        full_args = "".join(
            d.arguments for d in arg_deltas if d.arguments is not None
        )
        assert json.loads(full_args) == {"location": "New York"}


def test_parse_delta_ignores_invoke_after_end_tag() -> None:
    """Test that invoke blocks after </minimax:tool_call> are not parsed."""
    fixed_uuid = uuid.UUID("12345678-1234-5678-9abc-def012345678")
    with patch(
        "max.pipelines.lib.tool_parsing.uuid.uuid4",
        return_value=fixed_uuid,
    ):
        parser = MinimaxM2ToolParser()

        response = (
            "<minimax:tool_call>"
            '<invoke name="real"><parameter name="k">v</parameter></invoke>'
            "</minimax:tool_call>"
            '<invoke name="spurious"><parameter name="x">y</parameter></invoke>'
        )

        result = parser.parse_delta(response)

        assert result is not None
        name_deltas = [d for d in result if d.name]
        assert len(name_deltas) == 1
        assert name_deltas[0].name == "real"


def test_parse_delta_multiple_tool_call_blocks() -> None:
    """Test streaming with multiple <minimax:tool_call> blocks."""
    uuid_first = uuid.UUID("11111111-1111-1111-1111-111111111111")
    uuid_second = uuid.UUID("22222222-2222-2222-2222-222222222222")
    with patch(
        "max.pipelines.lib.tool_parsing.uuid.uuid4",
        side_effect=[uuid_first, uuid_second],
    ):
        parser = MinimaxM2ToolParser()

        response = (
            "<minimax:tool_call>"
            '<invoke name="first"><parameter name="a">1</parameter></invoke>'
            "</minimax:tool_call>"
            "<minimax:tool_call>"
            '<invoke name="second"><parameter name="b">2</parameter></invoke>'
            "</minimax:tool_call>"
        )

        result = parser.parse_delta(response)

        assert result is not None
        name_deltas = [d for d in result if d.name]
        assert len(name_deltas) == 2
        assert name_deltas[0].name == "first"
        assert name_deltas[1].name == "second"

        arg_deltas = [d for d in result if d.arguments is not None]
        assert len(arg_deltas) == 2
        assert arg_deltas[0].arguments is not None
        assert arg_deltas[1].arguments is not None
        assert json.loads(arg_deltas[0].arguments) == {"a": 1}
        assert json.loads(arg_deltas[1].arguments) == {"b": 2}


def test_parse_delta_no_tool_calls() -> None:
    """Test that parse_delta streams content when no tool calls are present."""
    parser = MinimaxM2ToolParser()

    chunks = ["Hello, ", "this is ", "a plain response."]

    all_deltas: list[ParsedToolCallDelta] = []
    for chunk in chunks:
        result = parser.parse_delta(chunk)
        if result:
            all_deltas.extend(result)

    content_deltas = [d for d in all_deltas if d.content is not None]
    full_content = "".join(
        d.content for d in content_deltas if d.content is not None
    )
    assert full_content == "Hello, this is a plain response."
    assert not any(d.name for d in all_deltas)
    assert not any(d.arguments is not None for d in all_deltas)


def test_parse_delta_no_content_after_end_tag() -> None:
    """Test that text after </minimax:tool_call> produces no content deltas."""
    fixed_uuid = uuid.UUID("12345678-1234-5678-9abc-def012345678")
    with patch(
        "max.pipelines.lib.tool_parsing.uuid.uuid4",
        return_value=fixed_uuid,
    ):
        parser = MinimaxM2ToolParser()

        chunks = [
            "<minimax:tool_call>",
            '<invoke name="test"><parameter name="k">v</parameter></invoke>',
            "</minimax:tool_call>",
            "trailing content that should be ignored",
        ]

        all_deltas: list[ParsedToolCallDelta] = []
        for chunk in chunks:
            result = parser.parse_delta(chunk)
            if result:
                all_deltas.extend(result)

        content_deltas = [d for d in all_deltas if d.content is not None]
        assert len(content_deltas) == 0


def test_parse_complete_whitespace_only_before_tool_calls() -> None:
    """Test that whitespace-only content before tool calls yields content=None."""
    parser = MinimaxM2ToolParser()

    response = '\n\n<minimax:tool_call>\n<invoke name="ping">\n</invoke>\n</minimax:tool_call>'

    result = parser.parse_complete(response)

    assert result.content is None
    assert len(result.tool_calls) == 1
    assert result.tool_calls[0].name == "ping"


def test_parse_delta_streaming_empty_invoke() -> None:
    """Test that a streaming invoke with no parameters produces a {} argument delta."""
    fixed_uuid = uuid.UUID("12345678-1234-5678-9abc-def012345678")
    with patch(
        "max.pipelines.lib.tool_parsing.uuid.uuid4",
        return_value=fixed_uuid,
    ):
        parser = MinimaxM2ToolParser()

        chunks = [
            "<minimax:tool_call>",
            '<invoke name="ping">',
            "</invoke>",
            "</minimax:tool_call>",
        ]

        all_deltas: list[ParsedToolCallDelta] = []
        for chunk in chunks:
            result = parser.parse_delta(chunk)
            if result:
                all_deltas.extend(result)

        expected_id = f"call_{fixed_uuid.hex[:_TOOL_CALL_ID_LENGTH]}"
        assert all_deltas == [
            ParsedToolCallDelta(index=0, id=expected_id, name="ping"),
            ParsedToolCallDelta(index=0, arguments="{}"),
        ]


# ---- Grammar tests (PR1) -----------------------------------------------------


def _tool_def(name: str) -> dict[str, Any]:
    """Build a minimal OpenAI-style tool definition for grammar tests."""
    return {"type": "function", "function": {"name": name, "parameters": {}}}


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
    """Minimal LLTokenizer for grammar compile smoke tests."""
    wrapper = TokenizerWrapper(_MinimalTokenizer())
    return LLTokenizer(wrapper, n_vocab=256)


def test_generate_tool_call_grammar_compiles_with_tool_names(
    ll_tokenizer: LLTokenizer,
) -> None:
    """Grammar with explicit tool names compiles cleanly."""
    grammar = MinimaxM2ToolParser.generate_tool_call_grammar(
        tools=[_tool_def("get_weather"), _tool_def("search")]
    )
    assert isinstance(grammar, str)
    assert len(grammar) > 0
    matcher = LLMatcher(ll_tokenizer, grammar)
    assert matcher is not None


def test_generate_tool_call_grammar_compiles_without_tool_names(
    ll_tokenizer: LLTokenizer,
) -> None:
    """Grammar with ``tools=None`` (any identifier) compiles cleanly."""
    grammar = MinimaxM2ToolParser.generate_tool_call_grammar(tools=None)
    assert isinstance(grammar, str)
    assert len(grammar) > 0
    matcher = LLMatcher(ll_tokenizer, grammar)
    assert matcher is not None


def test_generate_tool_call_grammar_escapes_special_chars(
    ll_tokenizer: LLTokenizer,
) -> None:
    """Tool names containing regex metacharacters are escaped."""
    grammar = MinimaxM2ToolParser.generate_tool_call_grammar(
        tools=[
            _tool_def("get_weather.v2"),
            _tool_def("search+plus"),
            _tool_def("tool[0]"),
        ]
    )
    matcher = LLMatcher(ll_tokenizer, grammar)
    assert matcher is not None


def test_generate_tool_call_grammar_compiles_with_response_format_schema(
    ll_tokenizer: LLTokenizer,
) -> None:
    """Combined tool-call + JSON-schema grammar compiles cleanly."""
    schema = {
        "type": "object",
        "properties": {"answer": {"type": "string"}},
        "required": ["answer"],
    }
    grammar = MinimaxM2ToolParser.generate_tool_call_grammar(
        tools=[_tool_def("get_weather")],
        response_format_schema=schema,
    )
    assert isinstance(grammar, str)
    assert len(grammar) > 0
    matcher = LLMatcher(ll_tokenizer, grammar)
    assert matcher is not None


# ---- Real-tokenizer acceptance tests (PR1.3) ---------------------------------

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _grammar_fixtures  # type: ignore[import-not-found]

_MINIMAX_HF_REPO = "MiniMaxAI/MiniMax-M2.7"


def _token_allowed(bitmask: bytes, tok_id: int) -> bool:
    """Return True if *tok_id* is set in the packed-int32 *bitmask*."""
    word = struct.unpack_from("<I", bitmask, (tok_id // 32) * 4)[0]
    return bool(word & (1 << (tok_id % 32)))


@pytest.fixture(scope="module")
def minimax_ll_tokenizer() -> tuple[LLTokenizer, PreTrainedTokenizerFast]:
    """Real MiniMax HF tokenizer wrapped for llguidance.

    Skipped if the tokenizer can't be downloaded (no HF_TOKEN, offline, etc).
    """
    try:
        tok = AutoTokenizer.from_pretrained(_MINIMAX_HF_REPO)
    except Exception as e:
        pytest.skip(f"Could not download MiniMax tokenizer: {e}")
    assert isinstance(tok, PreTrainedTokenizerFast)
    return llguidance.hf.from_tokenizer(tok, n_vocab=len(tok)), tok


@pytest.mark.parametrize(
    "fixture", _grammar_fixtures.GOOD_ENVELOPES, ids=lambda f: f.name
)
def test_grammar_accepts_known_good_envelope(
    minimax_ll_tokenizer: tuple[LLTokenizer, PreTrainedTokenizerFast],
    fixture: "_grammar_fixtures.GoodEnvelope",
) -> None:
    """Walk each known-good envelope token-by-token; grammar must accept all.

    Uses the real MiniMax HF tokenizer so token-boundary edge cases are
    exercised (e.g., ``<minimax:tool_call>`` is token ID 200052 — a single
    token — and must be accepted atomically by the Lark grammar rule rather
    than byte-by-byte via a regex prefix).
    """
    ll_tok, hf_tok = minimax_ll_tokenizer
    tool_names = sorted({name for name, _ in fixture.expected_calls})
    grammar = MinimaxM2ToolParser.generate_tool_call_grammar(
        tools=[_tool_def(n) for n in tool_names]
    )
    matcher = LLMatcher(ll_tok, grammar)

    token_ids = hf_tok.encode(fixture.envelope, add_special_tokens=False)
    for i, tok_id in enumerate(token_ids):
        bitmask = matcher.compute_bitmask()
        assert _token_allowed(bitmask, tok_id), (
            f"Grammar rejected valid token at step {i} "
            f"(token_id={tok_id}, decoded={hf_tok.decode([tok_id])!r}) "
            f"in fixture {fixture.name!r}"
        )
        matcher.consume_token(tok_id)
    assert matcher.is_accepting(), (
        f"Matcher did not reach accepting state for fixture {fixture.name!r}"
    )


@pytest.mark.parametrize(
    "fixture", _grammar_fixtures.BAD_ENVELOPES, ids=lambda f: f.name
)
def test_grammar_rejects_malformed_envelope(
    minimax_ll_tokenizer: tuple[LLTokenizer, PreTrainedTokenizerFast],
    fixture: "_grammar_fixtures.BadEnvelope",
) -> None:
    """Walk each malformed envelope token-by-token; grammar must reject."""
    ll_tok, hf_tok = minimax_ll_tokenizer
    # tools=None uses the permissive name pattern; rejection must hold
    # even when names are unconstrained.
    grammar = MinimaxM2ToolParser.generate_tool_call_grammar(tools=None)
    matcher = LLMatcher(ll_tok, grammar)

    token_ids = hf_tok.encode(fixture.envelope, add_special_tokens=False)
    rejected = False
    for tok_id in token_ids:
        bitmask = matcher.compute_bitmask()
        if not _token_allowed(bitmask, tok_id):
            rejected = True
            break
        matcher.consume_token(tok_id)

    if not rejected and matcher.is_accepting():
        pytest.fail(
            f"Grammar accepted malformed envelope {fixture.name!r}; "
            f"expected rejection because: {fixture.rejection_hint}"
        )
