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
import time
from typing import Any

import pytest
from max import _xgrammar as xgrammar
from max.pipelines.architectures.qwen3_5.tool_parser import Qwen3_5ToolParser
from max.pipelines.modeling.types import (
    ParsedToolCall,
    ParsedToolCallDelta,
    ParsedToolResponse,
)

# ---------------------------------------------------------------------------
# parse_complete
# ---------------------------------------------------------------------------


def test_parse_complete_single_call() -> None:
    parser = Qwen3_5ToolParser()
    response = (
        "<tool_call>\n"
        "<function=get_weather>\n"
        "<parameter=location>\nParis\n</parameter>\n"
        "<parameter=unit>\ncelsius\n</parameter>\n"
        "</function>\n"
        "</tool_call>"
    )

    result = parser.parse_complete(response)
    assert isinstance(result, ParsedToolResponse)
    assert result.content is None
    assert len(result.tool_calls) == 1

    call = result.tool_calls[0]
    assert isinstance(call, ParsedToolCall)
    assert call.id.startswith("call_")
    assert call.name == "get_weather"
    assert json.loads(call.arguments) == {
        "location": "Paris",
        "unit": "celsius",
    }


def test_parse_complete_complex_args() -> None:
    """Non-string values (numbers, bools, lists, objects) are JSON-encoded
    by the chat template and should round-trip through the parser."""
    parser = Qwen3_5ToolParser()
    response = (
        "<tool_call>\n"
        "<function=get_weather>\n"
        '<parameter=locations>\n[{"country": "France", "city": "Paris"}]\n</parameter>\n'
        "<parameter=temp_units>\ncelsius\n</parameter>\n"
        "<parameter=detail>\n3\n</parameter>\n"
        "<parameter=verbose>\ntrue\n</parameter>\n"
        "</function>\n"
        "</tool_call>"
    )

    result = parser.parse_complete(response)
    assert len(result.tool_calls) == 1
    args = json.loads(result.tool_calls[0].arguments)
    assert args == {
        "locations": [{"country": "France", "city": "Paris"}],
        "temp_units": "celsius",
        "detail": 3,
        "verbose": True,
    }


def test_parse_complete_multiple_calls() -> None:
    parser = Qwen3_5ToolParser()
    response = (
        "<tool_call>\n<function=a>\n<parameter=x>\n1\n</parameter>\n</function>\n</tool_call>\n"
        "<tool_call>\n<function=b>\n<parameter=y>\nhello\n</parameter>\n</function>\n</tool_call>"
    )

    result = parser.parse_complete(response)
    assert len(result.tool_calls) == 2
    assert result.tool_calls[0].name == "a"
    assert json.loads(result.tool_calls[0].arguments) == {"x": 1}
    assert result.tool_calls[1].name == "b"
    assert json.loads(result.tool_calls[1].arguments) == {"y": "hello"}
    assert result.tool_calls[0].id != result.tool_calls[1].id


def test_parse_complete_with_preamble() -> None:
    parser = Qwen3_5ToolParser()
    response = (
        "I should check the weather first.\n\n"
        "<tool_call>\n<function=get_weather>\n"
        "<parameter=city>\nBerlin\n</parameter>\n"
        "</function>\n</tool_call>"
    )

    result = parser.parse_complete(response)
    assert result.content == "I should check the weather first."
    assert len(result.tool_calls) == 1
    assert result.tool_calls[0].name == "get_weather"


def test_parse_complete_no_tool_call() -> None:
    parser = Qwen3_5ToolParser()
    result = parser.parse_complete("just a regular response")
    assert result.content == "just a regular response"
    assert result.tool_calls == []


def test_parse_complete_empty_args() -> None:
    parser = Qwen3_5ToolParser()
    response = "<tool_call>\n<function=ping>\n</function>\n</tool_call>"
    result = parser.parse_complete(response)
    assert len(result.tool_calls) == 1
    assert result.tool_calls[0].name == "ping"
    assert json.loads(result.tool_calls[0].arguments) == {}


def test_parse_complete_python_style_scalars() -> None:
    """The chat template renders bools/None with Python ``str()`` (``True``/
    ``False``/``None``), not JSON. They must decode to real JSON scalars."""
    parser = Qwen3_5ToolParser()
    response = (
        "<tool_call>\n<function=configure>\n"
        "<parameter=enabled>\nTrue\n</parameter>\n"
        "<parameter=verbose>\nFalse\n</parameter>\n"
        "<parameter=fallback>\nNone\n</parameter>\n"
        "<parameter=retries>\n5\n</parameter>\n"
        "<parameter=label>\nhello\n</parameter>\n"
        "</function>\n</tool_call>"
    )
    result = parser.parse_complete(response)
    assert json.loads(result.tool_calls[0].arguments) == {
        "enabled": True,
        "verbose": False,
        "fallback": None,
        "retries": 5,
        "label": "hello",
    }


def test_parse_complete_multiline_string_value() -> None:
    parser = Qwen3_5ToolParser()
    response = (
        "<tool_call>\n<function=write_note>\n"
        "<parameter=text>\nline one\nline two\nline three\n</parameter>\n"
        "</function>\n</tool_call>"
    )
    result = parser.parse_complete(response)
    assert json.loads(result.tool_calls[0].arguments) == {
        "text": "line one\nline two\nline three"
    }


def test_parse_complete_invalid_block_raises() -> None:
    parser = Qwen3_5ToolParser()
    # Open tag but no <function=> inside.
    with pytest.raises(ValueError, match=r"no valid calls parsed"):
        parser.parse_complete("<tool_call>\nbroken\n</tool_call>")


# ---------------------------------------------------------------------------
# parse_delta (streaming)
# ---------------------------------------------------------------------------


def _feed(parser: Qwen3_5ToolParser, *deltas: str) -> list[ParsedToolCallDelta]:
    """Feed deltas one at a time and return the flat list of all emissions."""
    out: list[ParsedToolCallDelta] = []
    for d in deltas:
        result = parser.parse_delta(d)
        if result:
            out.extend(result)
    return out


def _split_content_and_calls(
    deltas: list[ParsedToolCallDelta],
) -> tuple[str | None, list[ParsedToolCallDelta]]:
    content_parts: list[str] = []
    tool_deltas: list[ParsedToolCallDelta] = []
    for d in deltas:
        if d.content is not None:
            content_parts.append(d.content)
        else:
            tool_deltas.append(d)
    content = "".join(content_parts) if content_parts else None
    return content, tool_deltas


def test_parse_delta_single_call_one_shot() -> None:
    parser = Qwen3_5ToolParser()
    response = (
        "<tool_call>\n<function=get_weather>\n"
        "<parameter=city>\nParis\n</parameter>\n"
        "<parameter=unit>\ncelsius\n</parameter>\n"
        "</function>\n</tool_call>"
    )
    content, tool_deltas = _split_content_and_calls(_feed(parser, response))
    assert content is None

    head = tool_deltas[0]
    assert head.id is not None
    assert head.name == "get_weather"
    assert head.arguments == "{"

    args_concat = "".join(d.arguments or "" for d in tool_deltas)
    assert json.loads(args_concat) == {"city": "Paris", "unit": "celsius"}
    assert all(d.index == 0 for d in tool_deltas)


def test_parse_delta_single_char_deltas() -> None:
    """The state machine has to survive sentinels split across deltas."""
    parser = Qwen3_5ToolParser()
    response = (
        "<tool_call>\n<function=fn>\n"
        "<parameter=key>\nvalue\n</parameter>\n"
        "</function>\n</tool_call>"
    )
    content, tool_deltas = _split_content_and_calls(
        _feed(parser, *list(response))
    )
    assert content is None
    args_concat = "".join(d.arguments or "" for d in tool_deltas)
    assert json.loads(args_concat) == {"key": "value"}


def test_parse_delta_pre_call_content_streamed() -> None:
    parser = Qwen3_5ToolParser()
    deltas = _feed(
        parser,
        "Let me think. ",
        "Calling tool now.\n",
        "<tool_call>\n<function=fn>\n",
        "<parameter=x>\n1\n</parameter>\n",
        "</function>\n</tool_call>",
    )
    content, tool_deltas = _split_content_and_calls(deltas)
    assert content is not None
    assert "Let me think. Calling tool now." in content
    assert tool_deltas[0].name == "fn"
    args_concat = "".join(d.arguments or "" for d in tool_deltas)
    assert json.loads(args_concat) == {"x": 1}


def test_parse_delta_partial_tag_held_back() -> None:
    """A trailing partial '<tool_' must not leak as content."""
    parser = Qwen3_5ToolParser()

    out1 = parser.parse_delta("hello ")
    assert out1 is not None
    assert len(out1) == 1
    assert out1[0].content == "hello "

    # parse_delta should accumulate tokens; return [] to indicate parser is actively buffering and raw tokens shouldn't be used yet.
    out2 = parser.parse_delta("<tool_")
    assert out2 == []

    out3 = parser.parse_delta(
        "call>\n<function=fn>\n<parameter=k>\nv\n</parameter>\n"
        "</function>\n</tool_call>"
    )
    assert out3 is not None
    content, tool_deltas = _split_content_and_calls(out3)
    assert content is None
    assert tool_deltas[0].name == "fn"


def test_parse_delta_multiple_calls() -> None:
    parser = Qwen3_5ToolParser()
    response = (
        "<tool_call>\n<function=a>\n"
        "<parameter=x>\n1\n</parameter>\n</function>\n</tool_call>\n"
        "<tool_call>\n<function=b>\n"
        "<parameter=y>\nhello\n</parameter>\n</function>\n</tool_call>"
    )
    content, tool_deltas = _split_content_and_calls(_feed(parser, response))
    # The "\n" separating back-to-back calls is structural and must not leak
    # into assistant content.
    assert content is None
    indices = sorted({d.index for d in tool_deltas})
    assert indices == [0, 1]

    by_index: dict[int, str] = {0: "", 1: ""}
    for d in tool_deltas:
        if d.arguments:
            by_index[d.index] += d.arguments
    assert json.loads(by_index[0]) == {"x": 1}
    assert json.loads(by_index[1]) == {"y": "hello"}

    name_deltas = [d for d in tool_deltas if d.name is not None]
    assert [d.name for d in name_deltas] == ["a", "b"]
    assert name_deltas[0].id != name_deltas[1].id


def test_parse_delta_complex_arg_value() -> None:
    parser = Qwen3_5ToolParser()
    response = (
        "<tool_call>\n<function=search>\n"
        '<parameter=filters>\n{"min": 1, "max": 10}\n</parameter>\n'
        '<parameter=tags>\n["a", "b"]\n</parameter>\n'
        "</function>\n</tool_call>"
    )
    _, tool_deltas = _split_content_and_calls(_feed(parser, response))
    args_concat = "".join(d.arguments or "" for d in tool_deltas)
    assert json.loads(args_concat) == {
        "filters": {"min": 1, "max": 10},
        "tags": ["a", "b"],
    }


def test_parse_delta_id_emitted_only_on_first_chunk() -> None:
    parser = Qwen3_5ToolParser()
    response = (
        "<tool_call>\n<function=fn>\n"
        "<parameter=a>\n1\n</parameter>\n"
        "<parameter=b>\n2\n</parameter>\n"
        "</function>\n</tool_call>"
    )
    _, tool_deltas = _split_content_and_calls(_feed(parser, response))
    head = tool_deltas[0]
    rest = tool_deltas[1:]
    assert head.id is not None and head.name == "fn"
    assert all(d.id is None and d.name is None for d in rest)


def test_parse_delta_no_tool_calls_streams_content() -> None:
    parser = Qwen3_5ToolParser()
    deltas = _feed(parser, "Hello, ", "this is ", "a plain response.")
    content, tool_deltas = _split_content_and_calls(deltas)
    assert content == "Hello, this is a plain response."
    assert tool_deltas == []


def test_parse_delta_python_style_scalars() -> None:
    """Streaming decodes Python-style ``True``/``False``/``None`` the same as
    ``parse_complete``."""
    parser = Qwen3_5ToolParser()
    response = (
        "<tool_call>\n<function=configure>\n"
        "<parameter=enabled>\nTrue\n</parameter>\n"
        "<parameter=fallback>\nNone\n</parameter>\n"
        "</function>\n</tool_call>"
    )
    _, tool_deltas = _split_content_and_calls(_feed(parser, response))
    args_concat = "".join(d.arguments or "" for d in tool_deltas)
    assert json.loads(args_concat) == {"enabled": True, "fallback": None}


def test_parse_delta_multiple_calls_no_separator_leak_char_by_char() -> None:
    """Fed one char at a time, the inter-call ``\\n`` separator must not leak
    as content (the parser returns ``[]`` to suppress it, not ``None``)."""
    parser = Qwen3_5ToolParser()
    response = (
        "<tool_call>\n<function=a>\n"
        "<parameter=x>\n1\n</parameter>\n</function>\n</tool_call>\n"
        "<tool_call>\n<function=b>\n"
        "<parameter=y>\nhello\n</parameter>\n</function>\n</tool_call>"
    )
    content, tool_deltas = _split_content_and_calls(
        _feed(parser, *list(response))
    )
    assert content is None
    by_index: dict[int, str] = {0: "", 1: ""}
    for d in tool_deltas:
        if d.arguments:
            by_index[d.index] += d.arguments
    assert json.loads(by_index[0]) == {"x": 1}
    assert json.loads(by_index[1]) == {"y": "hello"}


def test_parse_delta_trailing_text_after_call_suppressed() -> None:
    """Per the template, no content follows a function call; any trailing
    text after the final ``</tool_call>`` is suppressed rather than streamed."""
    parser = Qwen3_5ToolParser()
    deltas = _feed(
        parser,
        "<tool_call>\n<function=fn>\n"
        "<parameter=x>\n1\n</parameter>\n</function>\n</tool_call>",
        "\n",
        "trailing junk",
    )
    content, tool_deltas = _split_content_and_calls(deltas)
    assert content is None
    assert tool_deltas[0].name == "fn"


def test_reset_clears_state() -> None:
    parser = Qwen3_5ToolParser()
    parser.parse_delta("<tool_call>\n<function=fn>\n")
    parser.reset()
    out = parser.parse_delta("hi")
    assert out is not None
    content, tool_deltas = _split_content_and_calls(out)
    assert content == "hi"
    assert tool_deltas == []


# --- Schema-aware argument decoding -----------------------------------------

_SCHEMAS: dict[str, dict[str, Any]] = {
    "lookup": {
        "type": "object",
        "$defs": {"Zip": {"type": "string"}},
        "properties": {
            "zip_code": {"type": "string"},
            "ref_zip": {"$ref": "#/$defs/Zip"},
            "count": {"type": "integer"},
            "ratio": {"type": "number"},
            "flag": {"type": "boolean"},
            "note": {"type": ["string", "null"]},
            "either": {"anyOf": [{"type": "string"}, {"type": "integer"}]},
            "tags": {"type": "array", "items": {"type": "string"}},
            "version": {"const": "2.0"},
            "level": {"enum": ["1", "2"]},
        },
    }
}


def _call(**params: str) -> str:
    body = "".join(
        f"<parameter={k}>\n{v}\n</parameter>\n" for k, v in params.items()
    )
    return f"<tool_call>\n<function=lookup>\n{body}</function>\n</tool_call>"


def _complete_args(
    response: str, schemas: dict[str, dict[str, Any]] | None
) -> dict[str, Any]:
    parser = Qwen3_5ToolParser()
    if schemas is not None:
        parser.set_tool_schemas(schemas)
    (call,) = parser.parse_complete(response).tool_calls
    args = json.loads(call.arguments)
    assert isinstance(args, dict)
    return args


def _streamed_args(
    response: str, schemas: dict[str, dict[str, Any]] | None
) -> dict[str, Any]:
    parser = Qwen3_5ToolParser()
    if schemas is not None:
        parser.set_streaming_tool_schemas(schemas)
    pieces: list[str] = []
    # One character at a time: the hardest chunking for the streaming path.
    for ch in response:
        for delta in parser.parse_delta(ch) or []:
            if delta.arguments:
                pieces.append(delta.arguments)
    args = json.loads("".join(pieces))
    assert isinstance(args, dict)
    return args


@pytest.mark.parametrize("decode", [_complete_args, _streamed_args])
def test_string_typed_values_that_look_like_literals_stay_strings(
    decode: Any,
) -> None:
    response = _call(
        zip_code="10001", ref_zip="00501", note="None", count="7", flag="True"
    )

    assert decode(response, _SCHEMAS) == {
        "zip_code": "10001",  # a number-looking string for a string field
        "ref_zip": "00501",  # $ref to a string schema
        "note": None,  # string | null: "None" is the null literal here
        "count": 7,  # integer field: still decoded as an integer
        "flag": True,  # boolean field: still decoded as a boolean
    }


@pytest.mark.parametrize("decode", [_complete_args, _streamed_args])
def test_union_types_keep_a_scalar_only_when_its_type_is_allowed(
    decode: Any,
) -> None:
    schemas = _SCHEMAS
    # string | integer: "7" is a valid integer, "7.5" is not (so it is text).
    assert decode(_call(either="7"), schemas) == {"either": 7}
    assert decode(_call(either="7.5"), schemas) == {"either": "7.5"}
    # string | null: a number-looking value is not null, so it stays text.
    assert decode(_call(note="10001"), schemas) == {"note": "10001"}


@pytest.mark.parametrize("decode", [_complete_args, _streamed_args])
def test_string_value_is_returned_verbatim(decode: Any) -> None:
    response = _call(zip_code="  padded  ")

    assert decode(response, _SCHEMAS) == {"zip_code": "  padded  "}


@pytest.mark.parametrize("decode", [_complete_args, _streamed_args])
def test_container_values_are_decoded_from_json(decode: Any) -> None:
    response = _call(tags='["a", "007", "None"]')

    assert decode(response, _SCHEMAS) == {"tags": ["a", "007", "None"]}


@pytest.mark.parametrize("decode", [_complete_args, _streamed_args])
def test_without_schemas_decoding_is_best_effort(decode: Any) -> None:
    response = _call(zip_code="10001", flag="True", note="None")

    assert decode(response, None) == {
        "zip_code": 10001,
        "flag": True,
        "note": None,
    }


def test_unknown_function_or_parameter_falls_back_to_best_effort() -> None:
    other = {"other_tool": _SCHEMAS["lookup"]}
    assert _complete_args(_call(zip_code="10001"), other) == {"zip_code": 10001}
    assert _complete_args(_call(unlisted="5"), _SCHEMAS) == {"unlisted": 5}


def test_reset_drops_schemas_from_the_previous_request() -> None:
    parser = Qwen3_5ToolParser()
    parser.set_tool_schemas(_SCHEMAS)
    parser.reset()

    (call,) = parser.parse_complete(_call(zip_code="10001")).tool_calls

    assert json.loads(call.arguments) == {"zip_code": 10001}


# --- Constrained-decoding grammar (``generate_tool_call_grammar``) ----------

_WEATHER_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string"}},
                "required": ["city"],
            },
        },
    }
]
# One call exactly as the chat template renders it, trailing newline included.
_WEATHER_CALL = (
    "<tool_call>\n<function=get_weather>\n"
    "<parameter=city>\nParis\n</parameter>\n</function>\n</tool_call>\n"
)


# A byte-level vocabulary: one token per byte value, plus one stop token.
_STOP_TOKEN = 256
_TOKENIZER_INFO = xgrammar.TokenizerInfo(
    [bytes([i]) for i in range(256)] + [b"</s>"],
    vocab_type=xgrammar.VocabType.RAW,
    vocab_size=257,
    stop_token_ids=[_STOP_TOKEN],
)


def _accepts(
    tool_choice: str | dict[str, Any], text: str, **kwargs: Any
) -> bool:
    """Whether the grammar accepts ``text`` and could end right after it."""
    grammar = Qwen3_5ToolParser.generate_tool_call_grammar(
        tools=_WEATHER_TOOLS, tool_choice=tool_choice, **kwargs
    )
    compiled = xgrammar.GrammarCompiler(_TOKENIZER_INFO).compile_structural_tag(
        xgrammar.StructuralTag.model_validate_json(grammar)
    )
    matcher = xgrammar.GrammarMatcher(compiled)
    if not all(matcher.accept_token(byte) for byte in text.encode()):
        return False
    return bool(matcher.is_completed())


def test_required_grammar_can_stop_after_a_call_with_trailing_newline() -> None:
    # A mandatory "\n" separator read the template's trailing newline as
    # "another call must follow", so a forced call never ended.
    assert _accepts("required", _WEATHER_CALL)


def test_required_grammar_allows_several_calls() -> None:
    assert _accepts("required", _WEATHER_CALL * 2)


def test_required_grammar_is_calls_only() -> None:
    assert not _accepts("required", "")  # at least one call
    assert not _accepts("required", "Sure!\n" + _WEATHER_CALL)
    assert not _accepts("required", _WEATHER_CALL + "Thanks.")


def test_auto_grammar_allows_plain_text_and_calls() -> None:
    assert _accepts("auto", "Hello there.")
    assert _accepts("auto", "Let me check.\n" + _WEATHER_CALL)


def test_named_tool_choice_forces_exactly_that_call() -> None:
    named = {"type": "function", "function": {"name": "get_weather"}}

    # A forced call is a single tag, so the call ends at ``</tool_call>``.
    assert _accepts(named, _WEATHER_CALL.rstrip("\n"))
    assert not _accepts(named, "Hello there.")


def test_required_grammar_with_response_format_accepts_either() -> None:
    schema = {
        "type": "object",
        "properties": {"answer": {"type": "string"}},
        "required": ["answer"],
    }

    assert _accepts("required", _WEATHER_CALL, response_format_schema=schema)
    assert _accepts(
        "required", '{"answer":"42"}', response_format_schema=schema
    )


@pytest.mark.parametrize("decode", [_complete_args, _streamed_args])
def test_const_and_enum_values_take_the_type_of_the_pinned_value(
    decode: Any,
) -> None:
    # Neither schema declares a ``type``, but the pinned values are strings.
    response = _call(version="2.0", level="1")

    assert decode(response, _SCHEMAS) == {"version": "2.0", "level": "1"}


def test_self_referencing_schema_does_not_crash_and_falls_back() -> None:
    # N = string | N: a client-supplied cycle through a union must not recurse
    # without bound (it used to raise RecursionError and drop the tool call).
    cyclic = {
        "lookup": {
            "type": "object",
            "$defs": {
                "N": {"anyOf": [{"type": "string"}, {"$ref": "#/$defs/N"}]}
            },
            "properties": {"x": {"$ref": "#/$defs/N"}},
        }
    }

    assert _complete_args(_call(x="10001"), cyclic) == {"x": 10001}


@pytest.mark.parametrize("keyword", ["anyOf", "oneOf", "allOf"])
@pytest.mark.parametrize("branches", [2, 3, 5])
def test_wide_self_referencing_schema_is_cheap(
    keyword: str, branches: int
) -> None:
    # A client-supplied cycle with several branches used to be walked
    # branches**depth times; it must stay a handful of visits.
    cyclic = {
        "lookup": {
            "type": "object",
            "$defs": {"N": {keyword: [{"$ref": "#/$defs/N"}] * branches}},
            "properties": {"x": {"$ref": "#/$defs/N"}},
        }
    }

    start = time.monotonic()
    args = _complete_args(_call(x="10001"), cyclic)

    assert time.monotonic() - start < 1.0
    assert args == {"x": 10001}


@pytest.mark.parametrize("decode", [_complete_args, _streamed_args])
def test_all_of_is_an_intersection_not_a_union(decode: Any) -> None:
    schemas = {
        "lookup": {
            "type": "object",
            "properties": {
                # One typed branch and one untyped branch: still a string.
                "zip_code": {
                    "allOf": [{"type": "string"}, {"minLength": 1}],
                },
                "n": {"allOf": [{"type": "integer"}, {"minimum": 0}]},
            },
        }
    }

    assert decode(_call(zip_code="10001", n="7"), schemas) == {
        "zip_code": "10001",
        "n": 7,
    }


@pytest.mark.parametrize("decode", [_complete_args, _streamed_args])
def test_nullable_string_value_is_returned_verbatim(decode: Any) -> None:
    assert decode(_call(note="  padded  "), _SCHEMAS) == {"note": "  padded  "}


def test_grammar_rewrite_does_not_touch_user_json_schemas() -> None:
    from max.pipelines.architectures.qwen3_5.tool_parser import (
        _without_mandatory_call_separator,
    )

    # A user schema whose const happens to look like the node being rewritten.
    lookalike = {"type": "tags_with_separator", "separator": "\n"}
    node = {
        "type": "tag",
        "content": {"type": "json_schema", "json_schema": {"const": lookalike}},
        "end": "</tool_call>",
    }

    assert _without_mandatory_call_separator(node) == node


def test_required_grammar_demands_the_template_trailing_newline() -> None:
    # Each call's end tag carries the newline the chat template writes, so a
    # call without it is not a complete required call.
    assert not _accepts("required", _WEATHER_CALL.rstrip("\n"))


@pytest.mark.parametrize("decode", [_complete_args, _streamed_args])
def test_long_enumeration_of_labelled_choices_is_still_typed(
    decode: Any,
) -> None:
    # A country-list style ``oneOf`` of ``{const, title}`` choices is far more
    # than a handful of nodes and must still resolve to a string type.
    choices = [
        {"const": f"{n:03d}", "title": f"choice {n}"} for n in range(300)
    ]
    schemas = {
        "lookup": {
            "type": "object",
            "properties": {"code": {"oneOf": choices}},
        }
    }

    assert decode(_call(code="042"), schemas) == {"code": "042"}


@pytest.mark.parametrize("decode", [_complete_args, _streamed_args])
def test_enum_and_const_values_ignore_padding_whitespace(decode: Any) -> None:
    # The grammar allows whitespace around a bare value; for a pinned value
    # it is formatting, not content (a free string stays verbatim).
    response = _call(level="\t\t 1 \n\t", version="  2.0\n")

    assert decode(response, _SCHEMAS) == {"level": "1", "version": "2.0"}
