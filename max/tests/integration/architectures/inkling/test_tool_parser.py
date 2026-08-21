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

"""Tests for the Inkling tool-call parser.

Every input is in post-detokenization form: only
``<|content_invoke_tool_json|>`` survives, so a call reaches the parser as
``NAME<|content_invoke_tool_json|>{"name":...,"args":{...}}`` with nothing
separating one call from the next, or from preceding text.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from typing import Any

import pytest
from max.pipelines.architectures.inkling.reasoning import (
    InklingReasoningParser,
)
from max.pipelines.architectures.inkling.tokenizer import TOOL_CALL_JSON_MARKER
from max.pipelines.architectures.inkling.tool_parser import InklingToolParser
from max.pipelines.modeling.types import ParsedToolCallDelta


def _wire(name: str, args_json: str) -> str:
    """Renders one tool call exactly as the parser receives it."""
    return f'{name}{TOOL_CALL_JSON_MARKER}{{"name":{json.dumps(name)},"args":{args_json}}}'


def _schemas(*names: str) -> dict[str, dict[str, Any]]:
    """Declared tool schemas as the router supplies them; only keys matter."""
    return {n: {"type": "object", "properties": {}} for n in names}


def _assemble_streamed(
    parser: InklingToolParser, token_chunks: Sequence[str]
) -> tuple[str, list[dict[str, str]]]:
    """Reconstructs what a streaming client sees, per tool-call index."""
    content: list[str] = []
    calls: dict[int, dict[str, str]] = {}
    for chunk in token_chunks:
        result = parser.parse_delta(chunk)
        if not result:
            continue
        for delta in result:
            if delta.content is not None:
                content.append(delta.content)
                continue
            call = calls.setdefault(
                delta.index, {"id": "", "name": "", "arguments": ""}
            )
            if delta.id is not None:
                call["id"] = delta.id
            if delta.name is not None:
                call["name"] = delta.name
            if delta.arguments is not None:
                call["arguments"] += delta.arguments
    return "".join(content), [calls[i] for i in sorted(calls)]


def _assert_streaming_matches_complete(
    response: str,
    declared: Sequence[str],
    chunks: Sequence[str],
    expected_content: str | None,
) -> None:
    """Asserts streaming ``chunks`` reproduce ``parse_complete(response)``.

    Arguments compare byte for byte: both paths forward the model's own ``args``
    text, so agreeing only after a JSON round-trip would not be enough.
    """
    expected = InklingToolParser().parse_complete(response)
    assert expected.content == expected_content

    parser = InklingToolParser()
    parser.set_streaming_tool_schemas(_schemas(*declared))
    content, streamed = _assemble_streamed(parser, chunks)

    assert content == (expected.content or "")
    assert [c["name"] for c in streamed] == [
        tc.name for tc in expected.tool_calls
    ]
    assert [c["arguments"] for c in streamed] == [
        tc.arguments for tc in expected.tool_calls
    ]


_SINGLE_CALL = _wire("get_weather", '{"city":"SF"}')
_TWO_CALLS = _SINGLE_CALL + _wire("add", '{"a":2,"b":3}')
_TEXT_THEN_CALL = "Let me check both." + _SINGLE_CALL


# ---------------------------------------------------------------------------
# parse_complete
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("args_json", "expected"),
    [
        pytest.param(
            '{"city":"SF","days":3,"metric":true}',
            {"city": "SF", "days": 3, "metric": True},
            id="typed-args",
        ),
        pytest.param("{}", {}, id="empty-args"),
    ],
)
def test_parse_complete_single_call(
    args_json: str, expected: dict[str, Any]
) -> None:
    result = InklingToolParser().parse_complete(_wire("get_weather", args_json))

    assert result.content is None
    assert len(result.tool_calls) == 1

    call = result.tool_calls[0]
    assert call.name == "get_weather"
    assert call.id.startswith("call_")
    # OpenAI's contract: ``arguments`` is a JSON *string*, not an object.
    assert isinstance(call.arguments, str)
    assert json.loads(call.arguments) == expected


def test_parse_complete_unicode_args_round_trip() -> None:
    """Arguments are serialized with ``ensure_ascii=False``: no ``\\uXXXX``."""
    result = InklingToolParser().parse_complete(
        _wire("say", '{"text":"café ü 你好"}')
    )

    arguments = result.tool_calls[0].arguments
    assert "café ü 你好" in arguments
    assert "\\u" not in arguments


@pytest.mark.parametrize(
    ("payload_args", "expected"),
    [
        pytest.param('{"city":"SF"}', '{"city":"SF"}', id="compact"),
        pytest.param('{"city": "SF"}', '{"city": "SF"}', id="spaced"),
        pytest.param("5", "5", id="scalar-root"),
    ],
)
def test_parse_complete_arguments_are_the_models_own_bytes(
    payload_args: str, expected: str
) -> None:
    """``arguments`` is forwarded verbatim, not re-serialized.

    The grammar caps interior whitespace rather than forbidding it, and a tool
    without ``parameters`` maps to the unconstrained schema, so both a spaced
    object and a scalar root are legal generations. Re-serializing would
    rewrite them into something streaming never sends.
    """
    result = InklingToolParser().parse_complete(_wire("f", payload_args))

    assert result.tool_calls[0].arguments == expected


@pytest.mark.parametrize(
    ("payload", "expected_args"),
    [
        pytest.param(
            '{"name":"f", "args": {"a":1}}',
            '{"a":1}',
            id="whitespace-around-the-args-key",
        ),
        pytest.param(
            '{"name":"f","args":{"a":1},"extra":2}',
            '{"a":1}',
            id="key-after-args",
        ),
        pytest.param(
            '{"name":"f","args":{"a":1,"name":"not-the-tool"}}',
            '{"a":1,"name":"not-the-tool"}',
            id="name-property-nested-in-args",
        ),
    ],
)
def test_off_canonical_payloads_parse_the_same_either_way(
    payload: str, expected_args: str
) -> None:
    """Deviations from the canonical frame must not split the two paths.

    The grammar emits the frame as a const string, so none of these arise while
    enforcement holds. They are reachable with constrained decode disabled, or
    after a rejected token fails enforcement open mid-request -- and each used
    to yield a tool call non-streaming while streaming silently produced no
    call at all, or arguments that were not valid JSON.
    """
    response = f"f{TOOL_CALL_JSON_MARKER}{payload}"

    _assert_streaming_matches_complete(response, ("f",), list(response), None)

    result = InklingToolParser().parse_complete(response)
    assert result.tool_calls[0].arguments == expected_args


def test_parse_complete_plain_text_passes_through() -> None:
    response = "The forecast looks fine, no tools needed."
    result = InklingToolParser().parse_complete(response)

    assert result.content == response
    assert result.tool_calls == []


@pytest.mark.parametrize(
    ("args_json", "expected"),
    [
        pytest.param(
            r'{"q":"he said \"}\" and left","p":"back\\slash"}',
            {"q": 'he said "}" and left', "p": "back\\slash"},
            id="escaped-quote-and-backslash",
        ),
        pytest.param(
            r'{"q":"trailing backslash \\"}',
            {"q": "trailing backslash \\"},
            id="string-ending-in-escaped-backslash",
        ),
    ],
)
def test_parse_complete_brace_balancer_ignores_string_contents(
    args_json: str, expected: dict[str, Any]
) -> None:
    """A string ending in an escaped backslash must not swallow its quote."""
    result = InklingToolParser().parse_complete(_wire("search", args_json))

    assert json.loads(result.tool_calls[0].arguments) == expected


def test_parse_complete_brace_balancer_stops_at_the_next_call() -> None:
    """A braced string in call *n* must not run on into call *n+1*."""
    response = _wire("search", '{"q":"{{{"}') + _wire("add", '{"a":1}')
    result = InklingToolParser().parse_complete(response)

    assert [tc.name for tc in result.tool_calls] == ["search", "add"]
    assert json.loads(result.tool_calls[0].arguments) == {"q": "{{{"}
    assert json.loads(result.tool_calls[1].arguments) == {"a": 1}
    assert result.tool_calls[0].id != result.tool_calls[1].id


@pytest.mark.parametrize(
    "payload",
    [
        pytest.param('{"name":"f","args":{,,}}', id="malformed-json"),
        pytest.param('{"args":{"a":1}}', id="no-name-key"),
        pytest.param("[1,2]", id="non-object-payload"),
        pytest.param('{"name":"f","args":{"a":1}', id="truncated-json"),
    ],
)
def test_parse_complete_raises_when_the_only_call_is_unusable(
    payload: str,
) -> None:
    response = f"f{TOOL_CALL_JSON_MARKER}{payload}"

    with pytest.raises(ValueError, match=r"no valid tool calls parsed"):
        InklingToolParser().parse_complete(response)


def test_parse_complete_skips_malformed_call_beside_a_valid_one() -> None:
    response = f'f{TOOL_CALL_JSON_MARKER}{{"name":"f","args":{{,,}}}}' + _wire(
        "add", '{"a":1}'
    )
    result = InklingToolParser().parse_complete(response)

    assert [tc.name for tc in result.tool_calls] == ["add"]
    assert json.loads(result.tool_calls[0].arguments) == {"a": 1}
    # Known wart: only the first surviving call's name is trimmed, so the
    # skipped call's bare name is left behind as content.
    assert result.content == "f"


# ---------------------------------------------------------------------------
# parse_delta (streaming)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("response", "declared", "expected_content"),
    [
        pytest.param(_SINGLE_CALL, ("get_weather",), None, id="single-call"),
        pytest.param(_TWO_CALLS, ("get_weather", "add"), None, id="two-calls"),
        pytest.param(
            _TEXT_THEN_CALL,
            ("get_weather",),
            "Let me check both.",
            id="text-then-call",
        ),
        pytest.param(_wire("ping", "{}"), ("ping",), None, id="empty-args"),
    ],
)
def test_streaming_char_by_char_matches_parse_complete(
    response: str, declared: tuple[str, ...], expected_content: str | None
) -> None:
    """The general property, at the strictest possible split."""
    _assert_streaming_matches_complete(
        response, declared, list(response), expected_content
    )


def test_streaming_argument_deltas_concatenate_to_the_args_object() -> None:
    """OpenAI streaming contract: deltas carry only newly-arrived bytes and
    concatenate to exactly the payload's ``args`` object."""
    args_json = '{"city":"Chicago","unit":"fahrenheit"}'
    parser = InklingToolParser()
    parser.set_streaming_tool_schemas(_schemas("get_weather"))

    deltas: list[ParsedToolCallDelta] = []
    for chunk in [
        "get_weather",
        TOOL_CALL_JSON_MARKER,
        '{"name":"get_',
        'weather","args":{"city":',
        '"Chicago","unit":',
        '"fahrenheit"}}',
    ]:
        deltas.extend(parser.parse_delta(chunk) or [])
    argument_deltas = [d.arguments for d in deltas if d.arguments is not None]

    assert all(argument_deltas)
    assert "".join(argument_deltas) == args_json
    assert json.loads("".join(argument_deltas)) == {
        "city": "Chicago",
        "unit": "fahrenheit",
    }


def test_streaming_undeclared_tool_name_surfaces_once_as_content() -> None:
    """Documented fallback: the router omits tools without ``parameters``, so
    such a name gets no holdback and reaches the client as content."""
    content, streamed = _assemble_streamed(
        InklingToolParser(), list(_SINGLE_CALL)
    )

    assert content == "get_weather"
    assert [c["name"] for c in streamed] == ["get_weather"]


@pytest.mark.parametrize(
    ("stream", "expected_content", "expected_names"),
    [
        pytest.param(
            "Let me check" + _SINGLE_CALL,
            "Let me check",
            ["get_weather"],
            id="exact-name-trimmed-out-of-a-fused-run",
        ),
        pytest.param(
            "please get more info",
            "please get more info",
            [],
            id="prefix-released-once-it-breaks",
        ),
        pytest.param(
            "please get", "please ", [], id="prefix-dropped-at-end-of-stream"
        ),
    ],
)
def test_streaming_name_holdback(
    stream: str, expected_content: str, expected_names: list[str]
) -> None:
    """Nothing separates content from the bare name, so a trailing run that
    still prefixes a declared tool is held back: released once a character
    breaks the prefix, dropped if the stream ends first, and trimmed at the
    exact declared name when content runs straight into it.
    """
    parser = InklingToolParser()
    parser.set_streaming_tool_schemas(_schemas("get_weather"))

    content, streamed = _assemble_streamed(parser, list(stream))

    assert content == expected_content
    assert [c["name"] for c in streamed] == expected_names


def test_streaming_back_to_back_calls() -> None:
    parser = InklingToolParser()
    parser.set_streaming_tool_schemas(_schemas("get_weather", "add"))

    content, streamed = _assemble_streamed(parser, list(_TWO_CALLS))

    # The second call's bare name sits between the two payloads.
    assert content == ""
    assert [c["name"] for c in streamed] == ["get_weather", "add"]
    assert streamed[0]["id"] != streamed[1]["id"]
    assert all(call["id"].startswith("call_") for call in streamed)


def test_streaming_incomplete_call_without_arguments_is_not_surfaced() -> None:
    """Generation cut off before ``,"args":`` must not leak a dangling call."""
    parser = InklingToolParser()
    parser.set_streaming_tool_schemas(_schemas("emit"))

    chunks = list(_wire("emit", '{"v":1}')) + list(
        f'emit{TOOL_CALL_JSON_MARKER}{{"name":"emit",'
    )
    _, streamed = _assemble_streamed(parser, chunks)

    assert len(streamed) == 1
    assert json.loads(streamed[0]["arguments"]) == {"v": 1}


def test_reset_clears_declared_tool_names() -> None:
    """A second request that declares no tools must not inherit the first's."""
    parser = InklingToolParser()
    parser.set_streaming_tool_schemas(_schemas("get_weather"))
    parser.parse_delta("get_weather")

    parser.reset()

    content, streamed = _assemble_streamed(parser, list(_SINGLE_CALL))
    assert content == "get_weather"
    assert [c["name"] for c in streamed] == ["get_weather"]


# ---------------------------------------------------------------------------
# reasoning handoff
# ---------------------------------------------------------------------------


def test_reasoning_parser_hands_over_only_the_tool_call() -> None:
    """A thinking block before a call must not reach this parser.

    Runs the real reasoning parser over the token sequence, then applies the
    tokenizer's rule that every special id is dropped except the tool marker,
    which is the exact text the serving path feeds to this parser. Ids are the
    published Inkling values.
    """
    thinking, end_message, message_model, tool_json = (
        200008,
        200010,
        200001,
        200049,
    )
    body = '{"name":"get_weather","args":{"city":"SF"}}'
    text = {
        tool_json: TOOL_CALL_JSON_MARKER,
        1: "weighing the options",
        2: "get_weather",
        3: body,
    }
    tokens = [thinking, 1, end_message, message_model, 2, tool_json, 3]

    span = (
        InklingReasoningParser(
            thinking_start_token_id=thinking,
            end_message_token_id=end_message,
            tool_call_start_token_id=tool_json,
        )
        .stream(tokens, is_currently_reasoning=False)
        .span
    )

    handed_over = "".join(
        text[token] for token in span.extract_content(tokens) if token in text
    )
    assert handed_over == _wire("get_weather", '{"city":"SF"}')

    parsed = InklingToolParser().parse_complete(handed_over)
    assert parsed.content is None
    assert [(c.name, json.loads(c.arguments)) for c in parsed.tool_calls] == [
        ("get_weather", {"city": "SF"})
    ]
