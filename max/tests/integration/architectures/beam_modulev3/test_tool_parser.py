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
"""Beam's tool parser, over the content its reasoning parser hands it."""

from __future__ import annotations

import json

from max.pipelines.architectures.beam_modulev3 import harmony
from max.pipelines.architectures.beam_modulev3.tool_parser import BeamToolParser


def _call(name: str, *params: tuple[str, str]) -> str:
    body = "\n".join(
        f'<rfl:parameter name="{key}">\n{value}\n</rfl:parameter>'
        for key, value in params
    )
    return f" to={name}<|channel|>analysis <|constrain|>xml<|message|>{body}"


_NEXT = "<|end|><|start|>assistant"
_SCHEMAS = {
    "get_weather": {
        "type": "object",
        "properties": {
            "city": {"type": "string"},
            "days": {"type": "integer"},
            "detailed": {"type": "boolean"},
            "tags": {"type": "array"},
            "note": {"type": ["string", "null"]},
        },
    }
}

# What the reasoning parser routes to content for a turn that answers, calls
# two tools in parallel, then answers again.
_CONTENT = (
    "Checking."
    + _call(
        "get_weather",
        ("city", "Paris"),
        ("days", "3"),
        ("detailed", "true"),
        ("tags", '["a", "b"]'),
        ("note", "line one\nline two"),
    )
    + _NEXT
    + _call("search", ("query", '{"looks": "like json"}'), ("limit", "5"))
    + "<|call|>"
)
_EXPECTED = [
    (
        "get_weather",
        {
            "city": "Paris",
            "days": 3,
            "detailed": True,
            "tags": ["a", "b"],
            "note": "line one\nline two",
        },
    ),
    # No schema: every value stays the text the model wrote.
    ("search", {"query": '{"looks": "like json"}', "limit": "5"}),
]


def test_parse_complete_types_values_by_schema() -> None:
    parser = BeamToolParser()
    parser.set_tool_schemas(_SCHEMAS)
    parsed = parser.parse_complete(_CONTENT)
    assert parsed.content == "Checking."
    assert [
        (c.name, json.loads(c.arguments)) for c in parsed.tool_calls
    ] == _EXPECTED


def _deltas(content: str) -> list[str]:
    """One character per delta, except that a header arrives whole, as the
    reasoning parser releases a header's tokens together."""
    out: list[str] = []
    cursor = 0
    for header in harmony.TOOL_HEADER.finditer(content):
        out.extend(content[cursor : header.start()])
        out.append(header.group())
        cursor = header.end()
    out.extend(content[cursor:])
    return out


def test_streamed_calls_concatenate_to_the_complete_parse() -> None:
    parser = BeamToolParser()
    parser.set_streaming_tool_schemas(_SCHEMAS)
    names: dict[int, str] = {}
    arguments: dict[int, str] = {}
    content = ""
    for piece in _deltas(_CONTENT):
        for delta in parser.parse_delta(piece) or []:
            content += delta.content or ""
            if delta.name:
                names[delta.index] = delta.name
            arguments[delta.index] = arguments.get(delta.index, "") + (
                delta.arguments or ""
            )
    complete = BeamToolParser()
    complete.set_tool_schemas(_SCHEMAS)
    expected = complete.parse_complete(_CONTENT)
    assert content == expected.content
    assert [(names[i], arguments[i]) for i in sorted(names)] == [
        (c.name, c.arguments) for c in expected.tool_calls
    ]


def test_streams_each_parameter_as_it_closes() -> None:
    parser = BeamToolParser()
    deltas = parser.parse_delta(
        "Hi."
        + _call("search", ("query", "q"))
        + '\n<rfl:parameter name="limit">\n5'
    )
    assert [(d.content, d.name, d.arguments) for d in deltas or []] == [
        ("Hi.", None, None),
        (None, "search", "{"),
        (None, None, '"query": "q"'),
    ]
    deltas = parser.parse_delta("\n</rfl:parameter><|call|>")
    assert [d.arguments for d in deltas or []] == [', "limit": "5"}']


def test_finish_closes_a_call_the_stream_ended_inside() -> None:
    # The last call never decodes a close, as when the model stops on an
    # end-of-sequence token with no text; `c` never closes either.
    content = (
        _call("f", ("a", "1"))
        + _NEXT
        + _call("g", ("b", "2"))
        + '\n<rfl:parameter name="c">\nunfinished'
    )
    parser = BeamToolParser()
    arguments: dict[int, str] = {}
    for delta in [
        d for piece in _deltas(content) for d in parser.parse_delta(piece) or []
    ] + parser.finish():
        arguments[delta.index] = arguments.get(delta.index, "") + (
            delta.arguments or ""
        )
    expected = BeamToolParser().parse_complete(content)
    assert [arguments[i] for i in sorted(arguments)] == [
        c.arguments for c in expected.tool_calls
    ]
    assert [json.loads(a) for a in arguments.values()] == [
        {"a": "1"},
        {"b": "2"},
    ]
    assert parser.finish() == []


def test_malformed_parameters_are_skipped() -> None:
    parsed = BeamToolParser().parse_complete(
        _call("f")
        + '<rfl:parameter name="a">no newline</rfl:parameter>'
        + '<rfl:parameter name="b">\nkept\n<rfl:parameter name="c">\nlast'
    )
    # `b` ends at the next parameter's open; `c` never closes.
    assert [json.loads(c.arguments) for c in parsed.tool_calls] == [
        {"b": "kept"}
    ]


def test_text_without_a_call_is_content() -> None:
    parsed = BeamToolParser().parse_complete("I want to go to=the store.")
    assert parsed.tool_calls == []
    assert parsed.content == "I want to go to=the store."
