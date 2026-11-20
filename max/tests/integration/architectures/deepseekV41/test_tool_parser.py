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
# ruff: noqa: RUF001
"""DeepSeek-V4.1 DSML tool-call parsing against the checkpoint's decoder.

Completions are rendered by the vendored encoder (``dsml_fixtures``) and
checked against the vendored reference decoder.
"""

from __future__ import annotations

import json
import random
from itertools import pairwise
from typing import Any

import dsml_fixtures
import pytest
from max.pipelines.architectures.deepseekV4.tool_parser import (
    DeepseekV4ToolParser,
)
from max.pipelines.architectures.deepseekV41 import deepseekV41_arch
from max.pipelines.architectures.deepseekV41.encoding_dsv41 import (
    encode_messages,
    eos_token,
    parse_message_from_completion_text,
    thinking_end_token,
)
from max.pipelines.architectures.deepseekV41.tool_parser import (
    DeepseekV41ToolParser,
)
from max.pipelines.lib.tool_parsing import create

_ASSISTANT = dsml_fixtures.ASSISTANT
_completions = dsml_fixtures.assistant_turns


def _content(completion: str) -> str:
    """Returns what the router hands the tool parser for a turn.

    The reasoning parser keeps everything through ``</think>`` and
    detokenization drops the EOS special token.
    """
    content = completion[
        completion.index(thinking_end_token) + len(thinking_end_token) :
    ]
    return content.removesuffix(eos_token)


def _reference(completion: str) -> dict[str, Any]:
    if completion.startswith(thinking_end_token):
        return parse_message_from_completion_text(
            completion.removeprefix(thinking_end_token), thinking_mode="chat"
        )
    return parse_message_from_completion_text(
        completion.removeprefix("<think>"), thinking_mode="thinking"
    )


def _stream(text: str, cuts: list[int]) -> tuple[str, list[tuple[str, str]]]:
    """Streams ``text`` split at ``cuts``; returns content and calls."""
    parser = DeepseekV41ToolParser()
    content: list[str] = []
    names: dict[int, str] = {}
    args: dict[int, list[str]] = {}
    bounds = [0, *cuts, len(text)]
    for lo, hi in pairwise(bounds):
        for delta in parser.parse_delta(text[lo:hi]) or []:
            if delta.content:
                content.append(delta.content)
            if delta.name:
                names[delta.index] = delta.name
            if delta.arguments:
                args.setdefault(delta.index, []).append(delta.arguments)
    content.append(parser.flush() or "")
    calls = [(names[i], "".join(args.get(i, []))) for i in sorted(names)]
    return "".join(content), calls


_MULTI_CALL_ARGS = [
    {"location": "Beijing", "days": 3},
    {"query": 'a "quoted" <b>', "filters": {"lang": ["en", "zh"]}},
]


def _assistant_turn(tool_calls: list[dict[str, Any]]) -> str:
    """Renders an assistant turn holding ``tool_calls`` with the encoder."""
    messages: list[dict[str, Any]] = [
        {"role": "user", "content": "hi"},
        {
            "role": "assistant",
            "reasoning_content": "Two lookups.",
            "content": "Checking both.",
            "tool_calls": tool_calls,
        },
    ]
    prompt = encode_messages(messages, thinking_mode="thinking")
    return prompt[prompt.rindex(_ASSISTANT) + len(_ASSISTANT) :]


def _multi_call_completion() -> str:
    return _assistant_turn(
        [
            {
                "type": "function",
                "function": {"name": name, "arguments": json.dumps(a)},
            }
            for name, a in zip(
                ["get_weather", "search"], _MULTI_CALL_ARGS, strict=True
            )
        ]
    )


_TOOL_COMPLETIONS = [
    pytest.param(_completions("weather_thinking")[0], id="weather_thinking"),
    pytest.param(_completions("search_chat")[0], id="search_chat"),
    pytest.param(_multi_call_completion(), id="multi_call"),
]


def test_registered_as_architecture_default() -> None:
    assert deepseekV41_arch.tool_parser == "deepseekv41"
    assert type(create("deepseekv41")) is DeepseekV41ToolParser


@pytest.mark.parametrize("completion", _TOOL_COMPLETIONS)
def test_completions_use_v41_tags(completion: str) -> None:
    assert "<｜DSML｜ calls>" in completion
    assert "<｜DSML｜tool_calls>" not in completion


@pytest.mark.parametrize("completion", _TOOL_COMPLETIONS)
def test_parse_complete_matches_reference(completion: str) -> None:
    reference = _reference(completion)
    parsed = DeepseekV41ToolParser().parse_complete(_content(completion))
    assert parsed.content == (reference["content"] or None)
    assert [(c.name, c.arguments) for c in parsed.tool_calls] == [
        (c["function"]["name"], c["function"]["arguments"])
        for c in reference["tool_calls"]
    ]
    assert parsed.tool_calls
    assert all(c.id.startswith("call_") for c in parsed.tool_calls)


def test_weather_call_round_trips_request_arguments() -> None:
    (call,) = (
        DeepseekV41ToolParser()
        .parse_complete(_content(_completions("weather_thinking")[0]))
        .tool_calls
    )
    assert call.name == "get_weather"
    assert json.loads(call.arguments) == dsml_fixtures.WEATHER_CALL_ARGUMENTS


def test_parsed_calls_re_render_as_history() -> None:
    completion = _multi_call_completion()
    parsed = DeepseekV41ToolParser().parse_complete(_content(completion))
    assert parsed.content == "Checking both."
    assert [json.loads(c.arguments) for c in parsed.tool_calls] == (
        _MULTI_CALL_ARGS
    )
    # Sent back as the assistant's tool-call history, the parsed calls
    # render to the very turn the model produced.
    history = [
        {
            "type": "function",
            "id": c.id,
            "function": {"name": c.name, "arguments": c.arguments},
        }
        for c in parsed.tool_calls
    ]
    assert _assistant_turn(history) == completion


@pytest.mark.parametrize("name", list(dsml_fixtures.CONVERSATIONS))
def test_plain_turns_are_content(name: str) -> None:
    completion = _completions(name)[1]
    reference = _reference(completion)
    parsed = DeepseekV41ToolParser().parse_complete(_content(completion))
    assert parsed.tool_calls == []
    assert parsed.content == reference["content"]
    assert _stream(_content(completion), []) == (reference["content"], [])


def _cut_schedules(length: int) -> list[list[int]]:
    schedules = [
        list(range(step, length, step)) for step in (1, 2, 3, 5, 7, 16, 64)
    ]
    rng = random.Random(0)
    for _ in range(20):
        schedules.append(
            sorted(rng.sample(range(1, length), rng.randint(1, length // 4)))
        )
    return schedules


@pytest.mark.parametrize("completion", _TOOL_COMPLETIONS)
def test_stream_matches_complete_at_any_chunking(completion: str) -> None:
    text = _content(completion)
    parsed = DeepseekV41ToolParser().parse_complete(text)
    expected = (
        parsed.content or "",
        [(c.name, c.arguments) for c in parsed.tool_calls],
    )
    for cuts in _cut_schedules(len(text)):
        assert _stream(text, cuts) == expected, cuts


def test_stream_holds_the_open_parameter() -> None:
    text = _content(_multi_call_completion())
    open_param = text.index('<｜DSML｜ parameter name="days"')
    _, calls = _stream(text[:open_param], [])
    assert calls == [("get_weather", '{"location": "Beijing"')]


_WELL_FORMED = _content(_completions("weather_thinking")[0])


def test_namespaced_call_keeps_its_qualified_name() -> None:
    text = _WELL_FORMED.replace('"get_weather"', '"weather::get_weather"')
    (call,) = DeepseekV41ToolParser().parse_complete(text).tool_calls
    assert call.name == "weather::get_weather"
    _, calls = _stream(text, list(range(1, len(text))))
    assert calls == [(call.name, call.arguments)]


def test_v4_and_v41_tags_are_not_interchangeable() -> None:
    v4_text = (
        _WELL_FORMED.replace("｜DSML｜ calls", "｜DSML｜tool_calls")
        .replace("｜DSML｜ invoke", "｜DSML｜invoke")
        .replace("｜DSML｜ parameter", "｜DSML｜parameter")
    )
    assert DeepseekV4ToolParser().parse_complete(v4_text).tool_calls
    with pytest.raises(ValueError):
        DeepseekV41ToolParser().parse_complete(v4_text)
    with pytest.raises(ValueError):
        DeepseekV4ToolParser().parse_complete(_WELL_FORMED)


_PARAM = (
    '<｜DSML｜ parameter name="unit" string="true">celsius</｜DSML｜ parameter>'
)


@pytest.mark.parametrize(
    "text",
    [
        pytest.param(
            _WELL_FORMED.replace(
                ' string="true">celsius', ' string="yes">celsius'
            ),
            id="bad_string_attr",
        ),
        pytest.param(
            _WELL_FORMED.replace(
                'name="get_weather">\n', 'name="get_weather">'
            ),
            id="missing_header_newline",
        ),
        pytest.param(
            _WELL_FORMED.replace('"get_weather"', '"a::b::get_weather"'),
            id="nested_namespace",
        ),
        pytest.param(_WELL_FORMED + "trailing", id="content_after_block"),
        pytest.param(
            _WELL_FORMED[: _WELL_FORMED.index("</｜DSML｜ invoke>")],
            id="truncated_mid_invoke",
        ),
        pytest.param(
            "Sure." + _WELL_FORMED.removeprefix("\n\n"),
            id="block_without_blank_line",
        ),
        pytest.param(
            "\n\n<｜DSML｜ calls>\n</｜DSML｜ calls>", id="empty_block"
        ),
    ],
)
def test_malformed_dsml_raises(text: str) -> None:
    with pytest.raises(ValueError):
        DeepseekV41ToolParser().parse_complete(text)


def test_repeated_parameter_keeps_the_first_value() -> None:
    (call,) = DeepseekV41ToolParser().parse_complete(_WELL_FORMED).tool_calls
    first = json.loads(call.arguments)["location"]
    text = _WELL_FORMED.replace(_PARAM, _PARAM.replace("unit", "location"))
    (repeated,) = DeepseekV41ToolParser().parse_complete(text).tool_calls
    assert json.loads(repeated.arguments)["location"] == first
    _, streamed = _stream(text, list(range(1, len(text))))
    assert streamed == [(repeated.name, repeated.arguments)]


@pytest.mark.parametrize(
    "text",
    [
        _WELL_FORMED.replace(' string="true">celsius', ' string="yes">celsius'),
        _WELL_FORMED.replace('"get_weather"', '"a::b::get_weather"'),
    ],
    ids=["bad_string_attr", "nested_namespace"],
)
def test_malformed_dsml_streams_no_bad_arguments(text: str) -> None:
    content, calls = _stream(text, list(range(1, len(text))))
    assert content == ""
    assert all(not args.endswith("}") for _, args in calls)


def test_response_cut_off_after_the_section_tag_is_content() -> None:
    tag = DeepseekV41ToolParser.SECTION_BEGIN.rstrip("\n")
    text = "Checking.\n\n" + tag
    parsed = DeepseekV41ToolParser().parse_complete(text)
    assert parsed.content == "Checking."
    assert parsed.tool_calls == []
    content, calls = _stream(text, list(range(1, len(text))))
    assert content == "Checking."
    assert calls == []
