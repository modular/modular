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
"""DeepSeek-V4 DSML tool-call parsing against the checkpoint's decoder.

``testdata/`` is ``encoding/tests/`` of DeepSeek-V4-Flash-0731 at revision
7872f01b1d1fe23eabc4c98b48bffcef5a386062 (MIT, Copyright (c) 2023 DeepSeek).
"""

from __future__ import annotations

import json
import os
import random
from itertools import pairwise
from pathlib import Path
from typing import Any

import pytest
from max.pipelines.architectures.deepseekV4 import deepseekV4_arch
from max.pipelines.architectures.deepseekV4.encoding_dsv4 import (
    encode_messages,
    eos_token,
    parse_message_from_completion_text,
    thinking_end_token,
)
from max.pipelines.architectures.deepseekV4.tool_parser import (
    DeepseekV4ToolParser,
)
from max.pipelines.lib.tool_parsing import create

_TESTDATA = Path(os.environ["PIPELINES_TESTDATA"])
_ASSISTANT = "<｜Assistant｜><think>"


def _completions(case: int) -> list[str]:
    """Returns each assistant turn of a vector, EOS included."""
    gold = (_TESTDATA / f"test_output_{case}.txt").read_text()
    turns = []
    start = gold.find(_ASSISTANT)
    while start != -1:
        start += len(_ASSISTANT)
        end = gold.index(eos_token, start) + len(eos_token)
        turns.append(gold[start:end])
        start = gold.find(_ASSISTANT, end)
    return turns


def _content(completion: str) -> str:
    """Returns what the router hands the tool parser for a thinking turn.

    The reasoning parser keeps everything through ``</think>`` and
    detokenization drops the EOS special token.
    """
    content = completion[
        completion.index(thinking_end_token) + len(thinking_end_token) :
    ]
    return content.removesuffix(eos_token)


def _stream(
    text: str, cuts: list[int]
) -> tuple[str, list[tuple[str, str]], DeepseekV4ToolParser]:
    """Streams ``text`` split at ``cuts``; returns content and calls."""
    parser = DeepseekV4ToolParser()
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
    return "".join(content), calls, parser


def _multi_call_completion() -> str:
    """Renders two calls with every value type through the reference encoder."""
    calls = [
        {"location": "Beijing", "days": 3},
        {"query": 'a "quoted" <b>', "filters": {"lang": ["en", "zh"]}},
    ]
    messages: list[dict[str, Any]] = [
        {"role": "user", "content": "hi"},
        {
            "role": "assistant",
            "reasoning_content": "Two lookups.",
            "content": "Checking both.",
            "tool_calls": [
                {
                    "type": "function",
                    "function": {"name": name, "arguments": json.dumps(a)},
                }
                for name, a in zip(
                    ["get_weather", "search"], calls, strict=True
                )
            ],
        },
    ]
    prompt = encode_messages(messages, thinking_mode="thinking")
    return prompt[prompt.rindex(_ASSISTANT) + len(_ASSISTANT) :]


_TOOL_COMPLETIONS = [
    pytest.param(_completions(1)[0], id="vector1"),
    pytest.param(_completions(3)[0], id="vector3"),
    pytest.param(_multi_call_completion(), id="multi_call"),
]


def test_registered_as_architecture_default() -> None:
    assert deepseekV4_arch.tool_parser == "deepseekv4"
    assert isinstance(create("deepseekv4"), DeepseekV4ToolParser)


@pytest.mark.parametrize("completion", _TOOL_COMPLETIONS)
def test_parse_complete_matches_reference(completion: str) -> None:
    reference = parse_message_from_completion_text(
        completion, thinking_mode="thinking"
    )
    parsed = DeepseekV4ToolParser().parse_complete(_content(completion))
    assert parsed.content == (reference["content"] or None)
    assert [(c.name, c.arguments) for c in parsed.tool_calls] == [
        (c["function"]["name"], c["function"]["arguments"])
        for c in reference["tool_calls"]
    ]
    assert all(c.id.startswith("call_") for c in parsed.tool_calls)


def test_vector1_round_trips_request_arguments() -> None:
    inputs = json.loads((_TESTDATA / "test_input_1.json").read_text())
    sent = inputs["messages"][2]["tool_calls"][0]["function"]
    (call,) = (
        DeepseekV4ToolParser()
        .parse_complete(_content(_completions(1)[0]))
        .tool_calls
    )
    assert call.name == sent["name"]
    assert json.loads(call.arguments) == json.loads(sent["arguments"])


def test_multi_call_types() -> None:
    parsed = DeepseekV4ToolParser().parse_complete(
        _content(_multi_call_completion())
    )
    assert parsed.content == "Checking both."
    assert [json.loads(c.arguments) for c in parsed.tool_calls] == [
        {"location": "Beijing", "days": 3},
        {"query": 'a "quoted" <b>', "filters": {"lang": ["en", "zh"]}},
    ]


@pytest.mark.parametrize("case", [1, 3])
def test_plain_turns_are_content(case: int) -> None:
    completion = _completions(case)[1]
    reference = parse_message_from_completion_text(
        completion, thinking_mode="thinking"
    )
    parsed = DeepseekV4ToolParser().parse_complete(_content(completion))
    assert parsed.tool_calls == []
    assert parsed.content == reference["content"]
    assert _stream(_content(completion), [])[:2] == (reference["content"], [])


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
    parsed = DeepseekV4ToolParser().parse_complete(text)
    expected = (
        parsed.content or "",
        [(c.name, c.arguments) for c in parsed.tool_calls],
    )
    for cuts in _cut_schedules(len(text)):
        assert _stream(text, cuts)[:2] == expected, cuts


def test_stream_holds_the_open_parameter() -> None:
    text = _content(_multi_call_completion())
    open_param = text.index('<｜DSML｜parameter name="days"')
    _, calls, _ = _stream(text[:open_param], [])
    # Only ``location`` has closed, so the object is still open.
    assert calls == [("get_weather", '{"location": "Beijing"')]


def test_dsml_inside_reasoning_is_not_a_tool_call() -> None:
    call = _content(_completions(1)[0]).removeprefix("\n\n")
    completion = (
        f"Maybe:\n\n{call}\nNo, answer directly.</think>Sunny.{eos_token}"
    )
    # The reference decoder rejects DSML inside the reasoning outright.
    with pytest.raises(AssertionError):
        parse_message_from_completion_text(completion, thinking_mode="thinking")
    # Served, it stays in the reasoning and the content has no tool call.
    parsed = DeepseekV4ToolParser().parse_complete(_content(completion))
    assert parsed.tool_calls == []
    assert parsed.content == "Sunny."
    assert _stream(_content(completion), [])[:2] == ("Sunny.", [])


def test_dsml_after_reasoning_with_dsml_inside_it() -> None:
    call = _content(_completions(1)[0])
    completion = f"Draft:{call}</think>{call}{eos_token}"
    (parsed,) = (
        DeepseekV4ToolParser().parse_complete(_content(completion)).tool_calls
    )
    assert parsed.name == "get_weather"


_WELL_FORMED = _content(_completions(1)[0])
_PARAM = (
    '<｜DSML｜parameter name="unit" string="true">celsius</｜DSML｜parameter>'
)


@pytest.mark.parametrize(
    "text",
    [
        pytest.param(
            _WELL_FORMED.replace(' string="true">celsius', ">celsius"),
            id="missing_string_attr",
        ),
        pytest.param(
            _WELL_FORMED.replace(
                ' string="true">celsius', ' string="yes">celsius'
            ),
            id="bad_string_attr",
        ),
        pytest.param(
            _WELL_FORMED.replace(_PARAM, _PARAM.replace("unit", "location")),
            id="duplicate_parameter",
        ),
        pytest.param(
            _WELL_FORMED.replace(
                'name="get_weather">\n', 'name="get_weather">'
            ),
            id="missing_header_newline",
        ),
        pytest.param(_WELL_FORMED + "trailing", id="content_after_block"),
        pytest.param(
            _WELL_FORMED[: _WELL_FORMED.index("</｜DSML｜invoke>")],
            id="truncated_mid_invoke",
        ),
        pytest.param(
            "Sure." + _WELL_FORMED.removeprefix("\n\n"),
            id="block_without_blank_line",
        ),
        pytest.param(
            "\n\n<｜DSML｜tool_calls>\n</｜DSML｜tool_calls>", id="empty_block"
        ),
    ],
)
def test_malformed_dsml_raises(text: str) -> None:
    with pytest.raises(ValueError):
        DeepseekV4ToolParser().parse_complete(text)


@pytest.mark.parametrize(
    "text",
    [
        _WELL_FORMED.replace(' string="true">celsius', ' string="yes">celsius'),
        _WELL_FORMED.replace('name="get_weather">\n', 'name="get_weather">'),
    ],
    ids=["bad_string_attr", "missing_header_newline"],
)
def test_malformed_dsml_streams_no_bad_arguments(text: str) -> None:
    content, calls, _ = _stream(text, list(range(1, len(text))))
    assert content == ""
    # Parameters before the malformed one may stream; nothing after it does.
    assert all(not args.endswith("}") for _, args in calls)
