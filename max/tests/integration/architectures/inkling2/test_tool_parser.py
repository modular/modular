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
"""Tests the Inkling 2 tool-call parser on payloads no grammar constrained.

Every input is post-detokenization: the checkpoint renders a call as
``<|message_model|>NAME<|content_invoke_tool_json|>{...}``, and decode keeps
only the JSON marker, so the name butts against any preceding text.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from typing import Any

import pytest
from max.pipelines.architectures.inkling2.tokenizer import (
    TOOL_CALL_JSON_MARKER,
)
from max.pipelines.architectures.inkling2.tool_parser import InklingToolParser


def _schemas(*names: str) -> dict[str, dict[str, Any]]:
    return {n: {"type": "object", "properties": {}} for n in names}


def _streamed(
    chunks: Sequence[str], declared: Sequence[str]
) -> tuple[str, list[dict[str, str]]]:
    """Returns the content and the calls a streaming client reassembles."""
    parser = InklingToolParser()
    parser.set_streaming_tool_schemas(_schemas(*declared))
    content: list[str] = []
    calls: dict[int, dict[str, str]] = {}
    for chunk in chunks:
        for delta in parser.parse_delta(chunk) or []:
            if delta.content is not None:
                content.append(delta.content)
                continue
            call = calls.setdefault(delta.index, {"name": "", "arguments": ""})
            if delta.name is not None:
                call["name"] = delta.name
            if delta.arguments is not None:
                call["arguments"] += delta.arguments
    return "".join(content), [calls[i] for i in sorted(calls)]


def _by_char(text: str) -> list[str]:
    return list(text)


@pytest.mark.parametrize(
    ("payload", "args"),
    [
        # A nested ``args`` key before the top-level one.
        (
            '{"name":"f","meta":{"args":{"wrong":1}},"args":{"right":2}}',
            {"right": 2},
        ),
        # ``name`` after ``args``.
        ('{"args":{"x":1},"name":"f"}', {"x": 1}),
        # A ``name`` nested inside ``args``.
        (
            '{"name":"f","args":{"name":"not-the-tool","x":1}}',
            {
                "name": "not-the-tool",
                "x": 1,
            },
        ),
    ],
)
def test_top_level_name_and_args_in_both_modes(
    payload: str, args: dict[str, Any]
) -> None:
    response = f"f{TOOL_CALL_JSON_MARKER}{payload}"

    complete = InklingToolParser().parse_complete(response)
    assert [tc.name for tc in complete.tool_calls] == ["f"]
    assert json.loads(complete.tool_calls[0].arguments) == args

    content, streamed = _streamed(_by_char(response), declared=["f"])
    assert content == ""
    assert [c["name"] for c in streamed] == ["f"]
    assert json.loads(streamed[0]["arguments"]) == args


@pytest.mark.parametrize(
    "chunks",
    [
        ["Let me checkget_", "weather"],
        ["Let me check", "get_", "weather"],
        ["Let me checkget_weath", "er"],
    ],
)
def test_text_running_into_a_name_never_leaks_it(chunks: list[str]) -> None:
    payload = '{"name":"get_weather","args":{"city":"Paris"}}'
    tail = TOOL_CALL_JSON_MARKER + payload
    response = "".join(chunks) + tail

    complete = InklingToolParser().parse_complete(response)
    assert complete.content == "Let me check"

    content, streamed = _streamed([*chunks, tail], declared=["get_weather"])
    assert content == "Let me check"
    assert [c["name"] for c in streamed] == ["get_weather"]
    assert json.loads(streamed[0]["arguments"]) == {"city": "Paris"}


def test_text_that_only_looks_like_a_name_prefix_is_released() -> None:
    content, streamed = _streamed(
        ["I'll get_", "going now."], declared=["get_weather"]
    )
    assert content == "I'll get_going now."
    assert streamed == []
