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
"""DeepSeek-V4 prompt rendering against the checkpoint's own test vectors.

``testdata/`` is ``encoding/tests/`` of DeepSeek-V4-Flash-0731 at revision
7872f01b1d1fe23eabc4c98b48bffcef5a386062 (MIT, Copyright (c) 2023 DeepSeek).
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import pytest
from max.pipelines.architectures.deepseekV4.encoding_dsv4 import (
    REASONING_EFFORT_PROMPTS,
    bos_token,
    encode_messages,
    parse_message_from_completion_text,
)
from max.pipelines.architectures.deepseekV4.tokenizer import render_prompt
from max.pipelines.modeling.types import TextGenerationRequestMessage

_TESTDATA = Path(os.environ["PIPELINES_TESTDATA"])

# The reference encoder's own defaults, which the vectors were rendered with.
_REFERENCE_THINKING = {
    "enable_thinking": True,
    "thinking": True,
    "reasoning_effort": "low",
}


def _load(case: int) -> tuple[Any, str]:
    inputs = json.loads((_TESTDATA / f"test_input_{case}.json").read_text())
    gold = (_TESTDATA / f"test_output_{case}.txt").read_text()
    return inputs, gold


def _flattened(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Round-trips messages through the serving request type."""
    return [
        TextGenerationRequestMessage.model_validate(m).flatten_content()
        for m in messages
    ]


@pytest.mark.parametrize(
    ("case", "thinking_mode"),
    [(1, "thinking"), (2, "thinking"), (3, "thinking"), (4, "chat")],
)
def test_vendored_encoder_matches_vectors(
    case: int, thinking_mode: str
) -> None:
    inputs, gold = _load(case)
    if isinstance(inputs, dict):
        messages = inputs["messages"]
        messages[0]["tools"] = inputs["tools"]
    else:
        messages = inputs
    assert encode_messages(messages, thinking_mode=thinking_mode) == gold


def test_render_prompt_tools_and_tool_results() -> None:
    inputs, gold = _load(1)
    prompt = render_prompt(
        _flattened(inputs["messages"]), inputs["tools"], _REFERENCE_THINKING
    )
    assert prompt == gold


def test_render_prompt_multi_turn_drops_earlier_reasoning() -> None:
    messages, gold = _load(2)
    prompt = render_prompt(_flattened(messages), None, _REFERENCE_THINKING)
    assert prompt == gold


def test_render_prompt_defaults_to_thinking_at_high_effort() -> None:
    messages, gold = _load(2)
    prompt = render_prompt(_flattened(messages), None, {})
    assert prompt == (
        bos_token + REASONING_EFFORT_PROMPTS["high"] + gold[len(bos_token) :]
    )


@pytest.mark.parametrize(
    "options",
    [
        {"enable_thinking": False},
        {"thinking": False},
        {"reasoning_effort": "none"},
    ],
)
def test_render_prompt_chat_mode(options: dict[str, Any]) -> None:
    messages, _ = _load(2)
    prompt = render_prompt(_flattened(messages[:-1]), None, options)
    assert prompt.endswith(
        "<｜User｜>What is the capital of France?<｜Assistant｜></think>"
    )
    assert "Reasoning Effort" not in prompt
    assert "<think>" not in prompt


@pytest.mark.parametrize(
    ("effort", "prefix"),
    [
        ("minimal", "low"),
        ("low", "low"),
        ("medium", "low"),
        ("high", "high"),
        ("xhigh", "high"),
        ("max", "max"),
    ],
)
def test_render_prompt_effort_mapping(effort: str, prefix: str) -> None:
    messages = [{"role": "user", "content": "hi"}]
    prompt = render_prompt(messages, None, {"reasoning_effort": effort})
    assert prompt == (
        bos_token
        + REASONING_EFFORT_PROMPTS[prefix]
        + "<｜User｜>hi<｜Assistant｜><think>"
    )


def test_render_prompt_inserts_system_for_tools() -> None:
    inputs, _ = _load(1)
    prompt = render_prompt(
        [{"role": "user", "content": "hi"}],
        inputs["tools"],
        _REFERENCE_THINKING,
    )
    assert prompt.startswith(bos_token + "\n\n## Tools\n\n")
    assert prompt.endswith("<｜User｜>hi<｜Assistant｜><think>")


def test_render_prompt_rejects_unknown_role() -> None:
    with pytest.raises(ValueError, match="DeepSeek-V4 prompt encoding"):
        render_prompt([{"role": "function", "content": "x"}], None, {})


def test_vector_completions_parse_as_reasoning_then_content() -> None:
    _, gold = _load(2)
    marker = "<｜Assistant｜><think>"
    completion = gold[gold.rfind(marker) + len(marker) :]
    parsed = parse_message_from_completion_text(
        completion, thinking_mode="thinking"
    )
    assert parsed["reasoning_content"] == (
        "The user asks about the capital of France. It is Paris."
    )
    assert parsed["content"] == "The capital of France is Paris."
