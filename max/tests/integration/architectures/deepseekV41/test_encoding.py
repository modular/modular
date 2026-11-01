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
"""Tests flattening request messages and rendering the V4.1 prompt."""

from __future__ import annotations

from typing import Any

from max.pipelines.architectures.deepseekV41.encoding_dsv41 import (
    ASSISTANT_SP_TOKEN,
)
from max.pipelines.architectures.deepseekV41.tokenizer import (
    _flatten_message,
    render_prompt,
)
from max.pipelines.modeling.types import TextGenerationRequestMessage

_MULTI_PART_MESSAGE = {
    "role": "user",
    "content": [
        {"type": "text", "text": "first part"},
        {"type": "text", "text": "second part"},
    ],
}


def test_flatten_message_joins_multi_part_text_with_blank_line() -> None:
    """Checks multi-part text joins with a blank line (``\\n\\n``)."""
    message = TextGenerationRequestMessage.model_validate(_MULTI_PART_MESSAGE)
    flattened = _flatten_message(message)
    assert flattened["content"] == "first part\n\nsecond part"


def test_render_prompt_renders_flattened_multi_part_message() -> None:
    message = TextGenerationRequestMessage.model_validate(_MULTI_PART_MESSAGE)
    prompt = render_prompt([_flatten_message(message)], None, {})
    assert "first part\n\nsecond part" in prompt


_TOOL_TURN: list[dict[str, Any]] = [
    {"role": "user", "content": "q1"},
    {"role": "assistant", "content": "a1", "reasoning_content": "r1"},
    {"role": "user", "content": "q2"},
    {
        "role": "assistant",
        "content": "",
        "reasoning_content": "r2",
        "tool_calls": [
            {
                "id": "c1",
                "type": "function",
                "function": {"name": "lookup", "arguments": "{}"},
            }
        ],
    },
    {"role": "tool", "tool_call_id": "c1", "content": "value"},
]


def test_render_prompt_keeps_current_turn_reasoning_across_tool_results() -> (
    None
):
    """Checks a tool result does not start a turn that drops its reasoning."""
    prompt = render_prompt(_TOOL_TURN, None, {})
    assert "r1" not in prompt
    assert "<think>r2</think>" in prompt
    assert prompt.endswith(
        f"<tool_result>value</tool_result>{ASSISTANT_SP_TOKEN}<think>"
    )


def test_render_prompt_drops_tool_turn_reasoning_after_next_user() -> None:
    prompt = render_prompt(
        [*_TOOL_TURN, {"role": "user", "content": "q3"}], None, {}
    )
    assert "r2" not in prompt
    assert prompt.endswith(f"q3{ASSISTANT_SP_TOKEN}<think>")
