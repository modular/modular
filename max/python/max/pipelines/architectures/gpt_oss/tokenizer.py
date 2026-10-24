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

"""gpt-oss tokenizer that maps reasoning effort onto the levels it knows."""

from __future__ import annotations

from typing import Any

from max.pipelines.context.exceptions import InputError
from max.pipelines.lib import TextTokenizer
from max.pipelines.modeling.types import (
    TextGenerationRequestMessage,
    TextGenerationRequestTool,
)

# gpt-oss renders ``Reasoning: <word>`` with whatever word it is given, so the
# OpenAI ladder lands on the three it was trained on, as the orchestrator's
# gpt-oss contract maps it. It has no word for "do not reason", so ``none`` is
# refused, as sglang and vLLM refuse it.
_EFFORT_MAP = {
    "minimal": "low",
    "low": "low",
    "medium": "medium",
    "high": "high",
    "xhigh": "high",
    "max": "high",
}


def normalize_gpt_oss_reasoning_effort(
    options: dict[str, Any],
) -> dict[str, Any]:
    """Maps ``reasoning_effort`` onto gpt-oss's levels.

    Raises:
        InputError: If ``reasoning_effort`` is ``"none"``.
    """
    effort = options.get("reasoning_effort")
    if not isinstance(effort, str):
        return options
    word = effort.strip().lower()
    if word == "none":
        raise InputError(
            'reasoning_effort "none" is not supported; supported values are'
            " low, medium, high"
        )
    if word not in _EFFORT_MAP:
        return options
    return {**options, "reasoning_effort": _EFFORT_MAP[word]}


class GptOssTokenizer(TextTokenizer):
    """Text tokenizer that maps ``reasoning_effort`` onto gpt-oss's levels."""

    def apply_chat_template(
        self,
        messages: list[TextGenerationRequestMessage],
        tools: list[TextGenerationRequestTool] | None,
        **chat_template_options: Any,
    ) -> str:
        return super().apply_chat_template(
            messages,
            tools,
            **normalize_gpt_oss_reasoning_effort(chat_template_options),
        )
