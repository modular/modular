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

"""Hy4 text tokenizer: think delimiters and ``reasoning_effort`` mapping."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, ClassVar

from max.pipelines.lib.tokenizer import ReasoningTextTokenizer
from max.pipelines.modeling.types import (
    TextGenerationRequestMessage,
    TextGenerationRequestTool,
)

from .reasoning import HYV4ReasoningParser

_THINKING_TOGGLES = ("enable_thinking", "thinking")


def normalize_hy4_reasoning_effort(
    chat_template_options: Mapping[str, Any],
) -> dict[str, Any]:
    """Maps a request's reasoning controls onto the template's two efforts.

    Hy4's template accepts only ``high`` and ``no_think`` and raises on any
    other value, while MAX folds OpenAI's ``reasoning_effort`` ladder and the
    ``enable_thinking`` toggle into the template kwargs. ``no_think``,
    ``none`` or a false toggle select ``no_think``; any other effort selects
    ``high``; an unset effort is left to the template's own ``high`` default.
    """
    options = dict(chat_template_options)
    toggles = [options.pop(key) for key in _THINKING_TOGGLES if key in options]
    effort = options.get("reasoning_effort")
    if effort in ("no_think", "none") or any(not t for t in toggles):
        options["reasoning_effort"] = "no_think"
    elif effort is not None:
        options["reasoning_effort"] = "high"
    return options


class HYV4Tokenizer(ReasoningTextTokenizer):
    """Text tokenizer for Hy4."""

    reasoning_delimiters: ClassVar[tuple[str, str]] = (
        HYV4ReasoningParser.REASONING_START,
        HYV4ReasoningParser.REASONING_END,
    )

    def apply_chat_template(
        self,
        messages: list[TextGenerationRequestMessage],
        tools: list[TextGenerationRequestTool] | None,
        **chat_template_options: Any,
    ) -> str:
        """Applies the Hy4 chat template with a valid ``reasoning_effort``."""
        return super().apply_chat_template(
            messages,
            tools,
            **normalize_hy4_reasoning_effort(chat_template_options),
        )
