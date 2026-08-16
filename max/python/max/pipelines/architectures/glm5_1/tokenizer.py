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

"""GLM-4.5+ text tokenizer that exposes its reasoning-delimiter token ids."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from max.pipelines.lib.config import PipelineConfig
from max.pipelines.lib.tokenizer import (
    TextTokenizer,
    resolve_single_special_token,
)
from max.pipelines.modeling.types import (
    TextGenerationRequestMessage,
    TextGenerationRequestTool,
)

_THINK_START_TOKEN = "<think>"
_THINK_END_TOKEN = "</think>"

# GLM's chat template exposes only two thinking levels. It renders
# "Reasoning Effort: High" when ``reasoning_effort`` is exactly ``"high"`` and
# "Reasoning Effort: Max" for every other value, an absent one included:
#
#     {%- set effective_reasoning_effort =
#         'high' if reasoning_effort is defined and reasoning_effort == 'high'
#         else 'max' -%}
#
# Remap OpenAI's ladder to GLM's two template rungs.
#   "none" -> thinking off
#   "low" -> "high"
#   "medium" -> "high"
#   "high" -> "high"
#   "xhigh" -> "max"
#   "max" or unset -> "max"
# Anything unrecognized joins the ladder on "high", so a bad value degrades to
# less reasoning rather than silently maxing out.
#
# ``xhigh`` is OpenRouter's name for the upper rung. Its model catalogue
# advertises exactly two efforts for this model, matching the template:
#
#     "z-ai/glm-5.2": {"supported_efforts": ["xhigh", "high"],
#                      "default_effort": "high", "default_enabled": true}
#
# so an OpenRouter-shaped client has no other way to ask for GLM's top rung,
# and MAX Serve already routes ``reasoning.effort`` here.
_GLM_EFFORT_HIGH = "high"
_GLM_EFFORT_MAX = "max"
_TOP_RUNG_ALIASES = frozenset({_GLM_EFFORT_MAX, "xhigh"})


def normalize_glm_reasoning_effort(
    chat_template_options: Mapping[str, Any],
) -> dict[str, Any]:
    """Rewrites an OpenAI ``reasoning_effort`` onto GLM's two template rungs.

    An effort of ``"none"`` is expressed by disabling thinking outright, since
    the template drops the effort line entirely once the toggle is off. MAX
    Serve already settles that toggle for requests it resolves, but callers
    that pass ``chat_template_kwargs`` straight through (benchmark configs, for
    instance) bypass that path, so it is settled here too when the caller left
    it unset.

    Both spellings of the upper rung are accepted: GLM's native ``"max"`` and
    OpenRouter's ``"xhigh"``, the only value an OpenRouter-shaped client can
    send to reach it.

    Args:
        chat_template_options: Keyword arguments bound for the chat template.

    Returns:
        A copy with ``reasoning_effort`` translated to the value GLM's template
        reads, and the thinking toggle settled for an effort of ``"none"``.
    """
    options = dict(chat_template_options)
    effort = options.get("reasoning_effort")
    if not isinstance(effort, str):
        return options

    normalized = effort.strip().lower()
    if normalized == "none":
        if "enable_thinking" not in options and "thinking" not in options:
            options["enable_thinking"] = False
            options["thinking"] = False
    elif normalized in _TOP_RUNG_ALIASES:
        options["reasoning_effort"] = _GLM_EFFORT_MAX
    else:
        options["reasoning_effort"] = _GLM_EFFORT_HIGH
    return options


class GlmTokenizer(TextTokenizer):
    """Text tokenizer for GLM-4.5+ (GLM-5.1 / GLM-5.2).

    Overridden to apply reasoning parsing normalization to the chat template,
    and remap reasoning effort to GLM's template.
    """

    def __init__(
        self,
        model_path: str,
        pipeline_config: PipelineConfig,
        *,
        revision: str | None = None,
        max_length: int | None = None,
        trust_remote_code: bool = False,
        enable_llama_whitespace_fix: bool = False,
        chat_template: str | None = None,
        **unused_kwargs: Any,
    ) -> None:
        super().__init__(
            model_path,
            pipeline_config,
            revision=revision,
            max_length=max_length,
            trust_remote_code=trust_remote_code,
            enable_llama_whitespace_fix=enable_llama_whitespace_fix,
            chat_template=chat_template,
            **unused_kwargs,
        )
        self._reasoning_start_token_id: int = resolve_single_special_token(
            self.delegate, _THINK_START_TOKEN
        )
        self._reasoning_end_token_id: int = resolve_single_special_token(
            self.delegate, _THINK_END_TOKEN
        )

    def apply_chat_template(
        self,
        messages: list[TextGenerationRequestMessage],
        tools: list[TextGenerationRequestTool] | None,
        **chat_template_options: Any,
    ) -> str:
        """Applies the GLM chat template, first normalizing the effort ladder."""
        return super().apply_chat_template(
            messages,
            tools,
            **normalize_glm_reasoning_effort(chat_template_options),
        )

    @property
    def reasoning_start_token_id(self) -> int:
        """Token id of ``<think>`` (opens a GLM reasoning span)."""
        return self._reasoning_start_token_id

    @property
    def reasoning_end_token_id(self) -> int:
        """Token id of ``</think>`` (closes a GLM reasoning span)."""
        return self._reasoning_end_token_id
