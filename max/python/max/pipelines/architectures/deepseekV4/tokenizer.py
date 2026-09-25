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

"""DeepSeek-V4 tokenizer rendering prompts with the checkpoint's encoder."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any

from max.pipelines.lib import TextTokenizer
from max.pipelines.lib.tokenizer import resolve_single_special_token
from max.pipelines.modeling.types import (
    TextGenerationRequestMessage,
    TextGenerationRequestTool,
)

from .encoding_dsv4 import encode_messages

if TYPE_CHECKING:
    from max.pipelines.lib.config import PipelineConfig

_THINK_START_TOKEN = "<think>"
_THINK_END_TOKEN = "</think>"


def render_prompt(
    messages: Sequence[Mapping[str, Any]],
    tools: Sequence[TextGenerationRequestTool] | None,
    chat_template_options: Mapping[str, Any],
) -> str:
    """Renders an OpenAI-style conversation into a DeepSeek-V4 prompt.

    Thinking defaults to on and reasoning effort to ``high`` (the encoder's
    own default is ``low``); both match vLLM's DeepSeek-V4 tokenizer. A
    request turns thinking off with ``thinking``/``enable_thinking`` false or
    ``reasoning_effort="none"``.

    Args:
        messages: Flattened messages (``role``, ``content`` and the optional
            ``tool_calls``, ``tool_call_id``, ``reasoning_content`` keys).
        tools: OpenAI-format tool definitions, attached to the leading system
            message (one is inserted when the conversation has none).
        chat_template_options: Request chat-template kwargs. Reads
            ``thinking``, ``enable_thinking``, ``reasoning_effort`` and
            ``drop_thinking``.

    Returns:
        The prompt string, BOS included.

    Raises:
        ValueError: If the encoder rejects the conversation.
    """
    if (
        "thinking" in chat_template_options
        or "enable_thinking" in chat_template_options
    ):
        thinking = bool(chat_template_options.get("thinking")) or bool(
            chat_template_options.get("enable_thinking")
        )
    else:
        thinking = True

    effort = chat_template_options.get("reasoning_effort")
    reasoning_effort: str | None
    if not isinstance(effort, str):
        reasoning_effort = "high" if thinking else None
    elif effort == "none":
        thinking = False
        reasoning_effort = None
    elif effort == "max":
        reasoning_effort = "max"
    elif effort in ("low", "minimal", "medium"):
        reasoning_effort = "low"
    else:
        reasoning_effort = "high"

    conversation = [dict(m) for m in messages]
    if tools:
        if not conversation or conversation[0].get("role") != "system":
            conversation.insert(0, {"role": "system", "content": ""})
        conversation[0]["tools"] = list(tools)

    try:
        return encode_messages(
            conversation,
            thinking_mode="thinking" if thinking else "chat",
            drop_thinking=bool(
                chat_template_options.get("drop_thinking", True)
            ),
            reasoning_effort=reasoning_effort,
        )
    except (AssertionError, NotImplementedError) as e:
        # The reference encoder signals unsupported input with these; surface
        # them as a request error rather than a server fault.
        raise ValueError(f"DeepSeek-V4 prompt encoding failed: {e}") from e


class DeepseekV4Tokenizer(TextTokenizer):
    """:class:`TextTokenizer` for DeepSeek-V4, whose checkpoint ships no chat template.

    Chat prompts are rendered by the checkpoint's reference encoder
    (``encoding_dsv4.py``) unless ``--chat-template`` is given. Implements
    :class:`~max.pipelines.modeling.types.ReasoningPipelineTokenizer`.
    """

    def __init__(
        self,
        model_path: str,
        pipeline_config: PipelineConfig,
        **kwargs,
    ) -> None:
        super().__init__(model_path, pipeline_config, **kwargs)
        self._reasoning_start_token_id: int = resolve_single_special_token(
            self.delegate, _THINK_START_TOKEN
        )
        self._reasoning_end_token_id: int = resolve_single_special_token(
            self.delegate, _THINK_END_TOKEN
        )

    @property
    def reasoning_start_token_id(self) -> int:
        """Token id of ``<think>``."""
        return self._reasoning_start_token_id

    @property
    def reasoning_end_token_id(self) -> int:
        """Token id of ``</think>``."""
        return self._reasoning_end_token_id

    def apply_chat_template(
        self,
        messages: list[TextGenerationRequestMessage],
        tools: list[TextGenerationRequestTool] | None,
        **chat_template_options: Any,
    ) -> str:
        """Renders messages with the DeepSeek-V4 encoder (see :func:`render_prompt`)."""
        if self._custom_template_provided:
            return super().apply_chat_template(
                messages, tools, **chat_template_options
            )
        return render_prompt(
            [message.flatten_content() for message in messages],
            tools,
            chat_template_options,
        )
