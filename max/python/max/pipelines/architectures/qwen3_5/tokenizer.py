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
"""Qwen3.5 tokenizer with thinking mode disabled by default."""

from __future__ import annotations

from typing import Any

from max.interfaces import (
    TextGenerationRequestMessage,
    TextGenerationRequestTool,
)
from max.pipelines.architectures.qwen3vl_moe.tokenizer import Qwen3VLTokenizer


class Qwen3_5Tokenizer(Qwen3VLTokenizer):
    """Tokenizer for Qwen3.5 multimodal models.

    Inherits full image-processing pipeline from :class:`Qwen3VLTokenizer` and
    enables thinking mode (`enable_thinking=True`) by default so that
    text-generation requests produce normal output instead of internal reasoning
    traces.
    """

    def apply_chat_template(
        self,
        messages: list[TextGenerationRequestMessage],
        tools: list[TextGenerationRequestTool] | None = None,
        chat_template_options: dict[str, Any] | None = None,
    ) -> str:
        """Apply chat template with thinking enabled by default.

        Args:
            messages: List of messages for the chat template.
            tools: Optional tools available for the model to invoke.
            chat_template_options: Optional dictionary of template options. Set
                `{"enable_thinking": True}` to enable thinking mode (default is True).

        Returns:
            The templated chat message as a string.
        """
        chat_template_options = chat_template_options or {}
        enable_thinking = chat_template_options.get("enable_thinking", True)

        return self.delegate.apply_chat_template(
            [msg.model_dump(exclude_none=True) for msg in messages],
            tools=tools,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=enable_thinking,
        )
