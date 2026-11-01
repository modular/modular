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
"""Text tokenization with the existing DeepSeek-V4.1 prompt encoder."""

from __future__ import annotations

from typing import Any

from max.pipelines.context import TextContext
from max.pipelines.context.exceptions import InputError
from max.pipelines.lib.tokenizer import (
    ReasoningDelimitersMixin,
    TextTokenizer,
    _render_custom_chat_template,
)
from max.pipelines.modeling.types import (
    TextContentPart,
    TextGenerationRequest,
    TextGenerationRequestMessage,
    TextGenerationRequestTool,
)

from ..deepseekV41.tokenizer import _flatten_message, render_prompt


class DeepseekV41Tokenizer(ReasoningDelimitersMixin, TextTokenizer):
    """Uses the V4.1 prompt renderer with ordinary text contexts."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._resolve_reasoning_delimiters(self.delegate)

    def apply_chat_template(
        self,
        messages: list[TextGenerationRequestMessage],
        tools: list[TextGenerationRequestTool] | None = None,
        **chat_template_options: Any,
    ) -> str:
        """Renders the messages with the model's chat template."""
        for message in messages:
            if not isinstance(message.content, str) and any(
                not isinstance(part, TextContentPart)
                for part in message.content
            ):
                raise InputError(
                    "DeepSeek-V4.1 ModuleV3 accepts text inputs only."
                )
        if self._chat_template is None:
            return render_prompt(
                [_flatten_message(message) for message in messages],
                tools,
                chat_template_options,
            )
        return _render_custom_chat_template(
            self.delegate,
            self._chat_template,
            [message.flatten_content() for message in messages],
            tools,
            **{"add_generation_prompt": True, **chat_template_options},
        )

    async def new_context(self, request: TextGenerationRequest) -> TextContext:
        """Rejects media inputs, then defers to the base tokenizer."""
        if request.videos or request.images_for_processing():
            raise InputError("DeepSeek-V4.1 ModuleV3 accepts text inputs only.")
        return await super().new_context(request)
