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

"""The checkpoint's reasoning-section parser."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, ClassVar

from max.pipelines.lib.reasoning import register
from max.pipelines.lib.tokenizer import convert_token_to_id
from max.pipelines.modeling.types import (
    ParsedReasoningDelta,
    PipelineTokenizer,
    ReasoningParser,
    ReasoningSpan,
)


@register("inkling2")
class InklingReasoningParser(ReasoningParser):
    """Inkling reasoning parser for ``<|content_thinking|>...<|end_message|>``."""

    REASONING_START: ClassVar[str] = "<|content_thinking|>"
    REASONING_END: ClassVar[str] = "<|end_message|>"

    def __init__(
        self,
        thinking_start_token_id: int,
        end_message_token_id: int,
        tool_call_start_token_id: int | None = None,
    ) -> None:
        self.thinking_start_token_id = thinking_start_token_id
        self.end_message_token_id = end_message_token_id
        self.tool_call_start_token_id = tool_call_start_token_id
        self._end_token_ids = (
            (end_message_token_id, tool_call_start_token_id)
            if tool_call_start_token_id is not None
            else (end_message_token_id,)
        )

    def stream(
        self,
        delta_token_ids: Sequence[int],
        is_currently_reasoning: bool = True,
    ) -> ParsedReasoningDelta:
        """Identifies a reasoning span within a streaming delta chunk."""
        start_token_idx: int | None = None
        end_token_idx: int | None = None
        for i, token_id in enumerate(delta_token_ids):
            if (
                start_token_idx is None
                and token_id == self.thinking_start_token_id
            ):
                start_token_idx = i
            elif token_id in self._end_token_ids:
                # Honor an end delimiter only while a span is active: every
                # Inkling message type ends with ``<|end_message|>``, so a
                # closer outside a span must not open one.
                if is_currently_reasoning or start_token_idx is not None:
                    end_token_idx = i
                    break

        if start_token_idx is None and not is_currently_reasoning:
            # No opener and not already reasoning: all content. Inkling
            # prefills no thinking marker, so there is no implicit start.
            empty_span = ReasoningSpan(
                reasoning_with_delimiters=(0, 0),
                reasoning=(0, 0),
            )
            return ParsedReasoningDelta(
                span=empty_span, is_still_reasoning=False
            )

        if start_token_idx is None:
            start_reasoning = 0
            start_reasoning_with_delimiters = 0
        else:
            start_reasoning = start_token_idx + 1
            start_reasoning_with_delimiters = start_token_idx

        if end_token_idx is None:
            end_reasoning = len(delta_token_ids)
            end_reasoning_with_delimiters = len(delta_token_ids)
        elif delta_token_ids[end_token_idx] == self.end_message_token_id:
            # <|end_message|> is consumed as a delimiter.
            end_reasoning = end_token_idx
            end_reasoning_with_delimiters = end_token_idx + 1
        else:
            # <|content_invoke_tool_json|> is not consumed: it belongs to
            # the content region where downstream tool parsing handles it.
            end_reasoning = end_token_idx
            end_reasoning_with_delimiters = end_token_idx

        span = ReasoningSpan(
            reasoning_with_delimiters=(
                start_reasoning_with_delimiters,
                end_reasoning_with_delimiters,
            ),
            reasoning=(start_reasoning, end_reasoning),
        )
        is_still_reasoning = end_token_idx is None
        return ParsedReasoningDelta(
            span=span, is_still_reasoning=is_still_reasoning
        )

    def will_reason_after_prompt(
        self,
        prompt_token_ids: Sequence[int],
    ) -> bool:
        """Predicts whether generation starts inside a reasoning span."""
        for token_id in reversed(prompt_token_ids):
            if token_id == self.thinking_start_token_id:
                return True
            if token_id in self._end_token_ids:
                return False
        return False

    @classmethod
    async def from_tokenizer(
        cls,
        tokenizer: PipelineTokenizer[Any, Any, Any],
    ) -> InklingReasoningParser:
        """Constructs a reasoning parser from a tokenizer."""
        thinking_start_id = await convert_token_to_id(
            tokenizer, cls.REASONING_START
        )
        end_message_id = await convert_token_to_id(tokenizer, cls.REASONING_END)

        if thinking_start_id is None or end_message_id is None:
            raise ValueError(
                f"{cls.__name__} could not locate thinking start/end message"
                " tokens in the tokenizer"
            )

        tool_call_start_id = await convert_token_to_id(
            tokenizer, "<|content_invoke_tool_json|>"
        )

        return cls(
            thinking_start_token_id=thinking_start_id,
            end_message_token_id=end_message_id,
            tool_call_start_token_id=tool_call_start_id,
        )

    @classmethod
    async def reasoning_end_token_id(
        cls,
        tokenizer: PipelineTokenizer[Any, Any, Any],
    ) -> int | None:
        """Returns the ``<|end_message|>`` token id that closes reasoning."""
        return await convert_token_to_id(tokenizer, cls.REASONING_END)
