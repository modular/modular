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

"""MiniMax-M2 reasoning parser for <think>...</think> sections."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from max.interfaces import (
    ParsedReasoningDelta,
    PipelineTokenizer,
    ReasoningParser,
    ReasoningSpan,
)
from max.pipelines.lib.reasoning import register
from max.pipelines.lib.tokenizer import convert_token_to_id


@register("minimax_m2")
class MiniMaxM2ReasoningParser(ReasoningParser):
    """MiniMax-M2 reasoning parser for <think>...</think> sections.

    Reasoning may end implicitly when a tool call begins
    (<minimax:tool_call>).

    Reasoning may begin implicitly, without an explicit <think> token
    (the chat template appends <think> to the assistant turn).
    """

    def __init__(
        self,
        think_start_token_id: int,
        think_end_token_id: int,
        tool_call_start_token_id: int | None = None,
    ) -> None:
        self.think_start_token_id = think_start_token_id
        self.think_end_token_id = think_end_token_id
        self.tool_call_start_token_id = tool_call_start_token_id

    def stream(
        self,
        delta_token_ids: Sequence[int],
    ) -> ParsedReasoningDelta:
        """Identify a reasoning span within a streaming delta chunk."""
        end_token_ids = (
            (self.think_end_token_id, self.tool_call_start_token_id)
            if self.tool_call_start_token_id is not None
            else (self.think_end_token_id,)
        )

        start_token_idx: int | None = None
        end_token_idx: int | None = None
        for i, token_id in enumerate(delta_token_ids):
            if (
                start_token_idx is None
                and token_id == self.think_start_token_id
            ):
                # Take the earliest start token
                start_token_idx = i
            elif token_id in end_token_ids:
                # Take the earliest end token
                end_token_idx = i
                break

        if start_token_idx is None:
            # Implicit start: chat template pre-fills <think>, so reasoning
            # begins at index 0 when no explicit <think> token is present.
            start_reasoning = 0
            start_reasoning_with_delimiters = 0
        else:
            start_reasoning = start_token_idx + 1
            start_reasoning_with_delimiters = start_token_idx

        if end_token_idx is None:
            end_reasoning = len(delta_token_ids)
            end_reasoning_with_delimiters = len(delta_token_ids)
        elif delta_token_ids[end_token_idx] == self.think_end_token_id:
            # </think> is consumed as a delimiter.
            end_reasoning = end_token_idx
            end_reasoning_with_delimiters = end_token_idx + 1
        else:
            # <minimax:tool_call> is not consumed — the tool call belongs
            # to the content region where downstream tool parsing handles
            # it.
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
            span=span,
            is_still_reasoning=is_still_reasoning,
        )

    def is_prompt_in_reasoning(self, prompt_token_ids: Sequence[int]) -> bool:
        """Decide whether the next generated token is in a reasoning span.

        Only checks for ``</think>`` — not ``<minimax:tool_call>`` — because
        the chat template embeds tool-call format tokens in the system prompt
        when tools are provided, which must not disable reasoning for the
        generation that follows.
        """
        return self.think_end_token_id not in prompt_token_ids

    @classmethod
    async def from_tokenizer(
        cls,
        tokenizer: PipelineTokenizer[Any, Any, Any],
    ) -> MiniMaxM2ReasoningParser:
        """Construct a reasoning parser from a tokenizer."""
        think_start_id = await convert_token_to_id(tokenizer, "<think>")
        think_end_id = await convert_token_to_id(tokenizer, "</think>")

        if think_start_id is None or think_end_id is None:
            raise ValueError(
                f"{cls.__name__} could not locate think start/end tokens in the tokenizer"
            )

        tool_call_start_id = await convert_token_to_id(
            tokenizer, "<minimax:tool_call>"
        )

        return cls(
            think_start_token_id=think_start_id,
            think_end_token_id=think_end_id,
            tool_call_start_token_id=tool_call_start_id,
        )
