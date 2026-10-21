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

"""Hy4 reasoning parser for ``<think:opensource>`` sections."""

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

# Hy4's structural tokens carry a per-checkpoint suffix; the published
# checkpoints use this one.
TOKEN_SUFFIX = ":opensource"


@register("hy_v4")
class HYV4ReasoningParser(ReasoningParser):
    """Hy4 reasoning parser for ``<think:opensource>`` sections.

    With ``reasoning_effort`` ``high`` (the template default) the generation
    prompt ends with the opening tag, so reasoning starts implicitly and ends
    at the closing tag. With ``no_think`` the prompt ends with both tags, so
    the output is all content.

    Once reasoning has closed, in the prompt or in the output, a later
    ``<think:opensource>`` is content: the model emits that id as plain text
    mid-answer, and Hy4 has no second reasoning span.
    """

    REASONING_START: ClassVar[str] = f"<think{TOKEN_SUFFIX}>"
    REASONING_END: ClassVar[str] = f"</think{TOKEN_SUFFIX}>"

    def __init__(self, think_start_token_id: int, think_end_token_id: int):
        self.think_start_token_id = think_start_token_id
        self.think_end_token_id = think_end_token_id

    def stream(
        self,
        delta_token_ids: Sequence[int],
        is_currently_reasoning: bool = True,
    ) -> ParsedReasoningDelta:
        """Identifies a reasoning span within a streaming delta chunk."""
        if not is_currently_reasoning:
            return ParsedReasoningDelta(
                span=ReasoningSpan(
                    reasoning_with_delimiters=(0, 0), reasoning=(0, 0)
                ),
                is_still_reasoning=False,
            )

        start_idx: int | None = None
        end_idx: int | None = None
        for i, token_id in enumerate(delta_token_ids):
            if start_idx is None and token_id == self.think_start_token_id:
                start_idx = i
            elif token_id == self.think_end_token_id:
                end_idx = i
                break

        begin = 0 if start_idx is None else start_idx + 1
        begin_with_delimiters = 0 if start_idx is None else start_idx
        end = len(delta_token_ids) if end_idx is None else end_idx
        end_with_delimiters = (
            len(delta_token_ids) if end_idx is None else end_idx + 1
        )
        return ParsedReasoningDelta(
            span=ReasoningSpan(
                reasoning_with_delimiters=(
                    begin_with_delimiters,
                    end_with_delimiters,
                ),
                reasoning=(begin, end),
            ),
            is_still_reasoning=end_idx is None,
        )

    def will_reason_after_prompt(self, prompt_token_ids: Sequence[int]) -> bool:
        """Returns whether the last think tag in the prompt is an opening one.

        Scans right to left because multi-turn prompts carry closed think
        blocks from earlier assistant turns.
        """
        for token_id in reversed(prompt_token_ids):
            if token_id == self.think_start_token_id:
                return True
            if token_id == self.think_end_token_id:
                return False
        return True

    @classmethod
    async def from_tokenizer(
        cls, tokenizer: PipelineTokenizer[Any, Any, Any]
    ) -> HYV4ReasoningParser:
        """Constructs a reasoning parser from a tokenizer."""
        start_id = await convert_token_to_id(tokenizer, cls.REASONING_START)
        end_id = await convert_token_to_id(tokenizer, cls.REASONING_END)
        if start_id is None or end_id is None:
            raise ValueError(
                f"{cls.__name__} could not locate {cls.REASONING_START} and"
                f" {cls.REASONING_END} as single tokens in the tokenizer"
            )
        return cls(think_start_token_id=start_id, think_end_token_id=end_id)

    @classmethod
    async def reasoning_end_token_id(
        cls, tokenizer: PipelineTokenizer[Any, Any, Any]
    ) -> int | None:
        """Returns the token id that closes a reasoning span."""
        return await convert_token_to_id(tokenizer, cls.REASONING_END)
