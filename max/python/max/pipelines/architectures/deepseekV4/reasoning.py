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
# ruff: noqa: RUF002

"""DeepSeek-V4 reasoning parser for sections framed by ``<think>`` and ``</think>``."""

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


@register("deepseekv4")
class DeepseekV4ReasoningParser(ReasoningParser):
    """DeepSeek-V4 reasoning parser for spans framed by ``<think>`` and ``</think>``.

    In thinking mode the prompt ends with a prefilled ``<think>``, so
    reasoning begins implicitly; in chat mode it ends with ``</think>``. Only
    ``</think>`` ends reasoning: the encoder's tool instructions require the
    complete reasoning block before any ``<｜DSML｜tool_calls>`` block.
    """

    REASONING_START: ClassVar[str] = "<think>"
    REASONING_END: ClassVar[str] = "</think>"

    def __init__(
        self,
        think_start_token_id: int,
        think_end_token_id: int,
    ) -> None:
        self.think_start_token_id = think_start_token_id
        self.think_end_token_id = think_end_token_id

    def stream(
        self,
        delta_token_ids: Sequence[int],
        is_currently_reasoning: bool = True,
    ) -> ParsedReasoningDelta:
        """Identifies a reasoning span within a streaming delta chunk.

        When ``is_currently_reasoning=False`` and the chunk contains no
        ``<think>`` opener, returns an empty span so chat-mode output and
        chunks after reasoning ended aren't misclassified as reasoning.
        """
        start_token_idx: int | None = None
        end_token_idx: int | None = None
        for i, token_id in enumerate(delta_token_ids):
            if (
                start_token_idx is None
                and token_id == self.think_start_token_id
            ):
                start_token_idx = i
            elif token_id == self.think_end_token_id:
                # A stray ``</think>`` outside an open span is content.
                if is_currently_reasoning or start_token_idx is not None:
                    end_token_idx = i
                    break

        if start_token_idx is None and not is_currently_reasoning:
            return ParsedReasoningDelta(
                span=ReasoningSpan(
                    reasoning_with_delimiters=(0, 0),
                    reasoning=(0, 0),
                ),
                is_still_reasoning=False,
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
        else:
            end_reasoning = end_token_idx
            end_reasoning_with_delimiters = end_token_idx + 1

        return ParsedReasoningDelta(
            span=ReasoningSpan(
                reasoning_with_delimiters=(
                    start_reasoning_with_delimiters,
                    end_reasoning_with_delimiters,
                ),
                reasoning=(start_reasoning, end_reasoning),
            ),
            is_still_reasoning=end_token_idx is None,
        )

    def will_reason_after_prompt(
        self,
        prompt_token_ids: Sequence[int],
    ) -> bool:
        """Predicts whether the model will emit reasoning after this prompt.

        Scans right-to-left: earlier turns and the tool instructions also
        carry ``<think>``/``</think>``, but the last delimiter is the one the
        encoder prefilled after ``<｜Assistant｜>``.
        """
        for token_id in reversed(prompt_token_ids):
            if token_id == self.think_start_token_id:
                return True
            if token_id == self.think_end_token_id:
                return False
        return False

    @classmethod
    async def from_tokenizer(
        cls,
        tokenizer: PipelineTokenizer[Any, Any, Any],
    ) -> DeepseekV4ReasoningParser:
        """Constructs a reasoning parser from a tokenizer."""
        think_start_id = await convert_token_to_id(
            tokenizer, cls.REASONING_START
        )
        think_end_id = await convert_token_to_id(tokenizer, cls.REASONING_END)
        if think_start_id is None or think_end_id is None:
            raise ValueError(
                f"{cls.__name__} could not locate think start/end tokens in the tokenizer"
            )
        return cls(
            think_start_token_id=think_start_id,
            think_end_token_id=think_end_id,
        )

    @classmethod
    async def reasoning_end_token_id(
        cls,
        tokenizer: PipelineTokenizer[Any, Any, Any],
    ) -> int | None:
        """Returns the ``</think>`` token id."""
        return await convert_token_to_id(tokenizer, cls.REASONING_END)
