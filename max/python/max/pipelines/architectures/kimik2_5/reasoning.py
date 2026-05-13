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

"""Kimi K2.5 reasoning parser for <think>...</think> sections."""

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


@register("kimik2_5")
class KimiK2_5ReasoningParser(ReasoningParser):
    """Kimi K2.5 reasoning parser for <think>...</think> sections.

    Per Moonshot's "Interleaved Thinking" design (see
    https://platform.moonshot.ai/docs/guide/use-kimi-k2-thinking-model and
    https://huggingface.co/moonshotai/Kimi-K2.5), a single assistant turn
    can interleave multiple ``<think>...</think>`` blocks with
    ``<|tool_calls_section_begin|>...<|tool_calls_section_end|>`` blocks.
    Only ``</think>`` ends a reasoning span; tool-call sections are
    neither reasoning nor content and are consumed by the tool parser
    (when the client opted in via ``tools=[...]``) or stripped from
    user-visible output otherwise.

    Reasoning may begin implicitly, without an explicit ``<think>``
    token, when the chat template prefilled the assistant turn already
    inside a thinking block.

    Reasoning can be disabled through the chat template by including a
    ``</think>`` token at the end of the prompt; this is detected by
    :meth:`is_prompt_in_reasoning`.
    """

    def __init__(
        self,
        think_start_token_id: int,
        think_end_token_id: int,
        tool_section_start_token_id: int | None = None,
    ) -> None:
        self.think_start_token_id = think_start_token_id
        self.think_end_token_id = think_end_token_id
        # Retained only to disambiguate the "implicit pre-fill" path in
        # :meth:`is_prompt_in_reasoning`. ``stream()`` never treats it as
        # an end-of-reasoning delimiter.
        self.tool_section_start_token_id = tool_section_start_token_id

    def stream(
        self,
        delta_token_ids: Sequence[int],
    ) -> ParsedReasoningDelta:
        """Identify a reasoning span within a streaming delta chunk."""
        start_token_idx: int | None = None
        end_token_idx: int | None = None
        for i, token_id in enumerate(delta_token_ids):
            if (
                start_token_idx is None
                and token_id == self.think_start_token_id
            ):
                # Take the earliest start token
                start_token_idx = i
            elif token_id == self.think_end_token_id:
                # Take the earliest end token
                end_token_idx = i
                break

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

    def is_prompt_in_reasoning(
        self,
        prompt_token_ids: Sequence[int],
    ) -> bool:
        """Decide whether the next generated token is in a reasoning span.

        Kimi K2.5 chat templates emit ``<think>`` to open the new assistant
        turn's reasoning section, and ``</think>`` to close the prior
        assistant turn's reasoning section. A multi-turn prompt therefore
        can contain many ``<think>``/``</think>`` tokens, only the
        most-recently-emitted one of which describes the *current* state.

        Scan right-to-left and return based on the first delimiter seen:

        * ``<think>`` → reasoning is currently open → ``True``.
        * ``</think>`` (or ``<|tool_calls_section_begin|>``) → reasoning
          is currently closed → ``False``.
        * No delimiters at all → assume reasoning, matching the implicit
          pre-fill seeding used elsewhere in the pipeline.
        """
        end_token_ids: tuple[int, ...]
        if self.tool_section_start_token_id is not None:
            end_token_ids = (
                self.think_end_token_id,
                self.tool_section_start_token_id,
            )
        else:
            end_token_ids = (self.think_end_token_id,)

        for token_id in reversed(prompt_token_ids):
            if token_id == self.think_start_token_id:
                return True
            if token_id in end_token_ids:
                return False
        return True

    @classmethod
    async def from_tokenizer(
        cls,
        tokenizer: PipelineTokenizer[Any, Any, Any],
    ) -> KimiK2_5ReasoningParser:
        """Construct a reasoning parser from a tokenizer."""
        think_start_id = await convert_token_to_id(tokenizer, "<think>")
        think_end_id = await convert_token_to_id(tokenizer, "</think>")

        if think_start_id is None or think_end_id is None:
            raise ValueError(
                f"{cls.__name__} could not locate think start/end tokens in the tokenizer"
            )

        tool_section_start_id = await convert_token_to_id(
            tokenizer, "<|tool_calls_section_begin|>"
        )

        return cls(
            think_start_token_id=think_start_id,
            think_end_token_id=think_end_id,
            tool_section_start_token_id=tool_section_start_id,
        )
