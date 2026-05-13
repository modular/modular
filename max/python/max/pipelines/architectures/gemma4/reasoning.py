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

"""Gemma 4 reasoning parser for <|channel>...<channel|> sections."""

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


@register("gemma4")
class Gemma4ReasoningParser(ReasoningParser):
    """Gemma 4 reasoning parser for ``<|channel>``...``<channel|>`` sections.

    When thinking is enabled, the chat template injects a ``<|think|>``
    token in the system message. The model then wraps reasoning output in
    ``<|channel>thought\\n...\\n<channel|>`` blocks. This parser identifies
    those blocks at the token-ID level.

    Reasoning may end implicitly when a tool call begins (``<|tool_call>``).
    The tool-call token is *not* consumed as a delimiter — it stays in the
    content region for downstream tool parsing.
    """

    # "thought\n" is not a special token — it is literal text that
    # immediately follows <|channel> to identify a thinking block.
    # See https://ai.google.dev/gemma/docs/core/model_card_4#2_thinking_mode_configuration
    reasoning_prefix = "thought\n"

    def __init__(
        self,
        channel_start_token_id: int,
        channel_end_token_id: int,
        tool_call_start_token_id: int | None = None,
        think_token_id: int | None = None,
    ) -> None:
        self.channel_start_token_id = channel_start_token_id
        self.channel_end_token_id = channel_end_token_id
        self.tool_call_start_token_id = tool_call_start_token_id
        self.think_token_id = think_token_id
        self._prefix_cursor = 0

    def reset(self) -> None:
        self._prefix_cursor = 0

    def _format_reasoning_text(self, reasoning: str) -> str | None:
        if self._prefix_cursor >= len(self.reasoning_prefix):
            return reasoning

        for i, ch in enumerate(reasoning):
            if self._prefix_cursor >= len(self.reasoning_prefix):
                return reasoning[i:]
            if self.reasoning_prefix[self._prefix_cursor] == ch:
                self._prefix_cursor += 1
            else:
                result = (
                    self.reasoning_prefix[: self._prefix_cursor] + reasoning[i:]
                )
                self._prefix_cursor = len(self.reasoning_prefix)
                return result

        return None

    def stream(
        self,
        delta_token_ids: Sequence[int],
    ) -> ParsedReasoningDelta:
        """Identifies a reasoning span within a streaming delta chunk.

        Returns a :class:`ParsedReasoningDelta` containing:

        - ``span``: a :class:`ReasoningSpan` with two index pairs into
          *delta_token_ids* — ``reasoning`` (content only) and
          ``reasoning_with_delimiters`` (includes boundary tokens).
        - ``is_still_reasoning``: ``True`` when no end delimiter was found
          in this chunk.
        - ``reasoning_text_formatter``: callback that strips the
          ``"thought\\n"`` prefix from decoded reasoning text.
        """
        end_token_ids = (
            (self.channel_end_token_id, self.tool_call_start_token_id)
            if self.tool_call_start_token_id is not None
            else (self.channel_end_token_id,)
        )

        start_token_idx: int | None = None
        end_token_idx: int | None = None
        for i, token_id in enumerate(delta_token_ids):
            if (
                start_token_idx is None
                and token_id == self.channel_start_token_id
            ):
                start_token_idx = i
            elif token_id in end_token_ids:
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
        elif delta_token_ids[end_token_idx] == self.channel_end_token_id:
            end_reasoning = end_token_idx
            end_reasoning_with_delimiters = end_token_idx + 1
        else:
            # <|tool_call> is not consumed — stays in content region.
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
            reasoning_text_formatter=self._format_reasoning_text,
        )

    def is_prompt_in_reasoning(
        self,
        prompt_token_ids: Sequence[int],
    ) -> bool:
        """Decide whether the next generated token is in a reasoning span.

        The Gemma 4 chat template does not prefill ``<|channel>`` at the
        assistant turn — the model self-generates it. Instead, the template
        injects ``<|think|>`` in the system message when thinking is enabled.
        Detect that token to know reasoning will start with the first
        generated token.
        """
        if self.think_token_id is not None:
            return self.think_token_id in prompt_token_ids
        return False

    @classmethod
    async def from_tokenizer(
        cls,
        tokenizer: PipelineTokenizer[Any, Any, Any],
    ) -> Gemma4ReasoningParser:
        """Construct a reasoning parser from a tokenizer."""
        channel_start_id = await convert_token_to_id(tokenizer, "<|channel>")
        channel_end_id = await convert_token_to_id(tokenizer, "<channel|>")

        if channel_start_id is None or channel_end_id is None:
            raise ValueError(
                f"{cls.__name__} could not locate channel start/end"
                " tokens in the tokenizer"
            )

        tool_call_start_id = await convert_token_to_id(
            tokenizer, "<|tool_call>"
        )
        think_id = await convert_token_to_id(tokenizer, "<|think|>")

        return cls(
            channel_start_token_id=channel_start_id,
            channel_end_token_id=channel_end_id,
            tool_call_start_token_id=tool_call_start_id,
            think_token_id=think_id,
        )
