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

"""Interfaces for interacting with the reasoning section of model output."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any, TypeVar

from .tokenizer import PipelineTokenizer

T = TypeVar("T")


class ReasoningSpan:
    """Identifies a reasoning span within a token ID sequence.

    A reasoning span is a contiguous span of tokens near the start of a reasoning
    model's output. In streaming mode with multiple chunks, the reasoning section
    may consist of one or more reasoning spans across multiple initial chunks.

    Tracks both the delimited reasoning span (including delimiter tokens like
    ``<think>`` and ``</think>``) and the reasoning span (excluding
    delimiters). Uses standard Python slice semantics: ``[start, end)``.

    Args:
        reasoning_with_delimiters: The full span including delimiter tokens.
        reasoning: The span excluding delimiter tokens. Must be contained
            within ``reasoning_with_delimiters``.
    """

    def __init__(
        self,
        reasoning_with_delimiters: tuple[int, int],
        reasoning: tuple[int, int],
    ) -> None:
        delimited_start, delimited_end = reasoning_with_delimiters
        reasoning_start, reasoning_end = reasoning
        assert delimited_start <= delimited_end
        assert reasoning_start <= reasoning_end
        assert delimited_start <= reasoning_start
        assert delimited_end >= reasoning_end
        self._reasoning_with_delimiters = reasoning_with_delimiters
        self._reasoning = reasoning

    def extract_content(self, seq: Sequence[T]) -> list[T]:
        """Extracts the non-reasoning elements from a sequence.

        Args:
            seq: The sequence from which to extract non-reasoning elements.

        Returns:
            The elements outside the delimited reasoning span.
        """
        delimited_start, delimited_end = self._reasoning_with_delimiters
        return list(seq[:delimited_start]) + list(seq[delimited_end:])

    def extract_reasoning(self, seq: Sequence[T]) -> list[T]:
        """Extracts the reasoning elements from a sequence.

        Args:
            seq: The sequence from which to extract reasoning elements.

        Returns:
            The elements within the reasoning span, excluding delimiters.
        """
        reasoning_start, reasoning_end = self._reasoning
        return list(seq[reasoning_start:reasoning_end])


@dataclass(frozen=True)
class ParsedReasoningDelta:
    """Result of applying reasoning parsing to a streaming delta chunk.

    Attributes:
        span: The :class:`ReasoningSpan` identifying the reasoning portion
            of the chunk.
        is_still_reasoning: Whether the reasoning section is still active.
        reasoning_text_formatter: Optional callback to post-process decoded
            reasoning text. Returns the formatted text, or ``None`` if the
            text should be ignored.
    """

    span: ReasoningSpan
    is_still_reasoning: bool
    reasoning_text_formatter: Callable[[str], str | None] | None = field(
        default=None
    )


class ReasoningParser(ABC):
    """Parser for identifying reasoning spans in model output."""

    @abstractmethod
    def stream(
        self,
        delta_token_ids: Sequence[int],
    ) -> ParsedReasoningDelta:
        """Identifies a reasoning span within a streaming delta chunk.

        Args:
            delta_token_ids: The token IDs of the incremental streaming chunk.

        Returns:
            A :class:`ParsedReasoningDelta` containing the reasoning span,
            whether reasoning is still active, and an optional formatter for
            decoded reasoning text.
        """
        ...

    def is_prompt_in_reasoning(
        self,
        prompt_token_ids: Sequence[int],
    ) -> bool:
        """Decide whether the next generated token continues a reasoning span.

        Called once at turn initiation, given the full prompt token ids
        (including any chat-template prefill). The result is used to seed
        the streaming reasoning state machine before the model emits its
        first token.

        Multi-turn prompts can legitimately contain ``</think>`` tokens
        from prior assistant turns. The default implementation delegates
        to :meth:`stream`, which scans left-to-right and would treat any
        such stale ``</think>`` as "reasoning has ended" — incorrect for
        the *new* assistant turn. Architectures whose chat templates emit
        reasoning delimiters per turn should override this to consider
        only the most recent delimiter (e.g., a right-to-left scan).

        Args:
            prompt_token_ids: The full prompt token id sequence.

        Returns:
            ``True`` if the next generated token should be treated as
            part of a reasoning span; ``False`` otherwise.
        """
        return self.stream(prompt_token_ids).is_still_reasoning

    def reset(self) -> None:
        """Resets per-request state.

        Called at the start of each request to clear any internal state
        accumulated during a prior request.
        """
        return

    @classmethod
    @abstractmethod
    async def from_tokenizer(
        cls,
        tokenizer: PipelineTokenizer[Any, Any, Any],
    ) -> ReasoningParser:
        """Constructs a reasoning parser from a tokenizer.

        Args:
            tokenizer: The :class:`~max.interfaces.PipelineTokenizer` to use
                for resolving reasoning delimiter token IDs.

        Returns:
            A new :class:`ReasoningParser` instance.
        """
        ...
