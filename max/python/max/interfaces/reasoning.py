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
from collections.abc import Sequence
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
        reasoning_with_delimiters: Full span including delimiter tokens.
        reasoning: Span excluding delimiter tokens. Must be contained within ``reasoning_with_delimiters``.
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
        """Extract the non-reasoning elements from a sequence.

        Returns the elements outside the delimited reasoning span.
        """
        delimited_start, delimited_end = self._reasoning_with_delimiters
        return list(seq[:delimited_start]) + list(seq[delimited_end:])

    def extract_reasoning(self, seq: Sequence[T]) -> list[T]:
        """Extract the reasoning elements from a sequence.

        Returns the elements within the reasoning span (excluding delimiters).
        """
        reasoning_start, reasoning_end = self._reasoning
        return list(seq[reasoning_start:reasoning_end])


class ReasoningParser(ABC):
    """Parser for identifying reasoning spans in model output."""

    @abstractmethod
    def stream(
        self,
        delta_token_ids: Sequence[int],
    ) -> tuple[ReasoningSpan, bool]:
        """Identify a reasoning span within a streaming delta chunk.

        Returns a tuple of (ReasoningSpan, is_still_reasoning) where
        is_still_reasoning indicates whether the reasoning section has ended.
        The ReasoningSpan identifies the reasoning portion of the chunk. If
        there is no reasoning in the chunk, the span is zero-width so that
        ``extract_content`` behaves as identity and ``extract_reasoning``
        returns an empty list.
        """
        ...

    @classmethod
    @abstractmethod
    async def from_tokenizer(
        cls,
        tokenizer: PipelineTokenizer[Any, Any, Any],
    ) -> ReasoningParser:
        """Construct a reasoning parser from a tokenizer."""
        ...
