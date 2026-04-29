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

"""Incremental detokenizer for proper UTF-8 handling during streaming.

This module provides an `IncrementalDetokenizer` class that properly handles
multi-byte UTF-8 sequences during incremental token decoding. The problem it
solves is that some characters (like emojis) require multiple tokens to
represent, and decoding these tokens individually produces replacement
characters (�) instead of the correct characters.

For example, the fingerprint emoji 🫆 (U+1FAC6) requires 4 bytes in UTF-8
encoding (\xf0\x9f\xab\x86). When tokenized, it may be split across multiple
tokens. Decoding each token separately produces replacement characters.

The solution is to use the `tokenizers` library's `DecodeStream`, which
internally buffers partial UTF-8 sequences until they form complete characters.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable, Sequence
from typing import TYPE_CHECKING, Any

import numpy as np
import numpy.typing as npt
from tokenizers import Tokenizer
from tokenizers.decoders import DecodeStream

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizerBase

# Type alias for token ID sequences that can be passed to the detokenizer
TokenIDSequence = Sequence[int] | npt.NDArray[np.integer[Any]] | Iterable[int]

logger = logging.getLogger("max.serve")

# Error string from tokenizers library that indicates invalid prefix
INVALID_PREFIX_ERR_MSG = "Invalid prefix encountered"


def is_fast_tokenizer(tokenizer: object) -> bool:
    """Checks if a tokenizer is a HuggingFace fast tokenizer.

    Fast tokenizers wrap the `tokenizers` library and expose a `_tokenizer`
    attribute which is the underlying `tokenizers.Tokenizer` object.

    Args:
        tokenizer: The tokenizer to check.

    Returns:
        True if the tokenizer is a fast tokenizer, False otherwise.
    """
    return hasattr(tokenizer, "_tokenizer") and isinstance(
        getattr(tokenizer, "_tokenizer", None), Tokenizer
    )


class IncrementalDetokenizer:
    """Handles incremental detokenization with proper UTF-8 sequence handling.

    This class wraps the `tokenizers` library's `DecodeStream` to provide
    correct incremental detokenization that handles multi-byte UTF-8 characters
    spanning multiple tokens.

    The `DecodeStream` is initialized with the prompt tokens to establish
    proper context for decoding (some tokenizers need preceding context to
    decode correctly, e.g., for whitespace handling).
    """

    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        prompt_token_ids: TokenIDSequence,
        skip_special_tokens: bool = True,
    ) -> None:
        """Initializes the incremental detokenizer.

        Args:
            tokenizer: A HuggingFace tokenizer. Must be a "fast" tokenizer
                that wraps the `tokenizers` library for proper UTF-8 handling.
                If a slow tokenizer is passed, incremental decoding will fall
                back to direct decoding which may produce replacement characters
                for multi-byte sequences.
            prompt_token_ids: The prompt token IDs to prime the decoder with.
                This establishes context for proper decoding.
            skip_special_tokens: Whether to skip special tokens during
                decoding (e.g., EOS tokens like <|im_end|>).

        Raises:
            TypeError: If the tokenizer is not a fast tokenizer.
        """
        self.skip_special_tokens = skip_special_tokens

        if not is_fast_tokenizer(tokenizer):
            raise TypeError(
                "IncrementalDetokenizer requires a HuggingFace fast tokenizer "
                "(PreTrainedTokenizerFast). The provided tokenizer does not "
                "have a _tokenizer attribute from the tokenizers library."
            )

        # Get the underlying tokenizers.Tokenizer from the HF tokenizer
        self._tokenizer: Tokenizer = tokenizer._tokenizer

        # Initialize DecodeStream with native prefill (tokenizers >= 0.22.0)
        # This primes the decoder with prompt context for proper whitespace
        # handling and establishes the decode stream state.
        self.stream = DecodeStream(
            ids=list(prompt_token_ids),
            skip_special_tokens=skip_special_tokens,
        )

    def decode(self, token_ids: TokenIDSequence) -> str:
        """Decodes a sequence of token IDs to text incrementally.

        This method handles partial UTF-8 sequences by buffering incomplete
        byte sequences until they can form complete UTF-8 characters.

        Args:
            token_ids: The token IDs to decode.

        Returns:
            The decoded text. May be empty if the tokens represent incomplete
            UTF-8 sequences that are buffered for later completion.
        """
        result_parts: list[str] = []

        for token_id in token_ids:
            text = self._protected_step(token_id)
            if text:
                result_parts.append(text)

        return "".join(result_parts)

    def _protected_step(self, token_id: int) -> str:
        """Performs a single step of incremental decoding with error handling.

        The `DecodeStream.step()` method can sometimes fail with "Invalid prefix
        encountered" for certain token sequences that produce non-monotonic or
        invalid UTF-8 output. This method resets the stream and retries once
        in such cases.

        Args:
            token_id: The token ID to decode.

        Returns:
            The decoded text for this token, or empty string if the token
            contributes to a buffered partial UTF-8 sequence.
        """
        for _ in range(2):
            try:
                # DecodeStream.step() exists at runtime but is missing from
                # type stubs (tokenizers library). Ruff B009 forbids getattr
                # workarounds, so we use type: ignore here.
                result: str | None = self.stream.step(  # type: ignore[attr-defined]
                    self._tokenizer, token_id
                )
                return result or ""
            except Exception as e:
                if INVALID_PREFIX_ERR_MSG not in str(e):
                    raise
                # Reset the stream and retry once
                logger.debug(
                    "Resetting decode stream due to invalid prefix error"
                )
                self.stream = DecodeStream(
                    skip_special_tokens=self.skip_special_tokens
                )
        return ""


def get_hf_tokenizer(tokenizer: object) -> PreTrainedTokenizerBase:
    """Gets the underlying HuggingFace tokenizer from a PipelineTokenizer.

    This handles the common cases where the tokenizer is a TextTokenizer or
    TextAndVisionTokenizer which have a `delegate` attribute containing the
    HuggingFace tokenizer.

    Args:
        tokenizer: A PipelineTokenizer implementation.

    Returns:
        The underlying HuggingFace tokenizer.

    Raises:
        ValueError: If the tokenizer does not have a delegate attribute.
    """
    # Check for delegate attribute (TextTokenizer, TextAndVisionTokenizer)
    if hasattr(tokenizer, "delegate"):
        return tokenizer.delegate
    raise ValueError(
        f"Tokenizer {type(tokenizer).__name__} does not have a delegate "
        "attribute. IncrementalDetokenizer requires a PipelineTokenizer with "
        "an underlying HuggingFace tokenizer."
    )


def create_incremental_detokenizer(
    tokenizer: object,
    prompt_token_ids: TokenIDSequence,
    skip_special_tokens: bool = True,
) -> IncrementalDetokenizer | None:
    """Creates an IncrementalDetokenizer if the tokenizer supports it.

    This is a factory function that handles the common case of creating an
    IncrementalDetokenizer from a PipelineTokenizer. It extracts the underlying
    HuggingFace tokenizer and creates the IncrementalDetokenizer if possible.

    Note: TikToken-based tokenizers (used by DeepSeek, Kimi K2.5) are supported
    as long as they are loaded via AutoTokenizer, which provides a fast tokenizer
    with the required `_tokenizer` attribute.

    Args:
        tokenizer: A PipelineTokenizer implementation (e.g., TextTokenizer).
        prompt_token_ids: The prompt token IDs to prime the decoder with.
        skip_special_tokens: Whether to skip special tokens during decoding.

    Returns:
        An IncrementalDetokenizer if the tokenizer supports it, or None if
        the tokenizer is not a supported type (e.g., slow tokenizer or missing
        delegate).
    """
    try:
        hf_tokenizer = get_hf_tokenizer(tokenizer)
    except ValueError:
        return None

    if not is_fast_tokenizer(hf_tokenizer):
        return None

    return IncrementalDetokenizer(
        tokenizer=hf_tokenizer,
        prompt_token_ids=prompt_token_ids,
        skip_special_tokens=skip_special_tokens,
    )
