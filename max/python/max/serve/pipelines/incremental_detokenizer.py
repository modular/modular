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

"""Buffered detokenizer for proper UTF-8 handling during streaming.

This module provides a `BufferedDetokenizer` abstraction that properly handles
multi-byte UTF-8 sequences during incremental token decoding. The problem it
solves is that some characters (like emojis) require multiple tokens to
represent, and decoding these tokens individually produces replacement
characters (U+FFFD) instead of the correct characters.

For example, the emoji 😊 (U+1F60A) requires 4 bytes in UTF-8 encoding.
When tokenized, it may be split across multiple tokens. Decoding each token
separately produces replacement characters.

Three implementations are provided:
- `DecodeStreamDetokenizer`: Uses the native `tokenizers` library's
  `DecodeStream` for fast tokenizers (most efficient).
- `Utf8BufferingDetokenizer`: Buffers tokens and re-decodes when replacement
  characters are detected (for non-fast tokenizers).
- `PassthroughDetokenizer`: Direct decode without buffering (fallback).

Use `create_buffered_detokenizer()` to automatically select the best
implementation for a given tokenizer.
"""

from __future__ import annotations

import abc
import logging
from collections.abc import Coroutine, Iterable, Sequence
from typing import TYPE_CHECKING, Any, Protocol

import numpy as np
import numpy.typing as npt
from tokenizers import Tokenizer
from tokenizers.decoders import DecodeStream

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizerBase

# Type alias for token ID sequences that can be passed to the detokenizer
TokenIDSequence = Sequence[int] | npt.NDArray[np.integer[Any]] | Iterable[int]


class AsyncDecodeFunc(Protocol):
    """Protocol for async decode functions that accept keyword arguments."""

    def __call__(
        self,
        token_ids: npt.NDArray[np.integer[Any]],
        *,
        skip_special_tokens: bool,
    ) -> Coroutine[Any, Any, str]:
        """Decodes token IDs to text.

        Args:
            token_ids: Array of token IDs to decode.
            skip_special_tokens: Whether to skip special tokens during decoding.

        Returns:
            Decoded text string.
        """
        ...


logger = logging.getLogger("max.serve")

# Unicode replacement character - indicates invalid/incomplete UTF-8
_REPLACEMENT_CHAR = "\ufffd"

# Maximum number of tokens to buffer when handling incomplete UTF-8 sequences.
# This covers emojis (typically 1-4 tokens) and most multi-byte characters.
_MAX_BUFFER_TOKENS = 8

# Error string from tokenizers library that indicates invalid prefix
_INVALID_PREFIX_ERR_MSG = "Invalid prefix encountered"


class BufferedDetokenizer(abc.ABC):
    """Abstract base class for detokenizers with UTF-8 buffering.

    All implementations provide an async `decode()` method that handles
    multi-byte UTF-8 sequences that may span multiple tokens.
    """

    @abc.abstractmethod
    async def decode(self, token_ids: TokenIDSequence) -> str:
        """Decodes token IDs to text with proper UTF-8 handling.

        Args:
            token_ids: The token IDs to decode.

        Returns:
            The decoded text. May be empty if tokens represent incomplete
            UTF-8 sequences that are buffered for later completion.
        """
        ...


class DecodeStreamDetokenizer(BufferedDetokenizer):
    """Detokenizer using the native `tokenizers` library's DecodeStream.

    This is the most efficient implementation, using the Rust-based
    `DecodeStream` which internally buffers partial UTF-8 sequences.
    Only works with HuggingFace fast tokenizers.
    """

    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        prompt_token_ids: TokenIDSequence,
        skip_special_tokens: bool = True,
        skipped_special_token_ids: set[int] | None = None,
    ) -> None:
        """Initializes the DecodeStream-based detokenizer.

        Args:
            tokenizer: A HuggingFace fast tokenizer with `_tokenizer` attribute.
            prompt_token_ids: The prompt token IDs to prime the decoder with.
            skip_special_tokens: Whether to skip special tokens during decoding.
            skipped_special_token_ids: Optional explicit set of special-token
                ids to skip. When provided alongside `skip_special_tokens=True`,
                the underlying `DecodeStream` runs with `skip_special_tokens=False`
                and these ids are filtered out instead.

        Raises:
            TypeError: If the tokenizer is not a fast tokenizer.
        """
        self._skip_special_tokens = skip_special_tokens
        self._skipped_special_token_ids = skipped_special_token_ids

        # Public properties for backward compatibility with IncrementalDetokenizer
        self.skip_special_tokens = skip_special_tokens
        self.skipped_special_token_ids = skipped_special_token_ids

        if not _is_fast_tokenizer(tokenizer):
            raise TypeError(
                "DecodeStreamDetokenizer requires a HuggingFace fast tokenizer "
                "(PreTrainedTokenizerFast). The provided tokenizer does not "
                "have a _tokenizer attribute from the tokenizers library."
            )

        self._tokenizer: Tokenizer = tokenizer._tokenizer

        # When caller supplies explicit exclusion set, keep DecodeStream in
        # skip_special_tokens=False mode and filter ourselves in decode().
        stream_skip_special = skip_special_tokens and (
            skipped_special_token_ids is None
        )

        self._stream = DecodeStream(
            ids=list(prompt_token_ids),
            skip_special_tokens=stream_skip_special,
        )

    async def decode(self, token_ids: TokenIDSequence) -> str:
        """Decodes token IDs using the native DecodeStream."""
        result_parts: list[str] = []
        excluded = self._skipped_special_token_ids

        for token_id in token_ids:
            if (
                self._skip_special_tokens
                and excluded is not None
                and token_id in excluded
            ):
                continue
            text = self._protected_step(token_id)
            if text:
                result_parts.append(text)

        return "".join(result_parts)

    def _protected_step(self, token_id: int) -> str:
        """Performs a single decode step with error recovery."""
        for _ in range(2):
            try:
                result: str | None = self._stream.step(  # type: ignore[attr-defined]
                    self._tokenizer, token_id
                )
                return result or ""
            except Exception as e:
                if _INVALID_PREFIX_ERR_MSG not in str(e):
                    raise
                logger.debug(
                    "Resetting decode stream due to invalid prefix error"
                )
                stream_skip_special = self._skip_special_tokens and (
                    self._skipped_special_token_ids is None
                )
                self._stream = DecodeStream(
                    skip_special_tokens=stream_skip_special
                )
        return ""


class Utf8BufferingDetokenizer(BufferedDetokenizer):
    """Detokenizer that buffers tokens to handle incomplete UTF-8 sequences.

    This implementation works with any tokenizer by detecting replacement
    characters (U+FFFD) at chunk boundaries and buffering the responsible
    tokens for re-decoding with the next chunk.
    """

    def __init__(
        self,
        decode_func: AsyncDecodeFunc,
        skip_special_tokens: bool = True,
    ) -> None:
        """Initializes the UTF-8 buffering detokenizer.

        Args:
            decode_func: An async function that decodes token IDs to text.
                Should have signature: decode(token_ids, skip_special_tokens) -> str
            skip_special_tokens: Whether to skip special tokens during decoding.
        """
        self._decode_func = decode_func
        self._skip_special_tokens = skip_special_tokens
        self._buffered_tokens: list[int] = []

    async def decode(self, token_ids: TokenIDSequence) -> str:
        """Decodes token IDs with UTF-8 buffering for incomplete sequences."""
        tokens = list(token_ids)

        # Prepend any buffered tokens from previous chunk
        if self._buffered_tokens:
            tokens = self._buffered_tokens + tokens
            self._buffered_tokens = []

        if not tokens:
            return ""

        decoded = await self._decode_func(
            np.array(tokens, dtype=np.int64),
            skip_special_tokens=self._skip_special_tokens,
        )

        if not decoded:
            return ""

        # Count trailing replacement characters
        trailing_replacements = 0
        for char in reversed(decoded):
            if char == _REPLACEMENT_CHAR:
                trailing_replacements += 1
            else:
                break

        if trailing_replacements == 0:
            return decoded

        # Try removing tokens until replacement chars are eliminated.
        # A single U+FFFD can represent an incomplete sequence from multiple
        # preceding tokens (e.g., a 4-byte emoji split across 3 tokens),
        # so we probe up to _MAX_BUFFER_TOKENS rather than just the count
        # of trailing replacement characters.
        tokens_to_buffer = min(len(tokens), _MAX_BUFFER_TOKENS)

        for num_buffer in range(1, tokens_to_buffer + 1):
            if num_buffer >= len(tokens):
                self._buffered_tokens = tokens
                return ""

            partial_tokens = tokens[:-num_buffer]
            partial_decoded = await self._decode_func(
                np.array(partial_tokens, dtype=np.int64),
                skip_special_tokens=self._skip_special_tokens,
            )

            if partial_decoded and not partial_decoded.endswith(
                _REPLACEMENT_CHAR
            ):
                self._buffered_tokens = tokens[-num_buffer:]
                return partial_decoded

        # Couldn't eliminate replacements - buffer the max window
        self._buffered_tokens = tokens[-tokens_to_buffer:]
        return (
            decoded[:-trailing_replacements]
            if trailing_replacements
            else decoded
        )


class PassthroughDetokenizer(BufferedDetokenizer):
    """Detokenizer that directly calls the tokenizer without buffering.

    This is the simplest implementation with no UTF-8 buffering. Use only
    when the tokenizer already handles UTF-8 correctly or when buffering
    is not needed.
    """

    def __init__(
        self,
        decode_func: AsyncDecodeFunc,
        skip_special_tokens: bool = True,
    ) -> None:
        """Initializes the passthrough detokenizer.

        Args:
            decode_func: An async function that decodes token IDs to text.
            skip_special_tokens: Whether to skip special tokens during decoding.
        """
        self._decode_func = decode_func
        self._skip_special_tokens = skip_special_tokens

    async def decode(self, token_ids: TokenIDSequence) -> str:
        """Decodes token IDs directly without buffering."""
        tokens = list(token_ids)
        if not tokens:
            return ""
        return await self._decode_func(
            np.array(tokens, dtype=np.int64),
            skip_special_tokens=self._skip_special_tokens,
        )


def _is_fast_tokenizer(tokenizer: object) -> bool:
    """Checks if a tokenizer is a HuggingFace fast tokenizer."""
    return hasattr(tokenizer, "_tokenizer") and isinstance(
        getattr(tokenizer, "_tokenizer", None), Tokenizer
    )


def _get_hf_tokenizer(tokenizer: object) -> PreTrainedTokenizerBase | None:
    """Gets the underlying HuggingFace tokenizer, or None if not available."""
    if hasattr(tokenizer, "delegate"):
        return tokenizer.delegate
    return None


def create_buffered_detokenizer(
    tokenizer: object,
    prompt_token_ids: TokenIDSequence,
    skip_special_tokens: bool = True,
) -> BufferedDetokenizer:
    """Creates a BufferedDetokenizer, selecting the best implementation.

    This factory function automatically selects the most appropriate
    detokenizer implementation:
    1. `DecodeStreamDetokenizer` for fast tokenizers (most efficient)
    2. `Utf8BufferingDetokenizer` for non-fast tokenizers (buffers tokens)
    3. `PassthroughDetokenizer` as fallback (no buffering)

    Args:
        tokenizer: A PipelineTokenizer implementation (e.g., TextTokenizer).
        prompt_token_ids: The prompt token IDs to prime the decoder with.
        skip_special_tokens: Whether to skip special tokens during decoding.

    Returns:
        A BufferedDetokenizer instance appropriate for the tokenizer type.
    """
    hf_tokenizer = _get_hf_tokenizer(tokenizer)

    # Try DecodeStreamDetokenizer for fast tokenizers
    if hf_tokenizer is not None and _is_fast_tokenizer(hf_tokenizer):
        skipped_special_token_ids: set[int] | None = getattr(
            tokenizer, "skipped_special_token_ids", None
        )
        return DecodeStreamDetokenizer(
            tokenizer=hf_tokenizer,
            prompt_token_ids=prompt_token_ids,
            skip_special_tokens=skip_special_tokens,
            skipped_special_token_ids=skipped_special_token_ids,
        )

    # Fall back to Utf8BufferingDetokenizer for non-fast tokenizers
    if hasattr(tokenizer, "decode"):
        return Utf8BufferingDetokenizer(
            decode_func=tokenizer.decode,
            skip_special_tokens=skip_special_tokens,
        )

    # Last resort: PassthroughDetokenizer (shouldn't normally reach here)
    raise ValueError(
        f"Tokenizer {type(tokenizer).__name__} does not have a decode method."
    )


# Legacy aliases for backward compatibility
IncrementalDetokenizer = DecodeStreamDetokenizer
FallbackUtf8Buffer = Utf8BufferingDetokenizer


def is_fast_tokenizer(tokenizer: object) -> bool:
    """Checks if a tokenizer is a HuggingFace fast tokenizer.

    Fast tokenizers wrap the `tokenizers` library and expose a `_tokenizer`
    attribute which is the underlying `tokenizers.Tokenizer` object.

    Args:
        tokenizer: The tokenizer to check.

    Returns:
        True if the tokenizer is a fast tokenizer, False otherwise.
    """
    return _is_fast_tokenizer(tokenizer)


def get_hf_tokenizer(tokenizer: object) -> PreTrainedTokenizerBase:
    """Gets the underlying HuggingFace tokenizer from a PipelineTokenizer.

    Args:
        tokenizer: A PipelineTokenizer implementation.

    Returns:
        The underlying HuggingFace tokenizer.

    Raises:
        ValueError: If the tokenizer does not have a delegate attribute.
    """
    result = _get_hf_tokenizer(tokenizer)
    if result is None:
        raise ValueError(
            f"Tokenizer {type(tokenizer).__name__} does not have a delegate "
            "attribute."
        )
    return result


def create_incremental_detokenizer(
    tokenizer: object,
    prompt_token_ids: TokenIDSequence,
    skip_special_tokens: bool = True,
) -> IncrementalDetokenizer | None:
    """Creates an IncrementalDetokenizer if the tokenizer supports it.

    Deprecated: Use `create_buffered_detokenizer()` instead, which always
    returns a detokenizer and handles all tokenizer types.

    Args:
        tokenizer: A PipelineTokenizer implementation.
        prompt_token_ids: The prompt token IDs to prime the decoder with.
        skip_special_tokens: Whether to skip special tokens during decoding.

    Returns:
        An IncrementalDetokenizer if the tokenizer supports it, or None.
    """
    hf_tokenizer = _get_hf_tokenizer(tokenizer)
    if hf_tokenizer is None or not _is_fast_tokenizer(hf_tokenizer):
        return None

    skipped_special_token_ids: set[int] | None = getattr(
        tokenizer, "skipped_special_token_ids", None
    )

    return IncrementalDetokenizer(
        tokenizer=hf_tokenizer,
        prompt_token_ids=prompt_token_ids,
        skip_special_tokens=skip_special_tokens,
        skipped_special_token_ids=skipped_special_token_ids,
    )


def create_fallback_utf8_buffer(
    decode_func: AsyncDecodeFunc,
    skip_special_tokens: bool = True,
) -> FallbackUtf8Buffer:
    """Creates a FallbackUtf8Buffer for non-fast tokenizers.

    Deprecated: Use `create_buffered_detokenizer()` instead.

    Args:
        decode_func: An async function that decodes token IDs to text.
        skip_special_tokens: Whether to skip special tokens during decoding.

    Returns:
        A FallbackUtf8Buffer instance for UTF-8 buffering.
    """
    return FallbackUtf8Buffer(
        decode_func=decode_func,
        skip_special_tokens=skip_special_tokens,
    )
