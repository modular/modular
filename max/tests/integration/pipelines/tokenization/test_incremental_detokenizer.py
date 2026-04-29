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
"""Tests for incremental detokenization with proper UTF-8 handling.

These tests verify that multi-byte UTF-8 characters (like emojis) that span
multiple tokens are correctly decoded without producing replacement characters.
This is a regression test for SERVSYS-1032.
"""

from __future__ import annotations

from collections.abc import Generator

import pytest
from max.serve.pipelines.incremental_detokenizer import (
    IncrementalDetokenizer,
    create_incremental_detokenizer,
    get_hf_tokenizer,
    is_fast_tokenizer,
)
from transformers import AutoTokenizer, PreTrainedTokenizerFast


@pytest.fixture
def llama_tokenizer() -> Generator[PreTrainedTokenizerFast, None, None]:
    """Loads a Llama tokenizer for testing."""
    yield AutoTokenizer.from_pretrained(
        "meta-llama/Llama-3.1-8B-Instruct",
        trust_remote_code=True,
    )


@pytest.fixture
def smol_tokenizer() -> Generator[PreTrainedTokenizerFast, None, None]:
    """Loads a small, fast tokenizer for testing."""
    yield AutoTokenizer.from_pretrained(
        "HuggingFaceTB/SmolLM2-135M-Instruct",
        trust_remote_code=True,
    )


class TestIsFastTokenizer:
    """Tests for the is_fast_tokenizer helper function."""

    def test_fast_tokenizer_returns_true(
        self, smol_tokenizer: PreTrainedTokenizerFast
    ) -> None:
        """Fast tokenizers should return True."""
        assert is_fast_tokenizer(smol_tokenizer)

    def test_non_tokenizer_returns_false(self) -> None:
        """Non-tokenizer objects should return False."""
        assert not is_fast_tokenizer("not a tokenizer")
        assert not is_fast_tokenizer(None)
        assert not is_fast_tokenizer(42)


class TestIncrementalDetokenizer:
    """Tests for the IncrementalDetokenizer class."""

    def test_basic_decoding(
        self, smol_tokenizer: PreTrainedTokenizerFast
    ) -> None:
        """Basic tokens should decode correctly."""
        prompt = "Hello"
        prompt_ids = smol_tokenizer.encode(prompt, add_special_tokens=False)

        detokenizer = IncrementalDetokenizer(
            tokenizer=smol_tokenizer,
            prompt_token_ids=prompt_ids,
            skip_special_tokens=True,
        )

        # Encode and decode "world"
        world_ids = smol_tokenizer.encode(" world", add_special_tokens=False)
        result = detokenizer.decode(world_ids)

        # The result should contain "world" (with possible whitespace)
        assert "world" in result

    def test_multi_byte_utf8_emoji(
        self, smol_tokenizer: PreTrainedTokenizerFast
    ) -> None:
        """Multi-byte UTF-8 emojis should decode correctly.

        The fingerprint emoji 🫆 (U+1FAC6) requires 4 bytes in UTF-8 and may
        be split across multiple tokens. Without incremental detokenization,
        each token decoded separately would produce replacement characters (�).
        """
        # The fingerprint emoji 🫆 (U+1FAC6)
        fingerprint_emoji = "🫆"

        # Encode a prompt containing the emoji
        prompt = f"This is the fingerprint emoji: {fingerprint_emoji}. "
        prompt_ids = smol_tokenizer.encode(prompt, add_special_tokens=False)

        detokenizer = IncrementalDetokenizer(
            tokenizer=smol_tokenizer,
            prompt_token_ids=[],  # No prompt context needed for this test
            skip_special_tokens=True,
        )

        # Decode the prompt tokens
        result = detokenizer.decode(prompt_ids)

        # The result should NOT contain replacement characters
        assert "�" not in result, (
            f"Replacement characters found in decoded output: {result!r}"
        )

        # The result should contain the fingerprint emoji
        assert fingerprint_emoji in result, (
            f"Expected fingerprint emoji {fingerprint_emoji!r} in result: {result!r}"
        )

    def test_incremental_decoding_preserves_utf8(
        self, smol_tokenizer: PreTrainedTokenizerFast
    ) -> None:
        """Incremental decoding should preserve UTF-8 across chunks."""
        # Encode text containing an emoji
        text_with_emoji = "Hello 😊 World"
        token_ids = smol_tokenizer.encode(
            text_with_emoji, add_special_tokens=False
        )

        detokenizer = IncrementalDetokenizer(
            tokenizer=smol_tokenizer,
            prompt_token_ids=[],
            skip_special_tokens=True,
        )

        # Decode tokens one at a time (simulating streaming)
        result_parts = []
        for token_id in token_ids:
            part = detokenizer.decode([token_id])
            result_parts.append(part)

        # Join all parts
        result = "".join(result_parts)

        # The result should NOT contain replacement characters
        assert "�" not in result, (
            f"Replacement characters found in incremental decode: {result!r}"
        )

        # The original text should be reconstructable (modulo whitespace)
        assert "😊" in result, f"Expected emoji in result: {result!r}"

    def test_various_emojis(
        self, smol_tokenizer: PreTrainedTokenizerFast
    ) -> None:
        """Various emojis should decode correctly."""
        emojis = [
            "🫆",  # Fingerprint (U+1FAC6) - 4 bytes
            "😊",  # Smiling face (U+1F60A) - 4 bytes
            "🙂",  # Slightly smiling (U+1F642) - 4 bytes
            "🤖",  # Robot (U+1F916) - 4 bytes
            "💻",  # Laptop (U+1F4BB) - 4 bytes
            "🔥",  # Fire (U+1F525) - 4 bytes
        ]

        for emoji in emojis:
            text = f"Test {emoji} text"
            token_ids = smol_tokenizer.encode(text, add_special_tokens=False)

            detokenizer = IncrementalDetokenizer(
                tokenizer=smol_tokenizer,
                prompt_token_ids=[],
                skip_special_tokens=True,
            )

            # Decode incrementally
            result = ""
            for token_id in token_ids:
                result += detokenizer.decode([token_id])

            # Should not have replacement characters
            assert "�" not in result, (
                f"Replacement character in result for emoji {emoji!r}: {result!r}"
            )

    def test_chinese_characters(
        self, smol_tokenizer: PreTrainedTokenizerFast
    ) -> None:
        """Chinese characters (3-byte UTF-8) should decode correctly."""
        chinese = "你好世界"  # Hello World in Chinese

        token_ids = smol_tokenizer.encode(chinese, add_special_tokens=False)

        detokenizer = IncrementalDetokenizer(
            tokenizer=smol_tokenizer,
            prompt_token_ids=[],
            skip_special_tokens=True,
        )

        # Decode incrementally
        result = ""
        for token_id in token_ids:
            result += detokenizer.decode([token_id])

        # Should not have replacement characters
        assert "�" not in result, (
            f"Replacement character in result for Chinese text: {result!r}"
        )

    def test_multiple_emojis_in_response(
        self, smol_tokenizer: PreTrainedTokenizerFast
    ) -> None:
        """Multiple emojis in a response are handled correctly (SERVSYS-1032).

        This simulates a model response containing multiple multi-byte emojis
        that may each be split across multiple tokens.
        """
        response = (
            "Here is the revised sentence: To take your 🫆, press your thumb "
            "against the ink pad and then the paper.\n\n"
            "Let me know if you have any questions! 😊"
        )

        token_ids = smol_tokenizer.encode(response, add_special_tokens=False)

        detokenizer = IncrementalDetokenizer(
            tokenizer=smol_tokenizer,
            prompt_token_ids=[],
            skip_special_tokens=True,
        )

        # Decode incrementally
        result = ""
        for token_id in token_ids:
            result += detokenizer.decode([token_id])

        # Should NOT produce replacement characters
        assert "�" not in result, (
            f"Replacement characters in response decode: {result!r}"
        )
        # Should contain both emojis
        assert "🫆" in result, f"Fingerprint emoji missing: {result!r}"
        assert "😊" in result, f"Smiling emoji missing: {result!r}"


class TestCreateIncrementalDetokenizer:
    """Tests for the create_incremental_detokenizer factory function."""

    def test_creates_detokenizer_for_fast_tokenizer(
        self, smol_tokenizer: PreTrainedTokenizerFast
    ) -> None:
        """Should create a detokenizer for fast tokenizers."""

        # Create a mock tokenizer wrapper that has a delegate attribute
        class MockTokenizerWrapper:
            def __init__(self, delegate: PreTrainedTokenizerFast) -> None:
                self.delegate = delegate

        wrapper = MockTokenizerWrapper(smol_tokenizer)

        detokenizer = create_incremental_detokenizer(
            tokenizer=wrapper,
            prompt_token_ids=[],
            skip_special_tokens=True,
        )

        assert detokenizer is not None
        assert isinstance(detokenizer, IncrementalDetokenizer)

    def test_returns_none_when_no_delegate(self) -> None:
        """Should return None when tokenizer has no delegate."""

        class NoDelegate:
            pass

        detokenizer = create_incremental_detokenizer(
            tokenizer=NoDelegate(),
            prompt_token_ids=[],
            skip_special_tokens=True,
        )

        assert detokenizer is None


class TestGetHfTokenizer:
    """Tests for the get_hf_tokenizer helper function."""

    def test_extracts_delegate(
        self, smol_tokenizer: PreTrainedTokenizerFast
    ) -> None:
        """Should extract the delegate from a tokenizer wrapper."""

        class MockWrapper:
            def __init__(self, delegate: PreTrainedTokenizerFast) -> None:
                self.delegate = delegate

        wrapper = MockWrapper(smol_tokenizer)
        hf_tokenizer = get_hf_tokenizer(wrapper)

        assert hf_tokenizer is smol_tokenizer

    def test_raises_for_no_delegate(self) -> None:
        """Should raise ValueError when no delegate attribute."""

        class NoDelegate:
            pass

        with pytest.raises(ValueError, match="does not have a delegate"):
            get_hf_tokenizer(NoDelegate())
