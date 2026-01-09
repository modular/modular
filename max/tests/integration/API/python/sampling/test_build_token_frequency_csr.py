# ===----------------------------------------------------------------------=== #
# Copyright (c) 2025, Modular Inc. All rights reserved.
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
"""Unit tests for _build_token_frequency_csr function.

This module tests the token frequency CSR matrix building function used
in the sampling logits processor. The CSR format stores token frequencies
efficiently for penalty calculations.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
from max.driver import CPU
from max.interfaces import TokenBuffer
from max.pipelines.core import TextContext
from max.pipelines.lib.sampling.sampling_logits_processor import (
    FrequencyData,
    _build_token_frequency_csr,
)


def create_text_context(
    prompt_tokens: npt.NDArray[np.int64],
    generated_tokens: npt.NDArray[np.int64] | None = None,
) -> TextContext:
    """Create a TextContext with specified prompt and generated tokens.

    Args:
        prompt_tokens: Array of prompt token IDs.
        generated_tokens: Optional array of generated token IDs.

    Returns:
        A TextContext with the specified token configuration.
    """
    if generated_tokens is None:
        generated_tokens = np.array([], dtype=np.int64)

    total_length = len(prompt_tokens) + len(generated_tokens)

    # Create context with prompt tokens only - TextContext.__post_init__
    # will set _prompt_len, _active_idx, _end_idx based on initial tokens
    context = TextContext(
        max_length=total_length + 100,  # Allow some headroom
        tokens=TokenBuffer(prompt_tokens.copy()),
    )

    # Use update() to add each generated token, which properly manages indices
    for token in generated_tokens:
        context.update(int(token))

    return context


class TestBuildTokenFrequencyCSRBasic:
    """Basic functionality tests for _build_token_frequency_csr."""

    def test_single_context_generated_only(self) -> None:
        """Test CSR building with a single context, generated tokens only."""
        prompt = np.array([1, 2, 3], dtype=np.int64)
        generated = np.array([4, 5, 4, 6, 4], dtype=np.int64)  # 4 appears 3x

        context = create_text_context(prompt, generated)
        batch = [context]
        device = CPU()
        padding_size = 2

        result = _build_token_frequency_csr(
            batch, padding_size, device, include_prompt=False
        )

        assert isinstance(result, FrequencyData)
        assert result.data is not None
        assert result.offsets is not None

        # Convert to numpy for verification
        data_np = result.data.to_numpy()
        offsets_np = result.offsets.to_numpy()

        # Verify offsets shape: batch_size + 1
        assert offsets_np.shape == (2,)
        assert offsets_np[0] == 0

        # Extract token-frequency pairs for the context
        start, end = offsets_np[0], offsets_np[1]
        context_data = data_np[start:end]

        # Filter out padding entries (token_id == -1)
        valid_entries = context_data[context_data[:, 0] != -1]

        # Build frequency dict from CSR data
        freq_dict = {int(row[0]): int(row[1]) for row in valid_entries}

        # Verify frequencies for generated tokens only
        assert freq_dict.get(4) == 3, (
            f"Token 4 should appear 3 times, got {freq_dict}"
        )
        assert freq_dict.get(5) == 1, (
            f"Token 5 should appear 1 time, got {freq_dict}"
        )
        assert freq_dict.get(6) == 1, (
            f"Token 6 should appear 1 time, got {freq_dict}"
        )
        # Prompt tokens should NOT be included
        assert 1 not in freq_dict, (
            "Prompt token 1 should not be in generated-only CSR"
        )
        assert 2 not in freq_dict, (
            "Prompt token 2 should not be in generated-only CSR"
        )
        assert 3 not in freq_dict, (
            "Prompt token 3 should not be in generated-only CSR"
        )

    def test_single_context_include_prompt(self) -> None:
        """Test CSR building with a single context, including prompt tokens."""
        prompt = np.array([1, 2, 3, 1], dtype=np.int64)  # 1 appears 2x
        generated = np.array([4, 5, 1], dtype=np.int64)  # 1 appears 1x more

        context = create_text_context(prompt, generated)
        batch = [context]
        device = CPU()
        padding_size = 2

        result = _build_token_frequency_csr(
            batch, padding_size, device, include_prompt=True
        )

        data_np = result.data.to_numpy()
        offsets_np = result.offsets.to_numpy()

        start, end = offsets_np[0], offsets_np[1]
        context_data = data_np[start:end]
        valid_entries = context_data[context_data[:, 0] != -1]
        freq_dict = {int(row[0]): int(row[1]) for row in valid_entries}

        # Token 1 appears 2x in prompt + 1x in generated = 3x total
        assert freq_dict.get(1) == 3, (
            f"Token 1 should appear 3 times, got {freq_dict}"
        )
        assert freq_dict.get(2) == 1
        assert freq_dict.get(3) == 1
        assert freq_dict.get(4) == 1
        assert freq_dict.get(5) == 1

    def test_empty_generated_tokens(self) -> None:
        """Test CSR building with no generated tokens."""
        prompt = np.array([1, 2, 3], dtype=np.int64)

        context = create_text_context(prompt)
        batch = [context]
        device = CPU()
        padding_size = 2

        # With include_prompt=False, should have no actual frequency data
        result = _build_token_frequency_csr(
            batch, padding_size, device, include_prompt=False
        )

        data_np = result.data.to_numpy()
        offsets_np = result.offsets.to_numpy()

        start, end = offsets_np[0], offsets_np[1]
        context_data = data_np[start:end]
        valid_entries = context_data[context_data[:, 0] != -1]

        # No generated tokens means no valid entries
        assert len(valid_entries) == 0

    def test_empty_generated_tokens_include_prompt(self) -> None:
        """Test CSR building with no generated tokens but include_prompt=True."""
        prompt = np.array([1, 2, 3, 1], dtype=np.int64)

        context = create_text_context(prompt)
        batch = [context]
        device = CPU()
        padding_size = 2

        result = _build_token_frequency_csr(
            batch, padding_size, device, include_prompt=True
        )

        data_np = result.data.to_numpy()
        offsets_np = result.offsets.to_numpy()

        start, end = offsets_np[0], offsets_np[1]
        context_data = data_np[start:end]
        valid_entries = context_data[context_data[:, 0] != -1]
        freq_dict = {int(row[0]): int(row[1]) for row in valid_entries}

        # Should include prompt tokens
        assert freq_dict.get(1) == 2
        assert freq_dict.get(2) == 1
        assert freq_dict.get(3) == 1


class TestBuildTokenFrequencyCSRBatch:
    """Tests for batch processing in _build_token_frequency_csr."""

    def test_multiple_contexts(self) -> None:
        """Test CSR building with multiple contexts in batch."""
        # Context 1: generated tokens [10, 10, 11]
        context1 = create_text_context(
            prompt_tokens=np.array([1, 2], dtype=np.int64),
            generated_tokens=np.array([10, 10, 11], dtype=np.int64),
        )

        # Context 2: generated tokens [20, 21, 20, 20]
        context2 = create_text_context(
            prompt_tokens=np.array([3, 4, 5], dtype=np.int64),
            generated_tokens=np.array([20, 21, 20, 20], dtype=np.int64),
        )

        batch = [context1, context2]
        device = CPU()
        padding_size = 1

        result = _build_token_frequency_csr(
            batch, padding_size, device, include_prompt=False
        )

        data_np = result.data.to_numpy()
        offsets_np = result.offsets.to_numpy()

        # Verify offsets shape: batch_size + 1
        assert offsets_np.shape == (3,)

        # Check context 1
        start1, end1 = offsets_np[0], offsets_np[1]
        data1 = data_np[start1:end1]
        valid1 = data1[data1[:, 0] != -1]
        freq1 = {int(row[0]): int(row[1]) for row in valid1}

        assert freq1.get(10) == 2
        assert freq1.get(11) == 1
        assert 20 not in freq1  # From context 2

        # Check context 2
        start2, end2 = offsets_np[1], offsets_np[2]
        data2 = data_np[start2:end2]
        valid2 = data2[data2[:, 0] != -1]
        freq2 = {int(row[0]): int(row[1]) for row in valid2}

        assert freq2.get(20) == 3
        assert freq2.get(21) == 1
        assert 10 not in freq2  # From context 1

    def test_batch_with_varying_lengths(self) -> None:
        """Test CSR building with contexts of different lengths."""
        # Short context
        context1 = create_text_context(
            prompt_tokens=np.array([1], dtype=np.int64),
            generated_tokens=np.array([2], dtype=np.int64),
        )

        # Medium context
        context2 = create_text_context(
            prompt_tokens=np.array([10, 11, 12], dtype=np.int64),
            generated_tokens=np.array([13, 14, 15, 16], dtype=np.int64),
        )

        # Long context with repetitions
        context3 = create_text_context(
            prompt_tokens=np.array([20, 21], dtype=np.int64),
            generated_tokens=np.array([22, 22, 22, 23, 23, 24], dtype=np.int64),
        )

        batch = [context1, context2, context3]
        device = CPU()
        padding_size = 3

        result = _build_token_frequency_csr(
            batch, padding_size, device, include_prompt=False
        )

        offsets_np = result.offsets.to_numpy()

        # Verify we have correct number of offsets
        assert offsets_np.shape == (4,)

        # Verify offsets are monotonically increasing
        for i in range(len(offsets_np) - 1):
            assert offsets_np[i] < offsets_np[i + 1]


class TestBuildTokenFrequencyCSREdgeCases:
    """Edge case tests for _build_token_frequency_csr."""

    def test_all_same_token(self) -> None:
        """Test CSR building when all tokens are the same."""
        prompt = np.array([5, 5, 5], dtype=np.int64)
        generated = np.array([5, 5, 5, 5], dtype=np.int64)

        context = create_text_context(prompt, generated)
        batch = [context]
        device = CPU()
        padding_size = 2

        result = _build_token_frequency_csr(
            batch, padding_size, device, include_prompt=True
        )

        data_np = result.data.to_numpy()
        offsets_np = result.offsets.to_numpy()

        start, end = offsets_np[0], offsets_np[1]
        context_data = data_np[start:end]
        valid_entries = context_data[context_data[:, 0] != -1]

        # Should have exactly one unique token
        assert len(valid_entries) == 1
        assert valid_entries[0, 0] == 5
        assert valid_entries[0, 1] == 7  # 3 prompt + 4 generated

    def test_all_unique_tokens(self) -> None:
        """Test CSR building when all tokens are unique."""
        prompt = np.array([1, 2, 3], dtype=np.int64)
        generated = np.array([4, 5, 6, 7], dtype=np.int64)

        context = create_text_context(prompt, generated)
        batch = [context]
        device = CPU()
        padding_size = 2

        result = _build_token_frequency_csr(
            batch, padding_size, device, include_prompt=True
        )

        data_np = result.data.to_numpy()
        offsets_np = result.offsets.to_numpy()

        start, end = offsets_np[0], offsets_np[1]
        context_data = data_np[start:end]
        valid_entries = context_data[context_data[:, 0] != -1]

        # Should have 7 unique tokens, each with count 1
        assert len(valid_entries) == 7
        for entry in valid_entries:
            assert entry[1] == 1

    def test_single_token_prompt(self) -> None:
        """Test CSR building with a single token prompt."""
        prompt = np.array([42], dtype=np.int64)
        generated = np.array([43, 44], dtype=np.int64)

        context = create_text_context(prompt, generated)
        batch = [context]
        device = CPU()
        padding_size = 1

        result = _build_token_frequency_csr(
            batch, padding_size, device, include_prompt=True
        )

        data_np = result.data.to_numpy()
        offsets_np = result.offsets.to_numpy()

        start, end = offsets_np[0], offsets_np[1]
        context_data = data_np[start:end]
        valid_entries = context_data[context_data[:, 0] != -1]
        freq_dict = {int(row[0]): int(row[1]) for row in valid_entries}

        assert freq_dict.get(42) == 1
        assert freq_dict.get(43) == 1
        assert freq_dict.get(44) == 1

    def test_single_generated_token(self) -> None:
        """Test CSR building with a single generated token."""
        prompt = np.array([1, 2, 3], dtype=np.int64)
        generated = np.array([99], dtype=np.int64)

        context = create_text_context(prompt, generated)
        batch = [context]
        device = CPU()
        padding_size = 1

        result = _build_token_frequency_csr(
            batch, padding_size, device, include_prompt=False
        )

        data_np = result.data.to_numpy()
        offsets_np = result.offsets.to_numpy()

        start, end = offsets_np[0], offsets_np[1]
        context_data = data_np[start:end]
        valid_entries = context_data[context_data[:, 0] != -1]

        assert len(valid_entries) == 1
        assert valid_entries[0, 0] == 99
        assert valid_entries[0, 1] == 1

    def test_large_token_ids(self) -> None:
        """Test CSR building with large token IDs (like Llama-3 vocab size)."""
        vocab_size = 128256
        large_tokens = np.array(
            [vocab_size - 1, vocab_size - 2, vocab_size - 1], dtype=np.int64
        )

        context = create_text_context(
            prompt_tokens=np.array([0, 1], dtype=np.int64),
            generated_tokens=large_tokens,
        )
        batch = [context]
        device = CPU()
        padding_size = 2

        result = _build_token_frequency_csr(
            batch, padding_size, device, include_prompt=False
        )

        data_np = result.data.to_numpy()
        offsets_np = result.offsets.to_numpy()

        start, end = offsets_np[0], offsets_np[1]
        context_data = data_np[start:end]
        valid_entries = context_data[context_data[:, 0] != -1]
        freq_dict = {int(row[0]): int(row[1]) for row in valid_entries}

        assert freq_dict.get(vocab_size - 1) == 2
        assert freq_dict.get(vocab_size - 2) == 1


class TestBuildTokenFrequencyCSRPadding:
    """Tests for padding behavior in _build_token_frequency_csr."""

    def test_padding_size_zero(self) -> None:
        """Test CSR building with zero padding."""
        prompt = np.array([1, 2], dtype=np.int64)
        generated = np.array([3, 4], dtype=np.int64)

        context = create_text_context(prompt, generated)
        batch = [context]
        device = CPU()
        padding_size = 0

        result = _build_token_frequency_csr(
            batch, padding_size, device, include_prompt=False
        )

        data_np = result.data.to_numpy()
        offsets_np = result.offsets.to_numpy()

        start, end = offsets_np[0], offsets_np[1]
        context_data = data_np[start:end]

        # No padding means only valid entries
        assert len(context_data) == 2  # 2 unique tokens

    def test_padding_included_correctly(self) -> None:
        """Test that padding entries have token_id=-1 and count=0."""
        prompt = np.array([1], dtype=np.int64)
        generated = np.array([2], dtype=np.int64)

        context = create_text_context(prompt, generated)
        batch = [context]
        device = CPU()
        padding_size = 5

        result = _build_token_frequency_csr(
            batch, padding_size, device, include_prompt=False
        )

        data_np = result.data.to_numpy()
        offsets_np = result.offsets.to_numpy()

        start, end = offsets_np[0], offsets_np[1]
        context_data = data_np[start:end]

        # Find padding entries
        padding_entries = context_data[context_data[:, 0] == -1]

        # Should have exactly padding_size padding entries
        assert len(padding_entries) == padding_size

        # All padding entries should have count=0
        for entry in padding_entries:
            assert entry[1] == 0


class TestBuildTokenFrequencyCSRDataTypes:
    """Tests for data type handling in _build_token_frequency_csr."""

    def test_output_tensor_types(self) -> None:
        """Test that output tensors have correct data types."""
        prompt = np.array([1, 2, 3], dtype=np.int64)
        generated = np.array([4, 5], dtype=np.int64)

        context = create_text_context(prompt, generated)
        batch = [context]
        device = CPU()
        padding_size = 2

        result = _build_token_frequency_csr(
            batch, padding_size, device, include_prompt=False
        )

        # Verify data array dtype is int64
        data_np = result.data.to_numpy()
        assert data_np.dtype == np.int32

        # Verify offsets array dtype is int64
        offsets_np = result.offsets.to_numpy()
        assert offsets_np.dtype == np.uint32

    def test_data_shape(self) -> None:
        """Test that data array has correct shape (N, 2)."""
        prompt = np.array([1, 2, 3], dtype=np.int64)
        generated = np.array([4, 5, 4], dtype=np.int64)

        context = create_text_context(prompt, generated)
        batch = [context]
        device = CPU()
        padding_size = 2

        result = _build_token_frequency_csr(
            batch, padding_size, device, include_prompt=False
        )

        data_np = result.data.to_numpy()

        # Should be 2D with second dimension = 2 (token_id, count)
        assert data_np.ndim == 2
        assert data_np.shape[1] == 2
