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
"""Checks ragged draft-prefix merges against independent row and uniform oracles."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from max.driver import CPU, Buffer
from max.dtype import DType
from max.engine import InferenceSession, Model
from max.graph import DeviceRef, TensorType
from max.pipelines.speculative.ragged_token_merger import ragged_token_merger

MergePrefixes = Callable[
    [np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    tuple[np.ndarray, np.ndarray],
]


def prefix_input_types(device: DeviceRef) -> tuple[TensorType, ...]:
    """Returns the prompt, offset, draft and length input types.

    Args:
        device: The device for all four inputs.

    Returns:
        The ordered tensor types for packed prompts, row offsets, draft rows
        and per-row verify lengths.
    """
    return (
        TensorType(DType.int64, ["prompt_len"], device=device),
        TensorType(DType.uint32, ["offsets_len"], device=device),
        TensorType(DType.int64, ["batch_size", "draft_len"], device=device),
        TensorType(DType.int32, ["batch_size"], device=device),
    )


def merge_outputs(
    model: Model, *arrays: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Returns the model's merged tokens and row offsets.

    Args:
        model: The compiled uniform or per-row prefix merger.
        arrays: Prompt tokens (int64), offsets (uint32), and drafts (int64),
            followed for per-row merges by kept lengths (int32) and, for a
            symbolic extent, a packed-length buffer (uint8).

    Returns:
        Merged tokens (int64) and cumulative row offsets (uint32).
    """
    tokens, offsets = model.execute(
        *(Buffer.from_numpy(array) for array in arrays)
    )
    assert isinstance(tokens, Buffer)
    assert isinstance(offsets, Buffer)
    return tokens.to_numpy(), offsets.to_numpy()


def _reference_merge(
    prompt_tokens: np.ndarray,
    prompt_lengths: list[int],
    draft_tokens: np.ndarray,
    verify_lengths: list[int],
) -> tuple[np.ndarray, np.ndarray]:
    """Merges each prompt with its kept draft prefix in row order."""
    merged_rows = []
    offsets = [0]
    prompt_offsets = np.cumsum([0, *prompt_lengths])
    for row, length in enumerate(prompt_lengths):
        kept = draft_tokens[row, : verify_lengths[row]]
        merged_rows.append(
            prompt_tokens[prompt_offsets[row] : prompt_offsets[row] + length]
        )
        merged_rows.append(kept)
        offsets.append(offsets[-1] + length + verify_lengths[row])
    merged = np.concatenate(merged_rows) if merged_rows else np.array([])
    return merged, np.array(offsets, dtype=np.uint32)


def _check_case(
    merge: MergePrefixes,
    prompt_lengths: list[int],
    draft_tokens: np.ndarray,
    verify_lengths: list[int],
) -> None:
    prompt_tokens = (
        np.arange(sum(prompt_lengths), dtype=draft_tokens.dtype) + 100
    )
    prompt_offsets = np.array(np.cumsum([0, *prompt_lengths]), dtype=np.uint32)
    lengths = np.array(verify_lengths, dtype=np.int32)
    got_tokens, got_offsets = merge(
        prompt_tokens, prompt_offsets, draft_tokens, lengths
    )
    want_tokens, want_offsets = _reference_merge(
        prompt_tokens, prompt_lengths, draft_tokens, verify_lengths
    )
    np.testing.assert_array_equal(got_tokens, want_tokens)
    np.testing.assert_array_equal(got_offsets, want_offsets)


def check_decode_rows_with_zero_and_partial_lengths(
    merge: MergePrefixes,
) -> None:
    """Checks full, empty and partial decode prefixes."""
    drafts = np.arange(15, dtype=np.int64).reshape(3, 5) + 200
    _check_case(merge, [1, 1, 1], drafts, [5, 0, 2])


def check_mixed_prefill_rows(merge: MergePrefixes) -> None:
    """Checks mixed prefill rows."""
    drafts = np.arange(15, dtype=np.int64).reshape(3, 5) + 200
    _check_case(merge, [4, 1, 3], drafts, [1, 5, 0])


def check_uniform_lengths_match_existing_merger(merge: MergePrefixes) -> None:
    """Compares full prefixes with the uniform merger."""
    drafts = np.arange(15, dtype=np.int64).reshape(3, 5) + 200
    prompt_lengths = [2, 3, 1]
    prompt_tokens = np.arange(sum(prompt_lengths), dtype=np.int64) + 100
    prompt_offsets = np.array(np.cumsum([0, *prompt_lengths]), dtype=np.uint32)
    uniform_model = InferenceSession(devices=[CPU()]).load(
        ragged_token_merger(DeviceRef.CPU())
    )
    uniform_tokens, uniform_offsets = merge_outputs(
        uniform_model, prompt_tokens, prompt_offsets, drafts
    )
    got_tokens, got_offsets = merge(
        prompt_tokens,
        prompt_offsets,
        drafts,
        np.array([5, 5, 5], dtype=np.int32),
    )
    np.testing.assert_array_equal(got_tokens, uniform_tokens)
    np.testing.assert_array_equal(got_offsets, uniform_offsets)


def check_single_row(merge: MergePrefixes) -> None:
    """Checks full and empty prefixes on a single row."""
    drafts = np.arange(5, dtype=np.int64).reshape(1, 5) + 200
    _check_case(merge, [2], drafts, [5])
    _check_case(merge, [2], drafts, [0])


def check_empty_first_row_full_second_row(merge: MergePrefixes) -> None:
    """Checks an empty prefix followed by a full prefix."""
    drafts = np.arange(10, dtype=np.int64).reshape(2, 5) + 200
    _check_case(merge, [1, 3], drafts, [0, 5])


def check_seeded_random_lengths(merge: MergePrefixes) -> None:
    """Checks seeded random lengths against the row oracle."""
    rng = np.random.default_rng(1234)
    drafts = rng.integers(0, 1000, size=(8, 5)).astype(np.int64)
    lengths = rng.multinomial(20, [1 / 8] * 8).astype(np.int32)
    lengths = np.minimum(lengths, 5)
    lengths[0] += 20 - int(lengths.sum())
    _check_case(merge, [3, 1, 2, 1, 4, 2, 1, 1], drafts, lengths.tolist())
