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
"""Batch padding."""

from __future__ import annotations

import enum
from typing import Any

import numpy as np
import numpy.typing as npt

from .causal_attention_mask import causal_attention_mask


class PaddingDirection(enum.Enum):
    """Padding (from) direction for batch collation."""

    LEFT = "left"
    RIGHT = "right"


def collate_batch(
    batch: list[npt.NDArray[np.integer[Any]]],
    direction: PaddingDirection = PaddingDirection.RIGHT,
    pad_value: int = 0,
    batch_size: int | None = None,
) -> tuple[npt.NDArray[np.integer[Any]], npt.NDArray[np.integer[Any]]]:
    """Generates a single batch tensor from a batch of inputs.

    These input tensors may have different lengths. The `pad_value` will be used
    to pad out the inputs to the same length.

    If `batch_size` is present, add additional values to the batch up to that
    size.

    Returns:
        A tuple of:
            A matrix with all rows padded to the max sequence length.
            A list with last token indices prior to any padding.

    Raises:
        ValueError: if the batch is empty.
        NotImplementedError: if the batch contains anything other than vectors.
    """
    if not batch:
        msg = "Must provide at least one batch item."
        raise ValueError(msg)

    if not all(a.ndim == 1 for a in batch):
        msg = "Collate only supports rank 1 tensors for now."
        raise NotImplementedError(msg)

    max_len = max((len(a) for a in batch), default=0)
    pad_to = max_len

    def pad(a: npt.NDArray[np.integer[Any]]) -> npt.NDArray[np.integer[Any]]:
        npad = pad_to - len(a)
        if npad == 0:
            return a
        padding = (npad, 0) if direction == PaddingDirection.LEFT else (0, npad)
        return np.pad(a, padding, mode="constant", constant_values=pad_value)

    if batch_size is not None:
        pad_batch_item = np.array([pad_value] * pad_to)
        batch.extend([pad_batch_item] * (batch_size - len(batch)))

    # Generate unpadded last token index.
    unpadded_last_token_index = (
        np.full(len(batch), -1)
        if direction == PaddingDirection.LEFT
        else np.array([len(a) - 1 for a in batch])
    )

    return np.stack([pad(a) for a in batch], axis=0), unpadded_last_token_index


def batch_padded_tokens_and_mask(
    start_pos: list[int],
    tokens: list[npt.NDArray[np.integer[Any]]],
) -> tuple[
    npt.NDArray[np.integer[Any]],
    npt.NDArray[np.integer[Any]],
    npt.NDArray[np.float32],
]:
    """Batches input tokens and computes a batched attention mask.

    Args:
        start_pos: index into the end of the KV cache for each batch item.
        tokens: unpadded input tokens for this batch.

    Returns:
        A (batched tokens, unpadded last token indices, batch attention mask) pair.
    """
    # Grab attention mask.
    attn_mask = causal_attention_mask(
        original_start_pos=start_pos,
        original_seq_len=[len(t) for t in tokens],
    )

    # Create batched input token tensor by padding all input token tensors
    # to the maximum sequence length in the batch.
    next_tokens_batch, unpadded_last_token_index = collate_batch(
        tokens, batch_size=len(tokens)
    )
    return next_tokens_batch, unpadded_last_token_index, attn_mask
