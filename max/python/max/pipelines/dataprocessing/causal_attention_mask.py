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

"""Constructs causal attention masks for variable-length sequence batches."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

FILL_VAL = -10000.0


def _normalize_token_mask(
    token_mask: npt.ArrayLike,
    *,
    mask_name: str = "token_mask",
) -> npt.NDArray[np.bool_]:
    """Normalize a token mask to rank-2 bool array [batch, padded_seq_len]."""
    token_mask_np = np.asarray(token_mask)
    if token_mask_np.ndim == 1:
        token_mask_np = token_mask_np[np.newaxis, :]
    elif token_mask_np.ndim != 2:
        raise ValueError(
            f"{mask_name} must be rank-1 or rank-2, got shape {token_mask_np.shape}."
        )

    token_mask_np = token_mask_np.astype(np.bool_, copy=False)
    if token_mask_np.shape[1] == 0:
        raise ValueError(f"{mask_name} must have non-empty sequence length.")
    if not np.all(token_mask_np.any(axis=1)):
        raise ValueError(
            f"Each {mask_name} row must contain at least one valid token."
        )

    return token_mask_np


def causal_attention_mask(
    original_start_pos: list[int],
    original_seq_len: list[int],
) -> npt.NDArray[np.float32]:
    """Builds a causal attention mask for a batch of variable-length sequences.

    Args:
        original_start_pos: Per-example start position (context length) in the batch.
        original_seq_len: Per-example sequence length for this pass.

    Returns:
        Float32 mask array where visible positions are 0 and masked positions
        are a large negative value (so that softmax treats them as -inf).
    """
    # Each example in the batch has a "start position", which is the length
    # of the previously encoded tokens ("context"), and a "sequence length",
    # which is the number of additional tokens to be encoded in this pass.
    #
    # "Causal attention" means that each token can "see" tokens before it,
    # as well as itself.
    # The attention layer adds the mask to the attention scores and then
    # performs a softmax, so for tokens that a given token can "see" the mask
    # wants to produce a 0, meaning to pass the attention through as normal,
    # and for tokens that can't be "seen" the mask should produce -inf, which
    # will result in them being functionally ignored after the softmax operation.
    #
    # We call the total length "post_seq_len", referring to the total context
    # length after this pass concludes.
    start_pos: npt.NDArray[np.int64] = np.array(
        original_start_pos, dtype=np.int64
    )
    seq_len: npt.NDArray[np.int64] = np.array(original_seq_len, dtype=np.int64)

    # Use the maximum sequence length as the padded length
    padded_length = seq_len.max()

    # Mask shape: for each token being generated, attend to tokens _before_ it
    # in the entire sequence including context. Pad all values to the longest
    # sequence length and total length.
    post_seq_len = (start_pos + padded_length).max()
    mask_shape = (padded_length, post_seq_len)

    # TODO(KERN-782): This should be -inf but softmax saturates with NaNs.
    fill_matrix = np.full(mask_shape, FILL_VAL, dtype=np.float32)

    return np.stack(
        # Set diagonal to k + 1 so that tokens attend to themselves.
        [np.triu(fill_matrix, k=k + 1) for k in start_pos]
    )


def causal_attention_mask_with_token_mask(
    original_start_pos: list[int],
    token_mask: npt.ArrayLike,
    *,
    mask_name: str = "token_mask",
) -> npt.NDArray[np.float32]:
    """Builds a causal attention mask and additionally masks invalid tokens.

    Args:
        original_start_pos: Per-example start position (context length) in the batch.
        token_mask: Per-example validity mask for tokens in the current pass.
            Shape [seq_len] or [batch, seq_len]. `True` marks a valid token and
            `False` marks padding or any token that should be hidden.
        mask_name: Name used in validation errors.

    Returns:
        Float32 additive mask array where visible positions are 0 and masked
        positions are a large negative value.
    """
    token_mask_np = _normalize_token_mask(token_mask, mask_name=mask_name)
    batch_size, padded_length = token_mask_np.shape

    if len(original_start_pos) != batch_size:
        raise ValueError(
            "original_start_pos and token_mask batch size must match "
            f"({len(original_start_pos)} != {batch_size})."
        )

    additive_mask = causal_attention_mask(
        original_start_pos,
        [padded_length] * batch_size,
    )
    post_seq_len = additive_mask.shape[2]
    full_token_mask = np.ones((batch_size, post_seq_len), dtype=np.bool_)

    for batch_idx, start_pos in enumerate(original_start_pos):
        full_token_mask[batch_idx, start_pos : start_pos + padded_length] = (
            token_mask_np[batch_idx]
        )

    return np.where(
        full_token_mask[:, np.newaxis, :],
        additive_mask,
        np.float32(FILL_VAL),
    ).astype(np.float32, copy=False)
