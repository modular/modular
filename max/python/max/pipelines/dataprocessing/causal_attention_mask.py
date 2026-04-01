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
    token_mask_np = np.asarray(token_mask)
    if token_mask_np.ndim == 1:
        token_mask_np = token_mask_np[np.newaxis, :]
    elif token_mask_np.ndim != 2:
        raise ValueError(
            f"{mask_name} must be rank-1 or rank-2, got shape {token_mask_np.shape}."
        )
    token_mask_np = token_mask_np.astype(np.bool_, copy=False)
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


def attention_bias_from_attention_mask_array(
    attention_mask: npt.ArrayLike,
    *,
    expected_seq_len: int | None = None,
    batch_size: int = 1,
    start_pos: int = 0,
    mask_name: str = "attention_mask",
) -> npt.NDArray[np.float32]:
    """Converts a token attention mask into a 4D additive attention bias.

    Args:
        attention_mask: Rank-1 or rank-2 token-validity mask where ``True``
            marks visible tokens.
        expected_seq_len: Optional expected sequence length to validate.
        batch_size: Expected batch size for the materialized bias.
        start_pos: Start position used when embedding the token mask into a
            larger causal context.
        mask_name: Name used in validation errors.

    Returns:
        Float32 additive attention bias with shape ``[batch, 1, seq_len, seq_len]``.
    """
    additive_mask = causal_attention_mask_with_token_mask(
        [start_pos] * batch_size,
        attention_mask,
        mask_name=mask_name,
    )
    if additive_mask.shape[0] != batch_size:
        raise ValueError(
            f"batch size must be {batch_size}, got {additive_mask.shape[0]}."
        )
    if (
        expected_seq_len is not None
        and additive_mask.shape[1] != expected_seq_len
    ):
        raise ValueError(
            "seq_len must match tokens "
            f"({additive_mask.shape[1]} != {expected_seq_len})."
        )
    return additive_mask[:, np.newaxis, :, :].astype(np.float32, copy=False)
