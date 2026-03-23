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

"""Shared utilities for vision-language model (VLM) architectures."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

import numpy as np
import numpy.typing as npt
from max.graph import TensorValue, ops
from max.interfaces.tokens import TokenBuffer
from max.nn.kernels import scatter_nd_skip_oob_indices


class VLMContextWithImageIndices(Protocol):
    """Protocol for VLM contexts that carry pre-computed image token indices."""

    @property
    def needs_vision_encoding(self) -> bool:
        """Whether this context requires vision encoding."""
        ...

    image_token_indices: npt.NDArray[np.int32]
    tokens: TokenBuffer


def merge_multimodal_embeddings(
    inputs_embeds: TensorValue,
    multimodal_embeddings: TensorValue,
    image_token_indices: TensorValue,
) -> TensorValue:
    """Merges multimodal embeddings into text embeddings at pre-computed indices.

    This is the MAX Graph API implementation of the embedding merge operation.
    It returns an updated copy of inputs_embeds with multimodal embeddings
    at positions specified by the indices.

    Indices may be oob (out of bounds), in which case the corresponding update will be skipped.

    Args:
        inputs_embeds: Text embeddings with shape [num_tokens, hidden_size].
        multimodal_embeddings: Vision embeddings to insert with shape
            [num_multimodal_tokens, hidden_size].
        image_token_indices: Pre-computed indices where to insert multimodal embeddings,
            with shape [num_multimodal_tokens].

    Returns:
        Copy of the inputs_embeds tensor with multimodal embeddings merged in.
    """
    # Use scatter_nd_skip_oob_indices to directly place embeddings at specified indices.
    # Expand indices to 2D for scatter_nd_skip_oob_indices: [num_tokens, 1]
    indices_2d = ops.unsqueeze(image_token_indices, -1)

    if multimodal_embeddings.dtype != inputs_embeds.dtype:
        multimodal_embeddings = ops.cast(
            multimodal_embeddings, dtype=inputs_embeds.dtype
        )

    # Scatter the multimodal embeddings into inputs_embeds at the specified
    # indices. Any negative values in the indices means that the corresponding
    # update will be skipped.
    return scatter_nd_skip_oob_indices(
        input=inputs_embeds,
        updates=multimodal_embeddings,
        indices=indices_2d,
    )


def compute_multimodal_merge_indices(
    batch: Sequence[VLMContextWithImageIndices],
) -> npt.NDArray[np.int32]:
    """Compute indices for a batch of VLM contexts to use in merge_multimodal_embeddings.

    Args:
        batch: Sequence of VLM contexts.

    Returns:
        npt.NDArray[np.int32]: Multimodal merge indices, some of which may be negative.
    """
    # Calculate sentinel OOB index value by finding the largest negative int32 value.
    oob_idx = np.iinfo(np.int32).min

    # Collect indices and offsets.
    indices_list = []
    total_active_tokens = 0

    for ctx in batch:
        if ctx.needs_vision_encoding:
            # This logic is quite tricky but is required for VLM prefix caching.
            # In the current approach, we run image decoding on all images.
            # We then select the rows of the image embeddings we want to use.
            # This may not be all of the rows in the event of a prefix cache
            # hit. This is done via a multimodal merge operation which filters
            # out negative indices.

            # First, get the pre-computed indices of where the image placeholder
            # tokens are in the prompt. This is populated by tokenizer.
            # eg: prompt = [0, 1, 2, 3, IMG, IMG, IMG, IMG, 8, 9, IMG, IMG]
            #    indices = [4, 5, 6, 7, 10, 11]
            indices = ctx.image_token_indices

            # Subtract all of the indices by the start_idx to get offsets
            # relative to the ragged next_tokens input sequence.
            # eg: start_idx = 6
            #     indices = [-2, -1, 0, 1, 4, 5]
            indices = indices - ctx.tokens.processed_length

            # Set any negative indices to -1, which means that they are ignored.
            # Bump remaining by accumulated value for the batch.
            indices_filtered = [
                idx + total_active_tokens if idx >= 0 else oob_idx
                for idx in indices.tolist()
            ]

            # Final scatter indices assuming the batch has 10 image tokens so far.
            # eg: indices_filtered = [-999, -999, 10, 11, 14, 15]
            #     This means that we will copy 4 image embeddings to the rows
            #     10-11 and 14-15 of the text embeddings.
            indices_list.append(indices_filtered)

        total_active_tokens += ctx.tokens.active_length

    # scatter_nd_skip_oob_indices uses int32 indices.
    if indices_list:
        return np.concatenate(indices_list, dtype=np.int32)
    else:
        return np.array([], dtype=np.int32)
