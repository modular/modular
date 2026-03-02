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

"""Utilities for merging multimodal embeddings in InternVL."""

from __future__ import annotations

from max.graph import TensorValue, ops
from max.nn.kernels import scatter_nd_skip_oob_indices


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
