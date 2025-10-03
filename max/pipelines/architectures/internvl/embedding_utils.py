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

"""Utilities for merging multimodal embeddings in InternVL."""

from __future__ import annotations

from max.graph import TensorValue, ops


def merge_multimodal_embeddings(
    inputs_embeds: TensorValue,
    multimodal_embeddings: TensorValue,
    image_token_indices: TensorValue,
) -> TensorValue:
    """Merges multimodal embeddings into text embeddings at pre-computed indices.

    This is the MAX Graph API implementation of the embedding merge operation.
    It returns an updated copy of inputs_embeds with multimodal embeddings
    at positions specified by the indices.

    Args:
        inputs_embeds: Text embeddings with shape [num_tokens, hidden_size].
        multimodal_embeddings: Vision embeddings to insert with shape
            [num_multimodal_tokens, hidden_size].
        image_token_indices: Pre-computed indices where to insert multimodal embeddings,
            with shape [num_multimodal_tokens].

    Returns:
        Copy of the inputs_embeds tensor with multimodal embeddings merged in.
    """
    # Use scatter_nd to directly place embeddings at specified indices.
    # Expand indices to 2D for scatter_nd: [num_tokens, 1]
    indices_2d = ops.unsqueeze(image_token_indices, -1)

    # Scatter the multimodal embeddings into inputs_embeds at the specified
    # indices.
    return ops.scatter_nd(
        inputs_embeds, updates=multimodal_embeddings, indices=indices_2d
    )
