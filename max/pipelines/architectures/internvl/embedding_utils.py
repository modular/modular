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
    input_ids: TensorValue,
    inputs_embeds: TensorValue,
    multimodal_embeddings: TensorValue,
    image_context_token_id: int,
) -> TensorValue:
    """Merges multimodal embeddings into text embeddings at image context token positions.

    This is the MAX Graph API implementation of the embedding merge operation.
    It performs an in-place update of inputs_embeds with multimodal embeddings
    at positions marked by image context tokens.

    Args:
        input_ids: Input token IDs with shape [num_tokens].
        inputs_embeds: Text embeddings with shape [num_tokens, hidden_size].
        multimodal_embeddings: Vision embeddings to insert with shape
            [num_multimodal_tokens, hidden_size].
        image_context_token_id: Token ID marking where to insert multimodal embeddings.

    Returns:
        The inputs_embeds tensor with multimodal embeddings merged in-place.
    """
    # Create mask for image context tokens
    mask = input_ids == image_context_token_id  # Shape: [num_tokens]
    mask = ops.unsqueeze(mask, -1)  # Shape: [num_tokens, 1]
    mask = ops.broadcast_to(
        mask, inputs_embeds.shape
    )  # Shape: [num_tokens, hidden_size]

    # Use masked_scatter to replace image context tokens with multimodal embeddings
    # This operation scatters multimodal_embeddings into inputs_embeds
    # at positions where mask is True
    result = ops.masked_scatter(
        inputs_embeds,
        mask,
        multimodal_embeddings,
        out_dim="unmasked_inputs",
    )

    return result
