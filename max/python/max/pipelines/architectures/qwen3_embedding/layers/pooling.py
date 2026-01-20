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
"""Pooling functions for Qwen3 Embedding models."""

from max.dtype import DType
from max.graph import TensorValue, ops


def last_token_pool(
    hidden_states: TensorValue,
    input_row_offsets: TensorValue,
) -> TensorValue:
    """Apply last token pooling to extract embeddings.

    This function extracts the hidden state of the last token
    for each sequence in the batch, as used by Qwen3-Embedding models.

    Args:
        hidden_states: Output from the transformer model in ragged format [total_seq_len, hidden_size]
        input_row_offsets: Row offsets defining sequence boundaries in flattened format [batch_size + 1]

    Returns:
        Pooled embeddings [batch_size, hidden_size]
    """
    # Compute sequence lengths from row offsets
    # For each sequence i: length = row_offsets[i+1] - row_offsets[i]
    # Get ending offsets for each sequence
    end_offsets = input_row_offsets[1:]  # Remove first element

    # Move end_offsets to the same device as hidden_states
    end_offsets = end_offsets.to(hidden_states.device)

    # Compute last token index for each sequence (end_offset - 1)
    # Since end_offset is exclusive, the last token is at end_offset - 1
    one = ops.constant(1, DType.uint32, device=hidden_states.device)
    last_token_indices = end_offsets - one

    # Cast to int32 for gather operation
    last_token_indices_i32 = last_token_indices.cast(DType.int32)

    # Gather the hidden states at the last token positions
    # This extracts embeddings for each sequence [batch_size, hidden_size]
    pooled = ops.gather(hidden_states, last_token_indices_i32, axis=0)

    return pooled


def normalize_embeddings(embeddings: TensorValue) -> TensorValue:
    """Apply L2 normalization to embeddings.

    Args:
        embeddings: Embeddings to normalize [batch_size, hidden_size]

    Returns:
        Normalized embeddings [batch_size, hidden_size]
    """
    # Cast to float32 BEFORE normalization for better numerical precision
    embeddings_f32 = embeddings.cast(DType.float32)

    # Apply L2 normalization: embeddings / ||embeddings||_2
    # This matches the upstream Qwen3-Embedding implementation which uses F.normalize(embeddings, p=2, dim=1)
    # Compute squared values
    embeddings_squared = embeddings_f32 * embeddings_f32
    # Sum along the last dimension (hidden_size) to get L2 norm squared for each sample
    # ops.sum keeps dimensions, so result is [batch_size, 1]
    norm_squared = ops.sum(embeddings_squared, axis=-1)
    # Compute L2 norm (sqrt of sum of squares) with epsilon for numerical stability
    epsilon = ops.constant(1e-12, DType.float32, embeddings_f32.device)
    norm = ops.sqrt(norm_squared + epsilon)
    # Normalize: embeddings / norm
    # Broadcasting: [batch_size, hidden_size] / [batch_size, 1] -> [batch_size, hidden_size]
    embeddings_normalized = embeddings_f32 / norm

    return embeddings_normalized
