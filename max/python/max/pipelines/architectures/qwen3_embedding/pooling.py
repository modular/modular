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
    attention_mask: TensorValue,
    input_row_offsets: TensorValue,
) -> TensorValue:
    """Apply last token pooling to extract embeddings.
    
    This function extracts the hidden state of the last non-padding token
    for each sequence in the batch, as used by Qwen3-Embedding models.
    
    Args:
        hidden_states: Output from the transformer model in ragged format [total_seq_len, hidden_size]
        attention_mask: Attention mask [batch_size, seq_len] marking valid tokens (1) vs padding (0)
        input_row_offsets: Row offsets defining sequence boundaries in flattened format [batch_size + 1]
    
    Returns:
        Pooled embeddings [batch_size, hidden_size]
    """
    # Compute sequence lengths by summing attention mask
    # attention_mask shape: [batch_size, seq_len]
    sequence_lengths = ops.sum(attention_mask, axis=1)  # [batch_size], float32
    
    # Convert to int64 for indexing
    sequence_lengths_int = ops.cast(sequence_lengths, DType.int64)
    
    # Get starting offsets for each sequence using Python slicing
    # For batch_size=1: input_row_offsets = [0, total_len], so start_offsets = [0]
    start_offsets = input_row_offsets[:-1]  # Remove last element
    start_offsets_int = ops.cast(start_offsets, DType.int64)
    
    # Compute the index of the last valid token for each sequence
    # last_index = start_offset + sequence_length - 1
    one = ops.constant(1, DType.int64, device=hidden_states.device)
    last_token_indices = start_offsets_int + sequence_lengths_int - one
    
    # Use gather to extract the embeddings - cast to int32 for gather
    last_token_indices_i32 = ops.cast(last_token_indices, DType.int32)
    
    # Gather the hidden states at the last valid token positions
    # For batch_size=1, this will give us [batch_size, hidden_size]
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
    embeddings_f32 = ops.cast(embeddings, DType.float32)
    
    # Apply L2 normalization: embeddings / ||embeddings||_2
    # This matches the upstream Qwen3-Embedding implementation which uses F.normalize(embeddings, p=2, dim=1)
    # Compute squared values
    embeddings_squared = ops.mul(embeddings_f32, embeddings_f32)
    # Sum along the last dimension (hidden_size) to get L2 norm squared for each sample
    # ops.sum keeps dimensions, so result is [batch_size, 1]
    norm_squared = ops.sum(embeddings_squared, axis=-1)
    # Compute L2 norm (sqrt of sum of squares) with epsilon for numerical stability
    epsilon = ops.constant(1e-12, DType.float32, embeddings_f32.device)
    norm = ops.sqrt(ops.add(norm_squared, epsilon))
    # Normalize: embeddings / norm
    # Broadcasting: [batch_size, hidden_size] / [batch_size, 1] -> [batch_size, hidden_size]
    embeddings_normalized = ops.div(embeddings_f32, norm)
    
    return embeddings_normalized
