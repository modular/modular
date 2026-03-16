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
"""Pooling functions for Qwen3 Embedding models."""

from max.dtype import DType
from max.experimental import functional as F
from max.experimental.tensor import Tensor


def last_token_pool(
    hidden_states: Tensor,
    input_row_offsets: Tensor,
) -> Tensor:
    """Apply last token pooling to extract embeddings.

    Extracts the hidden state of the last token for each sequence in the batch.
    """
    end_offsets = input_row_offsets[1:]
    end_offsets = end_offsets.to(hidden_states.device)

    one = F.constant(1, DType.uint32, device=hidden_states.device)
    last_token_indices = end_offsets - one
    last_token_indices_i32 = last_token_indices.cast(DType.int32)

    pooled = F.gather(hidden_states, last_token_indices_i32, axis=0)
    return pooled


def normalize_embeddings(embeddings: Tensor) -> Tensor:
    """Apply L2 normalization to embeddings."""
    embeddings_f32 = embeddings.cast(DType.float32)

    embeddings_squared = embeddings_f32 * embeddings_f32
    norm_squared = F.sum(embeddings_squared, axis=-1)
    epsilon = F.constant(1e-12, DType.float32, embeddings_f32.device)
    norm = F.sqrt(norm_squared + epsilon)
    embeddings_normalized = embeddings_f32 / norm

    return embeddings_normalized
