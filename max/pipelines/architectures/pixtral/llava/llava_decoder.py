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

from dataclasses import dataclass
from typing import Union

from max.dtype import DType
from max.graph import TensorValue, TensorValueLike, ops
from max.nn import (
    EmbeddingV1,
    LayerNormV1,
    LinearV1,
    RMSNormV1,
    TransformerBlock,
)
from max.nn.kv_cache import (
    FetchContinuousBatchingKVCacheCollection,
    FetchPagedKVCacheCollection,
    KVCacheParams,
)
from max.nn.layer import Layer


@dataclass
class Transformer(Layer):
    """Transformer model consisting for TransformerBlock layers.

    The differences between this transformer and the transformer in nn:

    - It takes as input the token embeddings rather than the token ids.
    - It skips the embedding generation (first step in nn.Transformer).

    TODO(AIPIPE-273): Once we have mo.if, we can update nn.Transformer
    to only generate embeddings if token ids are passed. That would
    eliminate the need for this class.
    """

    dim: int
    n_heads: int
    layers: list[TransformerBlock]
    norm: Union[RMSNormV1, LayerNormV1]
    output: LinearV1
    embedding: EmbeddingV1
    kv_params: KVCacheParams
    kv_collection_constructor: Union[
        FetchContinuousBatchingKVCacheCollection, FetchPagedKVCacheCollection
    ]
    all_logits: bool = False

    def __call__(
        self,
        embeds: TensorValue,
        kv_cache_inputs: tuple[
            TensorValue, TensorValue, TensorValue, TensorValue
        ],
        **kwargs,
    ) -> tuple[TensorValue, ...]:
        """Transformer model consisting of TransformerBlock layers.

        Args:
            embeds: embeddings of the sequence of text tokens and possibly images.
                shape = [batch_size, n_patches, hidden_dim]
            kv_cache_inputs: A tuple of 4 tensor values. In the case of paged attention,
                (blocks, cache_lengths, lookup_table, is_cache_empty). In the case of
                continuous attention, (blocks, cache_lengths, lookup_table, max_lengths).
        """
        h = embeds

        # Construct a kv cache for use downstream.
        kv_collection = self.kv_collection_constructor(*kv_cache_inputs)

        for _, layer in enumerate(self.layers):
            h = layer(h, kv_collection, **kwargs)

        if self.all_logits:
            # When echo is enabled, the logits of the input tokens are
            # returned.
            logits = ops.cast(self.output(self.norm(h)), DType.float32)
            if "input_row_offsets" in kwargs:
                # For ragged tensors gather the last tokens from packed dim 0.
                input_row_offsets: TensorValueLike = kwargs["input_row_offsets"]
                last_token_indices = input_row_offsets[1:] - 1  # type: ignore
                last_token_logits = ops.gather(
                    logits, last_token_indices, axis=0
                )
            else:
                # For padded tensors, use `gather_nd`.
                # Unsqueeze since `gather_nd` expects a static last dim.
                valid_lengths: TensorValueLike = kwargs["valid_lengths"]
                last_token_logits = ops.gather_nd(
                    logits,
                    indices=ops.unsqueeze(valid_lengths - 1, -1),  # type: ignore
                    batch_dims=1,
                )
            return (last_token_logits, logits)
        else:
            # Otherwise, only return the logits for the last non-pad token
            # (right-padded).
            if "input_row_offsets" in kwargs:
                # For ragged tensors gather the last tokens from packed dim 0.
                input_row_offsets = kwargs["input_row_offsets"]
                last_token_indices = input_row_offsets[1:] - 1  # type: ignore
                # Should be: last_token = h[last_token_indices]
                last_token = ops.gather(h, last_token_indices, axis=0)
            else:
                # For padded tensors, use `gather_nd`.
                # Unsqueeze since `gather_nd` expects a static last dim.
                valid_lengths = kwargs["valid_lengths"]
                last_token = ops.gather_nd(
                    h,
                    indices=ops.unsqueeze(valid_lengths - 1, -1),  # type: ignore
                    batch_dims=1,
                )

            # Always return float32 logits, no matter the activation type
            return (
                ops.cast(self.output(self.norm(last_token)), DType.float32),
            )
