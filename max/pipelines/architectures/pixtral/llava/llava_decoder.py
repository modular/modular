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
from __future__ import annotations

from collections.abc import Sequence

from max.dtype import DType
from max.graph import DeviceRef, TensorValue, ops
from max.nn import (
    Embedding,
    Layer,
    LayerList,
    Linear,
    Module,
    ReturnLogits,
    TransformerBlock,
)
from max.nn.kv_cache import (
    FetchPagedKVCacheCollection,
    KVCacheParams,
)
from max.nn.rotary_embedding import RotaryEmbedding


class Transformer(Module):
    """Transformer model consisting for TransformerBlock layers.

    The differences between this transformer and the transformer in nn:

    - It takes as input the token embeddings rather than the token ids.
    - It skips the embedding generation (first step in nn.Transformer).

    TODO(AIPIPE-273): Once we have mo.if, we can update nn.Transformer
    to only generate embeddings if token ids are passed. That would
    eliminate the need for this class.
    """

    def __init__(
        self,
        dim: int,
        n_heads: int,
        layers: list[TransformerBlock],
        norm: Layer,
        output: Linear,
        embedding: Embedding,
        kv_params: KVCacheParams,
        kv_collection_constructor: FetchPagedKVCacheCollection,
        rope: RotaryEmbedding,
        return_logits: ReturnLogits = ReturnLogits.LAST_TOKEN,
        embedding_multiplier: float = 1.0,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.n_heads = n_heads
        self.layers = LayerList(layers)
        self.norm = norm
        self.lm_head = output
        self.embed_tokens = embedding
        self.kv_params = kv_params
        self.kv_collection_constructor = kv_collection_constructor
        self.embedding_multiplier = embedding_multiplier
        self.rope = rope
        self.return_logits = return_logits

    def __call__(
        self,
        embeds: TensorValue,
        kv_cache_inputs: Sequence[TensorValue],
        return_n_logits: TensorValue,
        input_row_offsets: TensorValue,
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
        kv_collection = self.kv_collection_constructor(*kv_cache_inputs)

        freqs_cis = self.rope.freqs_cis
        for idx, layer in enumerate(self.layers):
            h = layer(
                ops.constant(idx, DType.uint32, device=DeviceRef.CPU()),
                h,
                kv_collection,
                freqs_cis=freqs_cis,
                input_row_offsets=input_row_offsets,
            )

        # Retrieve a variable number of tokens
        last_h = ops.gather(h, input_row_offsets[1:] - 1, axis=0)
        last_logits = ops.cast(self.lm_head(self.norm(last_h)), DType.float32)
        logits = None
        offsets = None

        if self.return_logits == ReturnLogits.VARIABLE:
            return_n_logits_range = ops.range(
                return_n_logits[0],
                0,
                -1,
                out_dim="return_n_logits_range",
                device=h.device,
                dtype=DType.int64,
            )
            offsets = (
                ops.unsqueeze(input_row_offsets[1:], -1) - return_n_logits_range
            )
            last_indices = ops.reshape(offsets, shape=(-1,))
            last_tokens = ops.gather(h, last_indices, axis=0)
            logits = ops.cast(
                self.lm_head(self.norm(last_tokens)), DType.float32
            )
            offsets = ops.range(
                0,
                TensorValue(last_indices.shape[0]) + return_n_logits[0],
                return_n_logits[0],
                out_dim="logit_offsets",
                device=h.device,
                dtype=DType.int64,
            )
        elif self.return_logits == ReturnLogits.ALL:
            logits = ops.cast(self.lm_head(self.norm(h)), DType.float32)
            offsets = input_row_offsets

        if offsets is not None:
            assert logits is not None
            return (last_logits, logits, offsets)
        else:
            return (last_logits,)
