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
from typing import Callable

from max.dtype import DType
from max.graph import TensorValue, TensorValueLike, ops
from max.pipelines.kv_cache import (
    ContinuousBatchingKVCacheCollection,
    FetchContinuousBatchingKVCacheCollection,
    FetchPagedKVCacheCollection,
    FetchPagedKVCacheCollectionFA3Fallback,
    KVCacheParams,
    PagedKVCacheCollection,
)

from ..attention.interfaces import (
    AttentionImpl,
    AttentionImplQKV,
)
from ..embedding import Embedding, EmbeddingV2
from ..layer import Layer, LayerList, Module
from ..linear import Linear, LinearV2


class TransformerBlock(Module):
    """Stack of Attention, FeedForward, and RMSNorm layers."""

    def __init__(
        self,
        attention: AttentionImpl | AttentionImplQKV | Module,
        mlp: Layer,
        attention_norm: Layer,
        mlp_norm: Layer,
        residual_multiplier: float = 1.0,
    ):
        super().__init__()
        self.self_attn = attention
        self.mlp = mlp
        self.input_layernorm = attention_norm
        self.post_attention_layernorm = mlp_norm
        self.residual_multiplier = residual_multiplier

    def __call__(
        self,
        x: TensorValue,
        kv_collection: ContinuousBatchingKVCacheCollection
        | PagedKVCacheCollection,
        **kwargs,
    ) -> TensorValue:
        residual_multiplier = ops.constant(self.residual_multiplier, x.dtype)
        attn_out = self.self_attn(
            self.input_layernorm(x),
            kv_collection,
            **kwargs,
        )

        if self.residual_multiplier != 1.0:
            attn_out = attn_out * residual_multiplier

        h = x + attn_out
        mlp = self.mlp(self.post_attention_layernorm(h))
        if self.residual_multiplier != 1.0:
            mlp = mlp * residual_multiplier

        return h + mlp


class Transformer(Module):
    """Transformer model consisting for TransformerBlock layers."""

    def __init__(
        self,
        dim: int,
        n_heads: int,
        layers: list[TransformerBlock],
        norm: Layer,
        output: Linear | LinearV2,
        embedding: Embedding | EmbeddingV2,
        kv_params: KVCacheParams,
        kv_collection_constructor: (
            FetchContinuousBatchingKVCacheCollection
            | FetchPagedKVCacheCollection
            | FetchPagedKVCacheCollectionFA3Fallback
        ),
        all_logits: bool = False,
        embedding_multiplier: float = 1.0,
        logits_postprocessor: Callable[[TensorValue], TensorValue]
        | None = None,
    ):
        super().__init__()
        self.dim = dim
        self.n_heads = n_heads
        self.layers = LayerList(layers)
        self.norm = norm
        self.lm_head = output
        self.embed_tokens = embedding
        self.kv_params = kv_params
        self.kv_collection_constructor = kv_collection_constructor
        self.all_logits = all_logits
        self.embedding_multiplier = embedding_multiplier
        self.logits_postprocessor = logits_postprocessor

    def _apply_logits_postprocessor(
        self, output: tuple[TensorValue, ...]
    ) -> tuple[TensorValue, ...]:
        if self.logits_postprocessor is None:
            return output
        return tuple(self.logits_postprocessor(elem) for elem in output)

    def __call__(
        self,
        tokens: TensorValueLike,
        kv_cache_inputs: Sequence[TensorValue],
        **kwargs,
    ) -> tuple[TensorValue, ...]:
        # TODO: Split into a ragged and non-ragged version.
        h = self.embed_tokens(tokens)

        if self.embedding_multiplier != 1.0:
            h = h * ops.constant(self.embedding_multiplier, h.dtype)

        ragged = "input_row_offsets" in kwargs
        kv_collection = self.kv_collection_constructor(*kv_cache_inputs)
        cache_lengths = kv_cache_inputs[1]

        if ragged:
            input_row_offsets = kwargs["input_row_offsets"]
            prompt_lengths = ops.rebind(
                input_row_offsets[1:] - input_row_offsets[:-1],
                cache_lengths.shape,
            )
        else:
            prompt_lengths = kwargs["valid_lengths"]

        context_lengths = prompt_lengths + cache_lengths
        kwargs["context_lengths"] = context_lengths

        for _, layer in enumerate(self.layers):
            h = layer(h, kv_collection, **kwargs)

        normalized = self.norm(h)

        if "input_row_offsets" in kwargs:
            # Ragged inputs/activations
            last_indices = kwargs["input_row_offsets"][1:] - 1
            last_tokens = ops.gather(normalized, last_indices, axis=0)
        else:
            # Dense padded inputs/activations
            valid_lengths = kwargs["valid_lengths"]
            # TODO: Remove once `gather_nd` works with nonstatic last dims.
            indices = ops.unsqueeze(valid_lengths - 1, -1)
            last_tokens = ops.gather_nd(normalized, indices, batch_dims=1)

        # Always return float32 logits, no matter the activation type.
        last_token_logits = ops.cast(self.lm_head(last_tokens), DType.float32)

        if self.all_logits:
            all_logits = ops.cast(self.lm_head(normalized), DType.float32)
            return self._apply_logits_postprocessor(
                (last_token_logits, all_logits)
            )

        return self._apply_logits_postprocessor((last_token_logits,))
