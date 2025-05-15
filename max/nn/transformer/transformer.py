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
from enum import Enum
from typing import Callable, TypeVar, cast

from max.dtype import DType
from max.graph import DeviceRef, TensorValue, TensorValueLike, ops

from ..attention.interfaces import AttentionImpl, AttentionImplQKV
from ..embedding import Embedding, EmbeddingV1
from ..kv_cache import (
    ContinuousBatchingKVCacheCollection,
    FetchContinuousBatchingKVCacheCollection,
    FetchPagedKVCacheCollection,
    KVCacheParams,
    PagedKVCacheCollection,
)
from ..layer import Layer, LayerList, Module
from ..linear import Linear, LinearV1


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
        residual_multiplier = ops.constant(
            self.residual_multiplier, x.dtype, device=x.device
        )
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


class ReturnLogits(str, Enum):
    LAST_TOKEN = "last_token"
    VARIABLE = "variable"
    ALL = "all"


Block = TypeVar("Block", bound=Module, covariant=True)


class Transformer(Module):
    """Transformer model consisting for TransformerBlock layers."""

    def __init__(
        self,
        dim: int,
        n_heads: int,
        layers: list[Block],
        norm: Layer,
        output: LinearV1 | Linear,
        embedding: EmbeddingV1 | Embedding,
        kv_params: KVCacheParams,
        kv_collection_constructor: (
            FetchContinuousBatchingKVCacheCollection
            | FetchPagedKVCacheCollection
        ),
        return_logits: ReturnLogits = ReturnLogits.LAST_TOKEN,
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
        self.embedding_multiplier = embedding_multiplier
        self.logits_postprocessor = logits_postprocessor
        self.return_logits = return_logits

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
        return_n_logits: TensorValue,
        **kwargs,
    ) -> tuple[TensorValue, ...]:
        h = self.embed_tokens(tokens)

        if self.embedding_multiplier != 1.0:
            h = h * ops.constant(
                self.embedding_multiplier, h.dtype, device=h.device
            )

        kv_collection = self.kv_collection_constructor(*kv_cache_inputs)
        input_row_offsets = kwargs["input_row_offsets"]

        for _, layer in enumerate(self.layers):
            h = layer(
                h,
                kv_collection,
                **kwargs,
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
                device=DeviceRef.CPU(),
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
                device=DeviceRef.CPU(),
                dtype=DType.int64,
            )
        elif self.return_logits == ReturnLogits.ALL:
            logits = ops.cast(self.lm_head(self.norm(h)), DType.float32)
            offsets = cast(TensorValue, kwargs["input_row_offsets"])

        if logits:
            last_logits, logits = self._apply_logits_postprocessor(
                (
                    last_logits,
                    logits,
                )
            )
        else:
            last_logits = self._apply_logits_postprocessor((last_logits,))[0]

        if offsets is not None:
            assert logits is not None
            return (last_logits, logits, offsets)
        else:
            return (last_logits,)
