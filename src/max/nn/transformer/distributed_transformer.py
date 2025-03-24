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

from dataclasses import dataclass

from max.dtype import DType
from max.graph import BufferValue, DeviceRef, TensorValue, TensorValueLike, ops
from max.pipelines.kv_cache import (
    ContinuousBatchingKVCacheCollection,
    FetchContinuousBatchingKVCacheCollection,
    FetchPagedKVCacheCollection,
    FetchPagedKVCacheCollectionFA3Fallback,
    KVCacheParams,
    PagedKVCacheCollection,
)

from ..embedding import VocabParallelEmbedding
from ..layer import LayerList, Module
from ..linear import ColumnParallelLinear
from ..norm import DistributedRMSNorm


# TODO (pavan): clean up duplicate instances of distribute_value, shard_col_value,
# shard_row_value across the codebase into a multi gpu utils file
def distribute_value(v, devices: list[DeviceRef]):
    return [v.to(device) for device in devices]


class DistributedTransformerBlock(Module):
    """Stack of Attention, FeedForward, and RMSNorm layers."""

    def __init__(
        self,
        attention: Module,
        mlp: Module,
        attention_norm: DistributedRMSNorm,
        mlp_norm: DistributedRMSNorm,
        devices: list[DeviceRef],
    ):
        super().__init__()
        self.self_attn = attention
        self.mlp = mlp
        self.input_layernorm = attention_norm
        self.post_attention_layernorm = mlp_norm
        self.devices = devices

    def __call__(
        self,
        xs: list[TensorValue],
        signal_buffers: list[BufferValue],
        kv_collections: list[
            ContinuousBatchingKVCacheCollection | PagedKVCacheCollection
        ],
        **kwargs,
    ) -> list[TensorValue]:
        attn_outs = self.self_attn(
            self.input_layernorm(xs), signal_buffers, kv_collections, **kwargs
        )

        hs = [x + attn_out for x, attn_out in zip(xs, attn_outs)]
        mlp_outs = self.mlp(self.post_attention_layernorm(hs), signal_buffers)
        hs = [h + mlp_out for h, mlp_out in zip(hs, mlp_outs)]

        return hs


@dataclass
class DistributedTransformer(Module):
    """Transformer model consisting for TransformerBlock layers."""

    def __init__(
        self,
        dim: int,
        n_heads: int,
        layers: list[DistributedTransformerBlock],
        norm: DistributedRMSNorm,
        output: ColumnParallelLinear,
        embedding: VocabParallelEmbedding,
        kv_params: KVCacheParams,
        kv_collection_constructor: (
            FetchContinuousBatchingKVCacheCollection
            | FetchPagedKVCacheCollection
            | FetchPagedKVCacheCollectionFA3Fallback
        ),
        devices: list[DeviceRef],
        return_n_logits: int = 1,
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
        self.return_n_logits = return_n_logits
        self.devices = devices

        if not (return_n_logits == -1 or return_n_logits == 1):
            raise ValueError("return_n_logits must be either -1 or 1")

    def __call__(
        self,
        tokens: TensorValueLike,
        signal_buffers: list[BufferValue],
        kv_cache_inputs_per_dev: list[tuple[TensorValue, ...]],
        **kwargs,
    ) -> tuple[TensorValue, ...]:
        h = self.embed_tokens(tokens, signal_buffers)

        kv_collections = [
            self.kv_collection_constructor(*kv_cache_inputs)
            for kv_cache_inputs in kv_cache_inputs_per_dev
        ]

        input_row_offsets = kwargs["input_row_offsets"]
        root_cache_lengths = kv_cache_inputs_per_dev[0][1]
        valid_lengths: TensorValue = ops.rebind(
            input_row_offsets[1:] - input_row_offsets[:-1],
            root_cache_lengths.shape,
        )
        context_lengths = valid_lengths + root_cache_lengths
        context_lengths = context_lengths.cast(DType.int32)
        for _, layer in enumerate(self.layers):
            h = layer(
                h,
                signal_buffers,
                kv_collections,
                context_lengths=context_lengths,
                **kwargs,
            )

        if self.return_n_logits == -1:
            logits = ops.cast(self.lm_head(self.norm(h))[0], DType.float32)
            last_token_indices = input_row_offsets[1:] - 1
            last_token_logits = ops.gather(logits, last_token_indices, axis=0)
            return (last_token_logits, logits)
        else:
            h0 = h[0]  # All the outputs are the same here.
            last_token_indices = input_row_offsets[1:] - 1
            last_token = ops.gather(h0, last_token_indices, axis=0)

            # Always return float32 logits, no matter the activation type
            last_token_distributed = distribute_value(last_token, self.devices)
            return (
                ops.cast(
                    self.lm_head(self.norm(last_token_distributed))[0],
                    DType.float32,
                ),
            )
