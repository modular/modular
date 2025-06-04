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

from collections.abc import Iterable
from itertools import islice
from typing import Union, cast

from max.dtype import DType
from max.graph import (
    BufferValue,
    DeviceRef,
    Graph,
    TensorType,
    TensorValue,
    TensorValueLike,
    Type,
    Value,
    ops,
)

from ..embedding import VocabParallelEmbedding
from ..kv_cache import (
    ContinuousBatchingKVCacheCollection,
    FetchContinuousBatchingKVCacheCollection,
    FetchPagedKVCacheCollection,
    KVCacheParams,
    PagedKVCacheCollection,
)
from ..layer import LayerList, Module
from ..linear import ColumnParallelLinear
from ..norm import DistributedRMSNorm
from .transformer import ReturnLogits


def take(it: Iterable[Value], n: int) -> list[Value]:
    """Return the next *n* items from *it* as a list."""
    return list(islice(it, n))


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
        layer_idx: TensorValue,
        xs: list[TensorValue],
        signal_buffers: list[BufferValue],
        kv_collections: list[
            ContinuousBatchingKVCacheCollection | PagedKVCacheCollection
        ],
        input_row_offsets: list[TensorValue],
    ) -> list[TensorValue]:
        attn_outs = self.self_attn(
            layer_idx,
            self.input_layernorm(xs),
            signal_buffers,
            kv_collections,
            input_row_offsets,
        )

        hs = [x + attn_out for x, attn_out in zip(xs, attn_outs)]
        mlp_outs = self.mlp(self.post_attention_layernorm(hs), signal_buffers)
        hs = [h + mlp_out for h, mlp_out in zip(hs, mlp_outs)]

        return hs


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
        ),
        devices: list[DeviceRef],
        return_logits: ReturnLogits = ReturnLogits.LAST_TOKEN,
        use_subgraphs: bool = False,
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
        self.return_logits = return_logits
        self.devices = devices
        self.use_subgraphs = use_subgraphs
        if self.return_logits == ReturnLogits.VARIABLE:
            raise ValueError(
                "DistributedTransformer does not support variable logits."
            )

    def __call__(
        self,
        tokens: TensorValueLike,
        signal_buffers: list[BufferValue],
        kv_cache_inputs_per_dev: list[tuple[TensorValue, ...]],
        return_n_logits: TensorValue,
        input_row_offsets: TensorValue,
    ) -> tuple[TensorValue, ...]:
        h = self.embed_tokens(tokens, signal_buffers)

        kv_collections = [
            self.kv_collection_constructor(*kv_cache_inputs)
            for kv_cache_inputs in kv_cache_inputs_per_dev
        ]

        input_row_offsets_ = distribute_value(input_row_offsets, self.devices)

        if self.use_subgraphs:
            subgraph_input_types: list[Type] = [
                TensorType(DType.uint32, shape=(), device=DeviceRef.CPU()),
                *[hidden.type for hidden in h],
                *[signal_buffer.type for signal_buffer in signal_buffers],
                *[kv_collection.type for kv_collection in kv_collections],
                *[offset.type for offset in input_row_offsets_],
            ]
            num_devices = len(self.devices)
            subgraph_layer = self.layers[0]
            assert isinstance(subgraph_layer, DistributedTransformerBlock)
            layer_weights = list(subgraph_layer.raw_state_dict().values())
            subgraph_weight_prefix = "layers.0."

            with Graph.current.add_subgraph(
                "dist_transformer_block",
                input_types=subgraph_input_types,
            ) as subgraph:
                inputs = iter(subgraph.inputs)
                arg_layer_idx = next(inputs)
                arg_xs = [x.tensor for x in take(inputs, num_devices)]
                arg_signal_buffers = [
                    x.buffer for x in take(inputs, num_devices)
                ]
                arg_kv_collections = cast(
                    list[
                        Union[
                            ContinuousBatchingKVCacheCollection,
                            PagedKVCacheCollection,
                        ]
                    ],
                    take(inputs, num_devices),
                )
                arg_input_row_offsets = [
                    x.tensor for x in take(inputs, num_devices)
                ]

                for weight in filter(
                    lambda w: w.name.startswith(subgraph_weight_prefix),
                    layer_weights,
                ):
                    weight._placeholder = True
                    weight.name = weight.name.removeprefix(
                        subgraph_weight_prefix
                    )

                # prevent re-entry
                results = subgraph_layer(
                    arg_layer_idx,
                    arg_xs,
                    arg_signal_buffers,
                    arg_kv_collections,
                    arg_input_row_offsets,
                )

                subgraph.output(*results)

            for idx in range(len(self.layers)):
                h = [
                    x.tensor
                    for x in ops.call(
                        subgraph,
                        ops.constant(idx, DType.uint32, device=DeviceRef.CPU()),
                        *h,
                        *signal_buffers,
                        *kv_collections,
                        *input_row_offsets_,
                        prefix=f"layers.{idx}.",
                    )
                ]
        else:
            for idx, layer in enumerate(self.layers):
                h = layer(
                    ops.constant(idx, DType.uint32, device=DeviceRef.CPU()),
                    h,
                    signal_buffers,
                    kv_collections,
                    input_row_offsets_,
                )
        h0 = h[0]
        last_token_indices = input_row_offsets[1:] - 1
        last_token_h = ops.gather(h0, last_token_indices, axis=0)
        last_token_distributed = distribute_value(last_token_h, self.devices)
        last_logits = ops.cast(
            self.lm_head(self.norm(last_token_distributed))[0], DType.float32
        )

        logits = None
        offsets = None

        if self.return_logits == ReturnLogits.VARIABLE:
            return_n_logits_range = ops.range(
                return_n_logits[0],
                ops.constant(0, DType.int64, device=DeviceRef.CPU()),
                ops.constant(-1, DType.int64, device=DeviceRef.CPU()),
                out_dim="return_n_logits_range",
                device=self.devices[0],
            )
            offsets = (
                ops.unsqueeze(input_row_offsets[1:], -1) - return_n_logits_range
            )
            last_indices = ops.reshape(offsets, shape=(-1,))
            logits = ops.gather(
                ops.cast(self.lm_head(self.norm(h))[0], DType.float32),
                last_indices,
                axis=0,
            )
            offsets = ops.range(
                ops.constant(0, DType.int64, device=DeviceRef.CPU()),
                last_indices.shape[0] + return_n_logits[0],
                return_n_logits[0],
                out_dim="logit_offsets",
                device=self.devices[0],
            )
        elif self.return_logits == ReturnLogits.ALL:
            logits = ops.cast(self.lm_head(self.norm(h))[0], DType.float32)
            offsets = input_row_offsets

        if logits is not None and offsets is not None:
            return (last_logits, logits, offsets)
        else:
            return (last_logits,)
