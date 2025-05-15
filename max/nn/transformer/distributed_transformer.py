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
from hashlib import md5
from typing import cast

from max.dtype import DType
from max.graph import (
    BufferValue,
    DeviceRef,
    Graph,
    TensorValue,
    TensorValueLike,
    _ChainType,
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
from ..linear import Linear
from ..norm import DistributedRMSNorm, LayerNorm, RMSNorm
from .transformer import ReturnLogits


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
        use_subgraph: bool = False,
    ):
        super().__init__()
        self.self_attn = attention
        self.mlp = mlp
        self.input_layernorm = attention_norm
        self.post_attention_layernorm = mlp_norm
        self.devices = devices
        self.use_subgraph = use_subgraph

    def build_subgraph(
        self,
        name: str,
    ) -> Module:
        num_devices = len(self.devices)

        raw_state_dict = self.raw_state_dict()
        weights = [
            value
            for _, value in sorted(raw_state_dict.items(), key=lambda x: x[0])
        ]
        weight_mlir_values = [w._mlir_value for w in weights]

        outer_self = self

        class DistributedTransformerBlockSubgraph(Module):
            def __call__(
                self,
                layer_idx: TensorValue,
                xs: list[TensorValue],
                signal_buffers: list[BufferValue],
                kv_collections: list[
                    ContinuousBatchingKVCacheCollection | PagedKVCacheCollection
                ],
                input_row_offsets: TensorValue,
            ) -> list[TensorValue]:
                input_row_offsets_type = input_row_offsets.type
                misc_input_types = [input_row_offsets_type]

                name_suffix = md5(
                    str(tuple(t.to_mlir() for t in misc_input_types)).encode()
                ).hexdigest()
                subgraph = Graph.current._subgraphs.get(f"{name}_{name_suffix}")

                if subgraph is None:
                    layer_idx_type = layer_idx.type
                    x_types = [x.type for x in xs]
                    signal_buffers_types = [
                        signal_buffer.type for signal_buffer in signal_buffers
                    ]
                    kv_collection_types = [
                        kv_collection.type for kv_collection in kv_collections
                    ]
                    graph_inputs = [
                        _ChainType(),
                        layer_idx_type,
                        *x_types,
                        *signal_buffers_types,
                        *kv_collection_types,
                        *misc_input_types,
                    ] + [w.type for w in weights]

                    with Graph.current.add_subgraph(
                        f"{name}_{name_suffix}", input_types=graph_inputs
                    ) as subgraph:
                        subgraph._current_chain._mlir_value = subgraph.inputs[
                            0
                        ]._mlir_value
                        arg_layer_idx = subgraph.inputs[1]
                        arg_xs = subgraph.inputs[2 : 2 + num_devices]
                        arg_signal_buffers = subgraph.inputs[
                            2 + num_devices : 2 + 2 * num_devices
                        ]
                        arg_kv_collections = subgraph.inputs[
                            2 + 2 * num_devices : 2 + 3 * num_devices
                        ]
                        arg_input_row_offsets = subgraph.inputs[
                            2 + 3 * num_devices
                        ]
                        subgraph._mlir_value_map.update(
                            {
                                w: subgraph_input._mlir_value
                                for w, subgraph_input in zip(
                                    weight_mlir_values,
                                    subgraph.inputs[2 + 3 * num_devices + 1 :],
                                )
                            }
                        )

                        # prevent re-entry
                        try:
                            outer_self.use_subgraph = False
                            results = outer_self(
                                arg_layer_idx,
                                arg_xs,
                                arg_signal_buffers,
                                arg_kv_collections,
                                input_row_offsets=arg_input_row_offsets,
                            )
                        finally:
                            outer_self.use_subgraph = True
                        subgraph.output(
                            subgraph._current_chain,
                            *results,
                        )

                call_args = [
                    layer_idx,
                    *xs,
                    *signal_buffers,
                    *kv_collections,
                    input_row_offsets,
                ]
                call_args.extend([w.tensor for w in weights])
                return [res.tensor for res in ops.call(subgraph, *call_args)]

        return DistributedTransformerBlockSubgraph()

    def __call__(
        self,
        layer_idx: TensorValue,
        xs: list[TensorValue],
        signal_buffers: list[BufferValue],
        kv_collections: list[
            ContinuousBatchingKVCacheCollection | PagedKVCacheCollection
        ],
        **kwargs,
    ) -> list[TensorValue]:
        if self.use_subgraph:
            return self.build_subgraph(
                "dist_transformer_block",
            )(layer_idx, xs, signal_buffers, kv_collections, **kwargs)

        attn_outs = self.self_attn(
            layer_idx,
            self.input_layernorm(xs),
            signal_buffers,
            kv_collections,
            **kwargs,
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
        norm: RMSNorm | LayerNorm,
        output: Linear,
        embedding: VocabParallelEmbedding,
        kv_params: KVCacheParams,
        kv_collection_constructor: (
            FetchContinuousBatchingKVCacheCollection
            | FetchPagedKVCacheCollection
        ),
        devices: list[DeviceRef],
        return_logits: ReturnLogits = ReturnLogits.LAST_TOKEN,
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
        **kwargs,
    ) -> tuple[TensorValue, ...]:
        h = self.embed_tokens(tokens, signal_buffers)

        kv_collections = [
            self.kv_collection_constructor(*kv_cache_inputs)
            for kv_cache_inputs in kv_cache_inputs_per_dev
        ]

        input_row_offsets = kwargs["input_row_offsets"]
        root_cache_lengths = kv_cache_inputs_per_dev[0][1]
        for idx, layer in enumerate(self.layers):
            h = layer(
                ops.constant(idx, DType.uint32, device=DeviceRef.CPU()),
                h,
                signal_buffers,
                kv_collections,
                **kwargs,
            )

        h0 = h[0]
        last_token_indices = input_row_offsets[1:] - 1
        last_token_h = ops.gather(h0, last_token_indices, axis=0)
        last_logits = ops.cast(
            self.lm_head(self.norm(last_token_h)), DType.float32
        )

        logits = None
        offsets = None

        # NOTE: ReturnLogits.VARIABLE is unsupported and the ctor checks this.
        if self.return_logits == ReturnLogits.ALL:
            logits = ops.cast(self.lm_head(self.norm(h0)), DType.float32)
            offsets = cast(TensorValue, kwargs["input_row_offsets"])

        if logits is not None and offsets is not None:
            return (last_logits, logits, offsets)
        else:
            return (last_logits,)
