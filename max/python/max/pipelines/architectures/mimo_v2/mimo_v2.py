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
"""The MiMo-V2.6-Flash text model."""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

from max.dtype import DType
from max.graph import (
    BufferValue,
    DeviceRef,
    Graph,
    ShardingStrategy,
    TensorValue,
    Type,
    Value,
    ops,
)
from max.graph import Module as GraphModule
from max.nn.comm.allreduce import Allreduce
from max.nn.embedding import Embedding
from max.nn.kv_cache import KVCacheParams, MultiKVCacheParams, PagedCacheValues
from max.nn.layer import LayerList, Module
from max.nn.linear import MLP, ColumnParallelLinear
from max.nn.moe import MoEQuantized
from max.nn.norm.rms_norm import RMSNorm
from max.nn.rotary_embedding import RotaryEmbedding
from max.nn.transformer import (
    DecoderOutputs,
    ReturnHiddenStates,
    forward_sequential_layers,
)
from max.nn.transformer.distributed_transformer import (
    DistributedLogitsPostprocessMixin,
    forward_sharded_layers,
)
from max.pipelines.lib.interfaces.batch_processor import (
    ragged_batch_inputs,
    signal_inputs,
)
from max.tree import Tree

from .layers.attention import MiMoV2Attention
from .layers.moe import mimo_v2_moe
from .model_config import FULL, SLIDING, MiMoV2Config


class MiMoV2DecoderLayer(Module):
    """Pre-norm attention and MLP (dense or MoE), tensor parallel."""

    def __init__(
        self,
        attention: MiMoV2Attention,
        mlp: MLP | MoEQuantized,
        input_layernorm: RMSNorm,
        post_attention_layernorm: RMSNorm,
        devices: list[DeviceRef],
    ) -> None:
        super().__init__()
        n = len(devices)
        self.self_attn = attention
        self.self_attn.sharding_strategy = ShardingStrategy.tensor_parallel(n)
        self.self_attn_shards = attention.shard(devices)
        self.mlp = mlp
        self.mlp.sharding_strategy = ShardingStrategy.tensor_parallel(n)
        self.mlp_shards = mlp.shard(devices)
        self.input_layernorm = input_layernorm
        self.input_layernorm.sharding_strategy = ShardingStrategy.replicate(n)
        self.input_layernorm_shards = input_layernorm.shard(devices)
        self.post_attention_layernorm = post_attention_layernorm
        self.post_attention_layernorm.sharding_strategy = (
            ShardingStrategy.replicate(n)
        )
        self.post_attention_layernorm_shards = post_attention_layernorm.shard(
            devices
        )
        self.allreduce = Allreduce(num_accelerators=n)

    def __call__(
        self,
        layer_idx: TensorValue,
        xs: list[TensorValue],
        signal_buffers: Sequence[BufferValue],
        kv_collections: Sequence[PagedCacheValues],
        freqs_cis: Sequence[TensorValue],
        input_row_offsets: Sequence[TensorValue],
    ) -> list[TensorValue]:
        norm_xs = forward_sharded_layers(self.input_layernorm_shards, xs)
        attn_out = [
            shard(
                layer_idx,
                norm_xs[i],
                kv_collections[i],
                freqs_cis[i],
                input_row_offsets[i],
            )
            for i, shard in enumerate(self.self_attn_shards)
        ]
        attn_out = self.allreduce(attn_out, signal_buffers)
        hs = [x + a for x, a in zip(xs, attn_out, strict=True)]

        norm_hs = forward_sharded_layers(
            self.post_attention_layernorm_shards, hs
        )
        mlp_out = forward_sharded_layers(self.mlp_shards, norm_hs)
        mlp_out = self.allreduce(mlp_out, signal_buffers)
        return [h + m for h, m in zip(hs, mlp_out, strict=True)]


class MiMoV2(DistributedLogitsPostprocessMixin, Module):
    """MiMo-V2.6-Flash: 48 hybrid SWA/full decoder layers and an untied head.

    Weight names are the checkpoint's without ``model.``, as the weight
    adapter emits them.

    With ``use_subgraphs``, the MoE layers of each attention type share one
    compiled subgraph; layer 0, the only dense one, runs inline.
    """

    subgraph_layer_prefix: str = "layers"
    """The weight name prefix of :attr:`layers`, which each subgraph call
    binds its layer's weights by; a model that nests this one prefixes it."""

    def __init__(self, config: MiMoV2Config) -> None:
        super().__init__()
        self.devices = config.devices
        self.return_logits = config.return_logits
        self.return_hidden_states = config.return_hidden_states
        self.target_layer_ids = config.target_layer_ids

        assert isinstance(config.kv_params, MultiKVCacheParams)
        self.kv_params = config.kv_params
        kv_params: dict[str, KVCacheParams] = {}
        for group, params in config.kv_params.children.items():
            assert isinstance(params, KVCacheParams)
            kv_params[group] = params
        self.ropes = {
            group: RotaryEmbedding(
                dim=config.hidden_size,
                n_heads=config.num_attention_heads,
                theta=theta,
                max_seq_len=config.max_seq_len,
                head_dim=config.rotary_dim,
                interleaved=False,
            )
            for group, theta in config.rope_thetas.items()
        }

        self.embed_tokens = Embedding(
            config.vocab_size,
            config.hidden_size,
            dtype=config.dtype,
            device=config.devices[0],
        )
        self.norm = self._rms_norm(config)
        self.norm.sharding_strategy = ShardingStrategy.replicate(
            len(config.devices)
        )
        self.norm_shards = self.norm.shard(config.devices)
        self.lm_head = ColumnParallelLinear(
            config.hidden_size,
            config.vocab_size,
            dtype=config.dtype,
            devices=config.devices,
        )

        group_layer_counts = {SLIDING: 0, FULL: 0}
        # Each layer's index within its KV group.
        self.kv_layer_indices: list[int] = []
        layers = []
        for i, group in enumerate(config.layer_types):
            attention = MiMoV2Attention(
                layout=config.qkv_layouts[group],
                head_dim=config.head_dim,
                v_head_dim=config.v_head_dim,
                hidden_size=config.hidden_size,
                kv_params=kv_params[group],
                scale=config.attention_scale,
                value_scale=config.attention_value_scale,
                sliding_window=(
                    config.sliding_window if group == SLIDING else None
                ),
                has_sinks=config.sinks[group],
                quant_config=config.quant.dense,
                devices=config.devices,
                fused_sliding_qkv=config.fused_sliding_qkv,
                fused_full_qkv=config.fused_full_qkv,
            )
            self.kv_layer_indices.append(group_layer_counts[group])
            group_layer_counts[group] += 1
            mlp: MLP | MoEQuantized
            if i in config.moe_layers:
                mlp = mimo_v2_moe(
                    hidden_dim=config.hidden_size,
                    num_experts=config.n_routed_experts,
                    num_experts_per_tok=config.num_experts_per_tok,
                    moe_dim=config.moe_intermediate_size,
                    norm_topk_prob=config.norm_topk_prob,
                    quant_config=config.quant.experts,
                    devices=config.devices,
                )
            else:
                mlp = MLP(
                    dtype=DType.float8_e4m3fn,
                    quantization_encoding=None,
                    hidden_dim=config.hidden_size,
                    feed_forward_length=config.intermediate_size,
                    devices=config.devices,
                    quant_config=config.quant.dense,
                )
            layers.append(
                MiMoV2DecoderLayer(
                    attention=attention,
                    mlp=mlp,
                    input_layernorm=self._rms_norm(config),
                    post_attention_layernorm=self._rms_norm(config),
                    devices=config.devices,
                )
            )
        self.layers = LayerList(layers)
        self._layer_groups = list(config.layer_types)
        subgraph_groups: dict[str, list[int]] = {}
        if config.use_subgraphs:
            for i in sorted(config.moe_layers):
                subgraph_groups.setdefault(config.layer_types[i], []).append(i)
        self.subgraph_layer_groups = list(subgraph_groups.values())

    def input_tree(self) -> OrderedDict[str, object]:
        """Names the graph's inputs; its leaves are its inputs.

        The order is GPT OSS's, so its batching feeds this graph.

        Returns:
            The input types under ``batch`` (whose ``row_offsets`` are one
            per device), ``signals`` and ``kv``, in graph order.
        """
        devices = list(self.devices)
        return OrderedDict(
            batch=ragged_batch_inputs(
                devices[0],
                ("tokens", "return_n_logits", "row_offsets"),
                row_offsets_devices=devices,
            ),
            signals=signal_inputs(devices),
            kv=self.kv_params.get_symbolic_inputs(),
        )

    def trace(
        self,
        *input_types: Type[Any],
        name: str | None = None,
        module: GraphModule | None = None,
        custom_extensions: Iterable[Path] = (),
        is_device_graph: bool = False,
    ) -> Graph:
        """Traces the graph, whose inputs are :meth:`input_tree`'s leaves.

        Adapts the flat graph inputs to :meth:`__call__`, splitting the KV
        cache into its sliding and full attention collections.

        Args:
            *input_types: The graph's input types, the leaves of
                :meth:`input_tree`.
            name: The name of the graph. Defaults to the class's qualified
                name.
            module: The MLIR module to add the graph to. Defaults to a new
                module.
            custom_extensions: Paths to custom kernel libraries the graph
                uses.
            is_device_graph: Whether to build a device graph.

        Returns:
            The traced :class:`~max.graph.Graph`.
        """
        n = len(self.devices)

        def forward(
            tokens: Value[Any],
            return_n_logits: Value[Any],
            *rest: Value[Any],
        ) -> DecoderOutputs:
            sliding_kv, full_kv = self.kv_params.unflatten_basic_kv_tree(
                iter(rest[2 * n :])
            )
            return self(
                tokens=tokens.tensor,
                signal_buffers=[v.buffer for v in rest[n : 2 * n]],
                sliding_kv_collections=sliding_kv,
                full_kv_collections=full_kv,
                return_n_logits=return_n_logits.tensor,
                input_row_offsets=[v.tensor for v in rest[:n]],
            )

        return Graph(
            type(self).__qualname__ if name is None else name,
            forward,
            input_types=input_types,
            custom_extensions=list(custom_extensions),
            module=module,
            is_device_graph=is_device_graph,
        )

    @staticmethod
    def _rms_norm(config: MiMoV2Config) -> RMSNorm:
        # The reference casts to BF16 before it scales; scaling in float32 and
        # casting once is the order of the kernel that fuses a norm with the
        # allreduce and residual add in front of it.
        return RMSNorm(
            config.hidden_size,
            config.dtype,
            eps=config.rms_norm_eps,
            multiply_before_cast=True,
        )

    def __call__(
        self,
        tokens: TensorValue,
        signal_buffers: Sequence[BufferValue],
        sliding_kv_collections: Sequence[PagedCacheValues],
        full_kv_collections: Sequence[PagedCacheValues],
        return_n_logits: TensorValue,
        input_row_offsets: Sequence[TensorValue],
    ) -> DecoderOutputs:
        h_embed = self.embed_tokens(tokens)
        # A broadcast over the signal buffers, not a GPU-to-GPU transfer: the
        # transfer makes one device's stream wait on another's, which device
        # graph capture cannot record.
        h = (
            ops.distributed_broadcast(h_embed, signal_buffers)
            if len(self.devices) > 1
            else [h_embed]
        )
        kv_collections = {
            SLIDING: list(sliding_kv_collections),
            FULL: list(full_kv_collections),
        }
        freqs_cis = {
            group: [
                ops.cast(rope.freqs_cis, h_embed.dtype).to(device)
                for device in self.devices
            ]
            for group, rope in self.ropes.items()
        }

        targets = set(self.target_layer_ids or ())
        return_taps = (
            self.return_hidden_states == ReturnHiddenStates.SELECTED_LAYERS
        )
        capture: list[list[TensorValue]] | None = (
            [] if targets and return_taps else None
        )

        def layer_inputs(i: int, h: list[TensorValue]) -> list[Tree[Any]]:
            group = self._layer_groups[i]
            return [
                ops.constant(
                    self.kv_layer_indices[i],
                    DType.uint32,
                    device=DeviceRef.CPU(),
                ),
                h,
                list(signal_buffers),
                kv_collections[group],
                freqs_cis[group],
                list(input_row_offsets),
            ]

        def capture_layer(i: int, h: list[TensorValue]) -> None:
            if capture is not None and i in targets:
                capture.append(list(h))

        h = forward_sequential_layers(
            list(self.layers),
            inputs_for_layer=layer_inputs,
            initial_hidden_states=h,
            on_layer_output=capture_layer,
            subgraph_layer_groups=self.subgraph_layer_groups or None,
            name_for_subgraph=lambda g: f"mimo_v2_layer_group_{g}",
            weight_prefix_for_layer=lambda i: (
                f"{self.subgraph_layer_prefix}.{i}."
            ),
        )

        return self._postprocess_logits(
            h,
            input_row_offsets,
            return_n_logits,
            signal_buffers,
            capture_hidden_states=capture if return_taps else None,
        )
