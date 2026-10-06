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

from collections.abc import Callable, Sequence

from max.dtype import DType
from max.graph import (
    BufferValue,
    DeviceRef,
    ShardingStrategy,
    TensorValue,
    ops,
)
from max.nn.comm.allreduce import Allreduce
from max.nn.embedding import Embedding
from max.nn.kv_cache import KVCacheParams, MultiKVCacheParams, PagedCacheValues
from max.nn.layer import LayerList, Module
from max.nn.linear import MLP, ColumnParallelLinear
from max.nn.moe import StackedMoE
from max.nn.norm.rms_norm import RMSNorm
from max.nn.rotary_embedding import RotaryEmbedding
from max.nn.transformer import ReturnHiddenStates
from max.nn.transformer.distributed_transformer import (
    DistributedLogitsPostprocessMixin,
    forward_sharded_layers,
)

from .layers.attention import MiMoV2Attention
from .layers.moe import mimo_v2_moe
from .model_config import FULL, SLIDING, MiMoV2Config

TapHook = Callable[
    [
        list[list[TensorValue]],
        Sequence[PagedCacheValues],
        Sequence[TensorValue],
        Sequence[BufferValue],
    ],
    None,
]
"""Consumes the layer taps inside the graph: ``taps[k][d]`` is the output of
the ``k``-th of ``target_layer_ids`` (in layer order, post-residual and
before the final norm) on device ``d``, then the tail KV group, the input row
offsets and the signal buffers, each per device."""


class MiMoV2DecoderLayer(Module):
    """Pre-norm attention and MLP (dense or MoE), tensor parallel."""

    def __init__(
        self,
        attention: MiMoV2Attention,
        mlp: MLP | StackedMoE,
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
        xs: list[TensorValue],
        signal_buffers: Sequence[BufferValue],
        kv_collections: Sequence[PagedCacheValues],
        input_row_offsets: Sequence[TensorValue],
    ) -> list[TensorValue]:
        norm_xs = forward_sharded_layers(self.input_layernorm_shards, xs)
        attn_out = [
            shard(norm_xs[i], kv_collections[i], input_row_offsets[i])
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
    adapter emits them. With a ``tap_hook``, the graph takes a tail KV group
    after the target's two and hands it, with the outputs of
    ``target_layer_ids``, to the hook, which writes it; a speculative
    drafter's context is written this way on the target's forward.
    """

    def __init__(
        self, config: MiMoV2Config, tap_hook: TapHook | None = None
    ) -> None:
        super().__init__()
        if tap_hook is not None and not config.target_layer_ids:
            raise ValueError("MiMo-V2: a tap hook needs target_layer_ids.")
        self.devices = config.devices
        self.return_logits = config.return_logits
        self.return_hidden_states = config.return_hidden_states
        self.target_layer_ids = config.target_layer_ids
        self.tap_hook = tap_hook

        assert isinstance(config.kv_params, MultiKVCacheParams)
        kv_params: dict[str, KVCacheParams] = {}
        for group, params in config.kv_params.children.items():
            assert isinstance(params, KVCacheParams)
            kv_params[group] = params
        ropes = {
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
        layers = []
        for i, group in enumerate(config.layer_types):
            attention = MiMoV2Attention(
                layout=config.qkv_layouts[group],
                head_dim=config.head_dim,
                v_head_dim=config.v_head_dim,
                hidden_size=config.hidden_size,
                kv_params=kv_params[group],
                layer_idx=group_layer_counts[group],
                rope=ropes[group],
                scale=config.attention_scale,
                value_scale=config.attention_value_scale,
                sliding_window=(
                    config.sliding_window if group == SLIDING else None
                ),
                has_sinks=config.sinks[group],
                quant_config=config.quant.dense,
                devices=config.devices,
            )
            group_layer_counts[group] += 1
            mlp: MLP | StackedMoE
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
        tail_kv_collections: Sequence[PagedCacheValues] | None = None,
    ) -> tuple[TensorValue, ...]:
        if (self.tap_hook is None) != (tail_kv_collections is None):
            raise ValueError(
                "MiMo-V2: a tail KV group and a tap hook come together."
            )
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
            SLIDING: sliding_kv_collections,
            FULL: full_kv_collections,
        }

        targets = set(self.target_layer_ids or ())
        return_taps = (
            self.return_hidden_states == ReturnHiddenStates.SELECTED_LAYERS
        )
        capture: list[list[TensorValue]] | None = (
            []
            if targets and (return_taps or self.tap_hook is not None)
            else None
        )
        for i, layer in enumerate(self.layers):
            h = layer(
                h,
                signal_buffers,
                kv_collections[self._layer_groups[i]],
                input_row_offsets,
            )
            if capture is not None and i in targets:
                capture.append(list(h))

        if self.tap_hook is not None:
            assert tail_kv_collections is not None
            assert capture is not None
            self.tap_hook(
                capture, tail_kv_collections, input_row_offsets, signal_buffers
            )
        return self._postprocess_logits(
            h,
            input_row_offsets,
            return_n_logits,
            signal_buffers,
            capture_hidden_states=capture if return_taps else None,
        )
