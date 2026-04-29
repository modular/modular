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

"""Implements the MiniMax-M2 model.

MiniMax-M2 is a Mixture-of-Experts decoder-only transformer with:
- GQA attention with QK norm and partial RoPE
- Sigmoid MoE routing with expert score correction bias
- Gated MLP (SwiGLU) experts
"""

from __future__ import annotations

import functools
from collections.abc import Callable, Sequence
from typing import Any

from max.dtype import DType
from max.graph import (
    BufferType,
    BufferValue,
    DeviceRef,
    ShardingStrategy,
    TensorType,
    TensorValue,
    TensorValueLike,
    Value,
    ops,
)
from max.graph.quantization import QuantizationEncoding
from max.nn.comm import Signals
from max.nn.comm.ep import EPBatchManager
from max.nn.data_parallelism import split_batch_replicated
from max.nn.embedding import VocabParallelEmbedding
from max.nn.kv_cache import KVCacheParamInterface, PagedCacheValues
from max.nn.layer import LayerList, Module
from max.nn.linear import ColumnParallelLinear, Linear
from max.nn.moe import MoE, MoEQuantized, forward_moe_sharded_layers
from max.nn.norm import RMSNorm
from max.nn.rotary_embedding import RotaryEmbedding
from max.nn.transformer import forward_sequential_layers
from max.nn.transformer.distributed_transformer import (
    DistributedLogitsPostprocessMixin,
    forward_sharded_layers,
)
from max.pipelines.architectures.gemma4.layers.rotary_embedding import (
    ProportionalScalingParams,
)

from .layers.attention import MiniMaxM2Attention
from .layers.moe_gate import MiniMaxM2TopKRouter
from .layers.rotary_embedding import MiniMaxM2RotaryEmbedding
from .model_config import MiniMaxM2Config


def _unpack_kv_collections(
    kv_collections: Sequence[PagedCacheValues],
) -> tuple[
    list[BufferValue],
    list[TensorValue],
    list[TensorValue],
    list[TensorValue],
    list[TensorValue],
]:
    """Unpack KV collections into component lists for subgraph compatibility.

    Subgraphs require all inputs to be flat Value objects. PagedCacheValues
    is a Python dataclass that cannot be passed directly.

    Returns:
        Tuple of (kv_blocks, cache_lengths, lookup_tables, max_lengths,
        dispatch_metadata_tensors).
    """
    dispatch_metadata_tensors = [
        kv.attention_dispatch_metadata
        for kv in kv_collections
        if kv.attention_dispatch_metadata is not None
    ]
    return (
        [kv.kv_blocks for kv in kv_collections],
        [kv.cache_lengths for kv in kv_collections],
        [kv.lookup_table for kv in kv_collections],
        [kv.max_lengths for kv in kv_collections],
        dispatch_metadata_tensors,
    )


class MiniMaxM2TransformerBlock(Module):
    """MiniMax-M2 transformer block with GQA attention and MoE feed-forward."""

    def __init__(
        self,
        config: MiniMaxM2Config,
        layer_idx: int,
        rope: RotaryEmbedding,
        create_norm: Callable[..., RMSNorm],
        linear_cls: Callable[..., Linear],
        ep_manager: EPBatchManager | None = None,
    ) -> None:
        super().__init__()
        assert config.quant_config is not None, (
            "MiniMax-M2 requires quantized weights (FP8 or NVFP4)"
        )
        self.devices = config.devices
        num_devices = len(config.devices)

        attn_dtype = config.attn_dtype or config.dtype
        attn_is_quantized = (
            attn_dtype == config.dtype and config.quant_config is not None
        )
        attn_quant_config = config.quant_config if attn_is_quantized else None
        attn_linear_cls = linear_cls if attn_is_quantized else Linear

        # Attention layer with per-layer QK norm
        self.self_attn = MiniMaxM2Attention(
            num_attention_heads=config.num_attention_heads,
            num_key_value_heads=config.num_key_value_heads,
            hidden_size=config.hidden_size,
            kv_params=config.kv_params,
            layer_idx=layer_idx,
            dtype=attn_dtype,
            rope=rope,
            linear_cls=attn_linear_cls,
            devices=config.devices,
            scale=config.attention_multiplier,
            norm_dtype=config.norm_dtype or config.dtype,
            quant_config=attn_quant_config,
        )
        self.self_attn.sharding_strategy = ShardingStrategy.replicate(
            num_devices
        )
        self.self_attn_shards = self.self_attn.shard(config.devices)

        # MoE layer (all layers are MoE in MiniMax-M2)
        self.ep_manager = ep_manager
        self.mlp = self._get_mlp(config, linear_cls)
        self.mlp.sharding_strategy = ShardingStrategy.expert_parallel(
            num_devices
        )
        self.mlp_shards = self.mlp.shard(config.devices)

        # Layer norms (replicated)
        self.input_layernorm = create_norm()
        self.input_layernorm.sharding_strategy = ShardingStrategy.replicate(
            num_devices
        )
        self.input_layernorm_shards = self.input_layernorm.shard(config.devices)

        self.post_attention_layernorm = create_norm()
        self.post_attention_layernorm.sharding_strategy = (
            ShardingStrategy.replicate(num_devices)
        )
        self.post_attention_layernorm_shards = (
            self.post_attention_layernorm.shard(config.devices)
        )

    def _get_mlp(
        self,
        config: MiniMaxM2Config,
        linear_cls: Callable[..., Linear],
    ) -> MoE:
        """Create the MoE layer with sigmoid routing."""
        moe_cls = MoEQuantized if config.quant_config is not None else MoE
        return moe_cls(
            devices=config.devices,
            hidden_dim=config.hidden_size,
            num_experts=config.num_local_experts,
            num_experts_per_token=config.num_experts_per_tok,
            moe_dim=config.intermediate_size,
            gate_cls=functools.partial(
                MiniMaxM2TopKRouter,
                norm_topk_prob=config.norm_topk_prob,
                gate_dtype=config.gate_dtype or config.dtype,
                correction_bias_dtype=(
                    config.correction_bias_dtype or DType.float32
                ),
            ),
            dtype=config.dtype,
            quant_config=config.quant_config,
            ep_batch_manager=self.ep_manager,
            ep_size=(
                self.ep_manager.config.n_gpus_per_node
                * self.ep_manager.config.n_nodes
                if self.ep_manager is not None
                else 1
            ),
        )

    def __call__(
        self,
        layer_idx: TensorValue,
        xs: list[TensorValue],
        signal_buffers: list[BufferValue],
        kv_blocks: list[BufferValue],
        kv_cache_lengths: list[TensorValue],
        kv_lookup_table: list[TensorValue],
        kv_max_lengths: list[TensorValue],
        dispatch_metadata_tensors: list[TensorValue],
        freqs_cis: list[TensorValue],
        input_row_offsets: list[TensorValue],
        ep_inputs: list[TensorValue | BufferValue] | None = None,
    ) -> list[TensorValue]:
        """Forward pass through the block."""
        # Re-pack flat KV args into PagedCacheValues for attention.
        # Subgraphs require flat Value inputs, so the caller unpacks and we
        # reconstruct here.
        num_devices = len(kv_blocks)
        kv_collections = [
            PagedCacheValues(
                kv_blocks[i],
                kv_cache_lengths[i],
                kv_lookup_table[i],
                kv_max_lengths[i],
                attention_dispatch_metadata=dispatch_metadata_tensors[i]
                if dispatch_metadata_tensors
                else None,
            )
            for i in range(num_devices)
        ]

        # Input layer norm
        norm_xs = forward_sharded_layers(self.input_layernorm_shards, xs)

        # Self-attention (replicated — no allreduce needed)
        attn_outs = [
            shard(
                layer_idx,
                norm_xs[i],
                kv_collections[i],
                freqs_cis[i],
                input_row_offsets[i],
            )
            for i, shard in enumerate(self.self_attn_shards)
        ]

        # Residual connection
        hs = [x + attn_out for x, attn_out in zip(xs, attn_outs, strict=True)]

        # Post-attention layer norm
        norm_outs = forward_sharded_layers(
            self.post_attention_layernorm_shards, hs
        )

        # Fetch EP buffers before MoE (if using EP)
        if self.ep_manager is not None and ep_inputs is not None:
            self.ep_manager.fetch_buffers(ep_inputs)

        # MoE (EP dispatch/combine handles routing — no allreduce needed)
        mlp_outs = forward_moe_sharded_layers(self.mlp_shards, norm_outs)

        # Residual connection
        hs = [h + mlp_out for h, mlp_out in zip(hs, mlp_outs, strict=True)]

        return hs


class MiniMaxM2(DistributedLogitsPostprocessMixin, Module):
    """MiniMax-M2 model supporting single-GPU and multi-GPU TP inference."""

    def __init__(self, config: MiniMaxM2Config) -> None:
        super().__init__()
        self.config = config
        self.devices = config.devices
        self.num_devices = len(config.devices)
        self.ep_manager: EPBatchManager | None = None
        if config.ep_config is not None:
            self.ep_manager = EPBatchManager(config.ep_config)

        if config.model_quantization_encoding == QuantizationEncoding.GPTQ:
            raise NotImplementedError("GPTQ MiniMax-M2 is not implemented yet")
        if (
            config.model_quantization_encoding is not None
            and config.model_quantization_encoding != QuantizationEncoding.GPTQ
        ):
            raise NotImplementedError("GGUFQ MiniMax-M2 is not implemented yet")

        # RoPE with proportional scaling for partial rotation
        # MiniMax-M2 has rotary_dim=64, head_dim=128 -> partial_rotary_factor=0.5
        # Uses MiniMaxM2RotaryEmbedding which zeros out non-rotated dims
        rope = MiniMaxM2RotaryEmbedding(
            dim=config.hidden_size,
            n_heads=config.num_attention_heads,
            theta=config.rope_theta,
            max_seq_len=config.max_seq_len,
            head_dim=config.kv_params.head_dim,
            interleaved=False,
            scaling_params=ProportionalScalingParams(
                partial_rotary_factor=config.partial_rotary_factor,
            ),
        )
        self.rope = rope

        # Norm factory
        create_norm = functools.partial(
            RMSNorm,
            config.hidden_size,
            dtype=config.norm_dtype or DType.float32,
            eps=config.rms_norm_eps or 1e-6,
            multiply_before_cast=False,
        )

        linear_cls = functools.partial(Linear, quant_config=config.quant_config)

        # Transformer layers
        self.layers = LayerList(
            [
                MiniMaxM2TransformerBlock(
                    config=config,
                    layer_idx=i,
                    rope=rope,
                    create_norm=create_norm,
                    linear_cls=linear_cls,
                    ep_manager=self.ep_manager,
                )
                for i in range(config.num_hidden_layers)
            ]
        )

        if config.use_subgraphs:
            self.subgraph_layer_groups: list[list[int]] = [
                list(range(config.num_hidden_layers))
            ]
        else:
            self.subgraph_layer_groups = []

        # Final norm (replicated)
        self.norm = create_norm()
        self.norm.sharding_strategy = ShardingStrategy.replicate(
            self.num_devices
        )
        self.norm_shards = self.norm.shard(config.devices)

        # Embedding and output layers
        embedding_dtype = config.dtype
        if config.quant_config and config.quant_config.embedding_output_dtype:
            embedding_dtype = config.quant_config.embedding_output_dtype

        self.embed_tokens = VocabParallelEmbedding(
            config.vocab_size,
            config.hidden_size,
            embedding_dtype,
            config.devices,
        )
        self.lm_head = ColumnParallelLinear(
            config.hidden_size,
            config.vocab_size,
            embedding_dtype,
            devices=config.devices,
            tied_weight=(
                self.embed_tokens.weight if config.tie_word_embeddings else None
            ),
        )

        self.kv_params = config.kv_params
        self.return_logits = config.return_logits

    def __call__(
        self,
        tokens: TensorValueLike,
        kv_collections: list[PagedCacheValues],
        return_n_logits: TensorValue,
        input_row_offsets: TensorValue,
        signal_buffers: list[BufferValue],
        ep_inputs: list[TensorValue | BufferValue] | None = None,
        data_parallel_splits: TensorValue | None = None,
        host_input_row_offsets: TensorValue | None = None,
    ) -> tuple[TensorValue, ...]:
        """Forward pass through the model.

        Args:
            tokens: Input token IDs.
            kv_collections: KV cache per device.
            return_n_logits: Number of logits to return.
            input_row_offsets: Row offsets for ragged batching.
            signal_buffers: Signal buffers for allreduce.
            ep_inputs: Expert parallelism communication buffers.
            data_parallel_splits: Per-rank token counts for DP splitting.
            host_input_row_offsets: Host-side row offsets for DP splitting.
        """
        # Embeddings (allreduce produces identical states on all GPUs)
        h = self.embed_tokens(tokens, signal_buffers)

        # Distribute RoPE frequencies
        freqs_cis = [self.rope.freqs_cis.to(device) for device in self.devices]

        # Broadcast input_row_offsets to all devices
        input_row_offsets_list = ops.distributed_broadcast(
            input_row_offsets.to(self.devices[0]), signal_buffers
        )

        # Split replicated batch across GPUs for DP
        assert data_parallel_splits is not None
        assert host_input_row_offsets is not None
        h, input_row_offsets_list = split_batch_replicated(
            self.devices,
            h,
            input_row_offsets_list,
            host_input_row_offsets.cast(DType.int64),
            data_parallel_splits,
        )

        # Unpack KV collections for subgraph compatibility
        (
            kv_blocks,
            cache_lengths,
            lookup_tables,
            max_lengths,
            dispatch_metadata_tensors,
        ) = _unpack_kv_collections(kv_collections)

        def inputs_for_layer(
            idx: int, h: list[TensorValue]
        ) -> list[Value[Any] | Sequence[Value[Any]]]:
            values: list[Value[Any] | Sequence[Value[Any]]] = [
                ops.constant(idx, DType.uint32, device=DeviceRef.CPU()),
                h,
                signal_buffers,
                kv_blocks,
                cache_lengths,
                lookup_tables,
                max_lengths,
                dispatch_metadata_tensors,
                freqs_cis,
                input_row_offsets_list,
            ]
            if ep_inputs is not None:
                values.append(ep_inputs)
            return values

        h = forward_sequential_layers(
            list(self.layers),
            inputs_for_layer=inputs_for_layer,
            weight_prefix_for_layer=lambda i: f"layers.{i}.",
            subgraph_layer_groups=self.subgraph_layer_groups or None,
            initial_hidden_states=h,
        )

        # DP logit postprocessing: allgather last tokens then lm_head
        last_token_per_dev = [
            ops.gather(h[i], input_row_offsets_list[i][1:] - 1, axis=0)
            for i in range(len(self.devices))
        ]
        last_token_distributed = ops.allgather(
            last_token_per_dev, signal_buffers
        )
        norm_last_token = forward_sharded_layers(
            self.norm_shards, last_token_distributed
        )
        last_logits = ops.cast(
            self.lm_head(norm_last_token, signal_buffers)[0],
            DType.float32,
        )
        return (last_logits,)

    def input_types(
        self, kv_params: KVCacheParamInterface
    ) -> tuple[TensorType | BufferType, ...]:
        """Get input types for graph construction."""
        device_ref = self.devices[0]

        tokens_type = TensorType(
            DType.int64, shape=["total_seq_len"], device=device_ref
        )
        input_row_offsets_type = TensorType(
            DType.uint32, shape=["input_row_offsets_len"], device=device_ref
        )
        return_n_logits_type = TensorType(
            DType.int64, shape=["return_n_logits"], device=DeviceRef.CPU()
        )
        data_parallel_splits_type = TensorType(
            DType.int64,
            shape=["data_parallel_splits_len"],
            device=DeviceRef.CPU(),
        )
        host_input_row_offsets_type = TensorType(
            DType.uint32,
            shape=["input_row_offsets_len"],
            device=DeviceRef.CPU(),
        )

        kv_inputs = kv_params.get_symbolic_inputs()

        base_inputs: list[TensorType | BufferType] = [
            tokens_type,
            input_row_offsets_type,
            return_n_logits_type,
            data_parallel_splits_type,
            host_input_row_offsets_type,
        ]

        signals = Signals(devices=self.devices)
        signal_buffer_types = signals.input_types()
        flattened_kv_types = kv_inputs.flatten()

        ep_input_types: list[TensorType | BufferType] = []
        if self.ep_manager is not None:
            ep_input_types = list(self.ep_manager.input_types())

        return tuple(
            base_inputs
            + signal_buffer_types
            + flattened_kv_types
            + ep_input_types
        )
