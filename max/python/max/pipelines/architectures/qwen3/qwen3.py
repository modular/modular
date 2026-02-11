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
"""Build a Qwen3 model that supports both single and multi-GPU inference."""

from __future__ import annotations

import functools
from collections.abc import Callable
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from max.pipelines.architectures.qwen3vl_moe.nn.moe import Qwen3VLMoE

from max.dtype import DType
from max.graph import (
    BufferType,
    BufferValue,
    DeviceRef,
    ShardingStrategy,
    TensorType,
    TensorValue,
    TensorValueLike,
    ops,
)
from max.graph.quantization import QuantizationEncoding
from max.nn.legacy.comm import Signals
from max.nn.legacy.comm.allreduce import Allreduce
from max.nn.legacy.embedding import VocabParallelEmbedding
from max.nn.legacy.kv_cache import KVCacheParams, PagedCacheValues
from max.nn.legacy.layer import LayerList, Module
from max.nn.legacy.linear import MLP, ColumnParallelLinear, Linear
from max.nn.legacy.norm import RMSNorm
from max.nn.legacy.rotary_embedding import Llama3RotaryEmbedding
from max.nn.legacy.transformer import ReturnLogits
from max.nn.legacy.transformer.distributed_transformer import (
    forward_sharded_layers,
)
from max.pipelines.architectures.qwen3.layers.attention import Qwen3Attention
from max.pipelines.architectures.qwen3.model_config import Qwen3Config


def distribute_value(
    v: TensorValue, devices: list[DeviceRef]
) -> list[TensorValue]:
    """Distribute a tensor to multiple devices."""
    return [v.to(device) for device in devices]


class Qwen3TransformerBlock(Module):
    """Qwen3 transformer block that supports both single and multi-GPU.

    Uses shardable attention and MLP/MoE layers with allreduce for combining
    outputs across devices.
    """

    def __init__(
        self,
        config: Qwen3Config,
        layer_idx: int,
        rope: Llama3RotaryEmbedding,
        create_norm: Callable[..., RMSNorm],
        linear_cls: Callable[..., Linear],
    ) -> None:
        super().__init__()
        self.devices = config.devices
        num_devices = len(config.devices)

        # Create attention layer
        self.self_attn = Qwen3Attention(
            num_attention_heads=config.num_attention_heads,
            num_key_value_heads=config.num_key_value_heads,
            hidden_size=config.hidden_size,
            kv_params=config.kv_params,
            layer_idx=layer_idx,
            dtype=config.dtype,
            rope=rope,
            linear_cls=linear_cls,
            devices=config.devices,
            scale=config.attention_multiplier,
            has_bias=config.attention_bias,
        )
        self.self_attn.sharding_strategy = ShardingStrategy.tensor_parallel(
            num_devices
        )
        self.self_attn_shards = self.self_attn.shard(config.devices)

        # Create MLP or MoE layer
        self.mlp = self._get_mlp(config, layer_idx, linear_cls)
        self.mlp.sharding_strategy = ShardingStrategy.tensor_parallel(
            num_devices
        )
        self.mlp_shards = self.mlp.shard(config.devices)

        # Create norm layers (replicated across devices)
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

        # Allreduce for combining sharded outputs
        self.allreduce = Allreduce(num_accelerators=num_devices)
        self.residual_multiplier = config.residual_multiplier

    def _get_mlp(
        self,
        config: Qwen3Config,
        layer_idx: int,
        linear_cls: Callable[..., Linear],
    ) -> MLP | Qwen3VLMoE:
        """Get MLP or MoE layer based on config and layer index."""
        use_moe = (
            config.num_experts > 0
            and layer_idx not in config.mlp_only_layers
            and (layer_idx + 1) % config.decoder_sparse_step == 0
        )

        if use_moe:
            from max.pipelines.architectures.qwen3vl_moe.nn.moe import (
                Qwen3VLMoE,
                Qwen3VLMoEGate,
            )

            return Qwen3VLMoE(
                devices=config.devices,
                hidden_dim=config.hidden_size,
                num_experts=config.num_experts,
                num_experts_per_token=config.num_experts_per_tok,
                moe_dim=config.moe_intermediate_size,
                gate_cls=Qwen3VLMoEGate,
                dtype=config.dtype,
                float8_config=config.float8_config,
            )
        else:
            return MLP(
                config.dtype,
                config.model_quantization_encoding,
                config.hidden_size,
                config.intermediate_size,
                config.devices,
                linear_cls,
                float8_config=config.float8_config,
            )

    def __call__(
        self,
        layer_idx: TensorValue,
        xs: list[TensorValue],
        kv_collections: list[PagedCacheValues],
        freqs_cis: list[TensorValue],
        input_row_offsets: list[TensorValue],
        signal_buffers: list[BufferValue],
    ) -> list[TensorValue]:
        """Forward pass through the block.

        Args:
            layer_idx: Layer index tensor.
            xs: Per-device hidden states.
            kv_collections: Per-device KV cache.
            freqs_cis: Per-device RoPE frequencies.
            input_row_offsets: Per-device row offsets.
            signal_buffers: Per-device signal buffers for allreduce.

        Returns:
            Per-device updated hidden states.
        """
        # Apply input layer norm
        norm_xs = forward_sharded_layers(self.input_layernorm_shards, xs)

        # Self-attention on each shard
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

        # Allreduce attention outputs (passthrough for single GPU)
        if len(self.devices) > 1:
            attn_outs = self.allreduce(attn_outs, signal_buffers)

        # Residual connection
        hs = [x + attn_out for x, attn_out in zip(xs, attn_outs, strict=True)]
        if self.residual_multiplier != 1.0:
            hs = [h * self.residual_multiplier for h in hs]

        # Apply post-attention layer norm
        norm_outs = forward_sharded_layers(
            self.post_attention_layernorm_shards, hs
        )

        # MLP/MoE on each shard
        mlp_outs = forward_sharded_layers(self.mlp_shards, norm_outs)

        # Allreduce MLP outputs (passthrough for single GPU)
        if len(self.devices) > 1:
            mlp_outs = self.allreduce(mlp_outs, signal_buffers)

        # Residual connection
        hs = [h + mlp_out for h, mlp_out in zip(hs, mlp_outs, strict=True)]

        return hs


class Qwen3(Module):
    """Unified Qwen3 model that supports both single and multi-GPU inference."""

    def __init__(self, config: Qwen3Config) -> None:
        super().__init__()
        self.config = config
        self.devices = config.devices
        self.num_devices = len(config.devices)

        # Validate quantization encoding
        if config.model_quantization_encoding == QuantizationEncoding.GPTQ:
            raise NotImplementedError("GPTQ Qwen3 is not implemented yet")
        if config.model_quantization_encoding is not None:
            raise NotImplementedError("GGUFQ Qwen3 is not implemented yet")

        # Create RoPE embedding
        rope = Llama3RotaryEmbedding(
            dim=config.hidden_size,
            n_heads=config.num_attention_heads,
            theta=config.rope_theta,
            max_seq_len=config.max_seq_len,
            head_dim=config.kv_params.head_dim,
            interleaved=config.interleaved_rope_weights,
            scaling_params=config.rope_scaling_params,
        )
        self.rope = rope

        # Norm factory
        if config.norm_method != "rms_norm" or config.rms_norm_eps is None:
            raise ValueError(
                "Qwen3 requires RMSNorm. Set norm_method='rms_norm' and "
                "provide rms_norm_eps."
            )

        create_norm = functools.partial(
            RMSNorm,
            config.hidden_size,
            dtype=config.norm_dtype or DType.float32,
            eps=config.rms_norm_eps,
            multiply_before_cast=False,
        )

        linear_cls = functools.partial(
            Linear, float8_config=config.float8_config
        )

        # Create transformer layers
        self.layers = LayerList(
            [
                Qwen3TransformerBlock(
                    config=config,
                    layer_idx=i,
                    rope=rope,
                    create_norm=create_norm,
                    linear_cls=linear_cls,
                )
                for i in range(config.num_hidden_layers)
            ]
        )

        # Final norm (replicated)
        self.norm = create_norm()
        self.norm.sharding_strategy = ShardingStrategy.replicate(
            self.num_devices
        )
        self.norm_shards = self.norm.shard(config.devices)

        # Embedding and output layers - always use parallel versions
        # They work correctly for single GPU too (parallel ops become no-ops)
        embedding_dtype = config.dtype
        if config.float8_config and config.float8_config.embedding_output_dtype:
            embedding_dtype = config.float8_config.embedding_output_dtype

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
        self.embedding_multiplier = config.embedding_multiplier

    def __call__(
        self,
        tokens: TensorValueLike,
        kv_collections: list[PagedCacheValues],
        return_n_logits: TensorValue,
        input_row_offsets: TensorValue,
        signal_buffers: list[BufferValue],
    ) -> tuple[TensorValue, ...]:
        """Forward pass through the model.

        Args:
            tokens: Input token IDs.
            kv_collections: KV cache per device.
            return_n_logits: Number of logits to return.
            input_row_offsets: Row offsets for ragged batching.
            signal_buffers: Signal buffers for allreduce (required even for single GPU).

        Returns:
            Tuple of logits tensors.
        """
        # Get embeddings - VocabParallelEmbedding returns list[TensorValue]
        h = self.embed_tokens(tokens, signal_buffers)

        # Apply embedding multiplier if needed
        if self.embedding_multiplier != 1.0:
            h = [hi * self.embedding_multiplier for hi in h]

        # Distribute RoPE frequencies and row offsets to all devices
        freqs_cis = distribute_value(self.rope.freqs_cis, self.devices)
        input_row_offsets_list = distribute_value(
            input_row_offsets, self.devices
        )

        # Process through transformer layers
        for idx, layer in enumerate(self.layers):
            layer_idx = ops.constant(idx, DType.uint32, device=DeviceRef.CPU())
            h = layer(
                layer_idx,
                h,
                kv_collections,
                freqs_cis,
                input_row_offsets_list,
                signal_buffers,
            )

        # Get last token for logits computation
        h0 = h[0]  # Use first device's output for indexing
        last_token_indices = input_row_offsets[1:] - 1
        last_token_h = ops.gather(h0, last_token_indices, axis=0)
        last_token_distributed = distribute_value(last_token_h, self.devices)

        # Apply final norm
        norm_last_token = forward_sharded_layers(
            self.norm_shards, last_token_distributed
        )

        # Get logits - ColumnParallelLinear returns list[TensorValue]
        last_logits = ops.cast(
            self.lm_head(norm_last_token, signal_buffers)[0],
            DType.float32,
        )

        # Handle additional logits based on return_logits setting
        logits = None
        offsets = None

        if self.return_logits == ReturnLogits.VARIABLE:
            return_n_logits_range = ops.range(
                start=return_n_logits[0],
                stop=0,
                step=-1,
                out_dim="return_n_logits_range",
                dtype=DType.int64,
                device=self.devices[0],
            )
            computed_offsets = (
                ops.unsqueeze(input_row_offsets[1:], -1) - return_n_logits_range
            )
            last_indices = ops.reshape(computed_offsets, shape=(-1,))

            # Gather from all hidden states
            variable_tokens = [
                ops.gather(h_device, last_indices, axis=0) for h_device in h
            ]
            variable_normed = forward_sharded_layers(
                self.norm_shards, variable_tokens
            )

            logits = ops.cast(
                self.lm_head(variable_normed, signal_buffers)[0],
                DType.float32,
            )

            offsets = ops.range(
                0,
                TensorValue(last_indices.shape[0]) + return_n_logits[0],
                return_n_logits[0],
                out_dim="logit_offsets",
                dtype=DType.int64,
                device=self.devices[0],
            )
        elif self.return_logits == ReturnLogits.ALL:
            # Apply normalization to all hidden states and get all logits
            all_normalized = forward_sharded_layers(self.norm_shards, h)

            logits = ops.cast(
                self.lm_head(all_normalized, signal_buffers)[0],
                DType.float32,
            )

            offsets = input_row_offsets

        if logits is not None and offsets is not None:
            return (last_logits, logits, offsets)
        else:
            return (last_logits,)

    def input_types(
        self, kv_params: KVCacheParams
    ) -> tuple[TensorType | BufferType, ...]:
        """Get input types for graph construction.

        Args:
            kv_params: KV cache parameters.

        Returns:
            Tuple of input types for the graph.
        """
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

        kv_inputs = kv_params.get_symbolic_inputs()

        base_inputs: list[TensorType | BufferType] = [
            tokens_type,
            input_row_offsets_type,
            return_n_logits_type,
        ]

        # Always add signal buffer types (required for VocabParallelEmbedding
        # and ColumnParallelLinear even with single GPU)
        signals = Signals(devices=self.devices)
        signal_buffer_types = signals.input_types()

        # Flatten KV types for all devices
        flattened_kv_types = [
            kv_type for sublist in kv_inputs for kv_type in sublist
        ]
        return tuple(base_inputs + signal_buffer_types + flattened_kv_types)
