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
"""Build a Qwen3 model that supports single-GPU, multi-GPU TP, and DP+EP."""

from __future__ import annotations

import functools
from collections.abc import Callable, Sequence

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
from max.nn.comm import Signals
from max.nn.comm.allreduce import Allreduce
from max.nn.comm.ep import EPBatchManager
from max.nn.data_parallelism import split_batch_replicated
from max.nn.embedding import VocabParallelEmbedding
from max.nn.kv_cache import KVCacheParamInterface, PagedCacheValues
from max.nn.layer import LayerList, Module
from max.nn.linear import MLP, ColumnParallelLinear, Linear
from max.nn.moe import MoE, MoEQuantized
from max.nn.moe.expert_parallel import forward_moe_sharded_layers
from max.nn.norm import RMSNorm
from max.nn.rotary_embedding import Llama3RotaryEmbedding
from max.nn.transformer.distributed_transformer import (
    DistributedLogitsPostprocessMixin,
    ReturnLogits,
    forward_sharded_layers,
)
from max.pipelines.architectures.qwen3.layers.attention import Qwen3Attention
from max.pipelines.architectures.qwen3.layers.moe import Qwen3MoEGate
from max.pipelines.architectures.qwen3.model_config import Qwen3Config


class Qwen3TransformerBlock(Module):
    """Qwen3 transformer block supporting TP and DP+EP parallelism strategies.

    In TP mode: attention and MLP/MoE weights are sharded across devices,
    allreduce combines partial outputs.

    In DP mode: attention weights are replicated (each device processes its own
    batch shard), MoE uses expert parallelism. No allreduce needed for attention.
    """

    def __init__(
        self,
        config: Qwen3Config,
        layer_idx: int,
        rope: Llama3RotaryEmbedding,
        create_norm: Callable[..., RMSNorm],
        linear_cls: Callable[..., Linear],
        ep_manager: EPBatchManager | None = None,
    ) -> None:
        super().__init__()
        self.devices = config.devices
        num_devices = len(config.devices)
        self.use_dp = config.data_parallel_degree > 1

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
            norm_dtype=config.norm_dtype or config.dtype,
            quant_config=config.quant_config,
        )

        if self.use_dp:
            self.self_attn.sharding_strategy = ShardingStrategy.replicate(
                num_devices
            )
        else:
            self.self_attn.sharding_strategy = ShardingStrategy.tensor_parallel(
                num_devices
            )
        self.self_attn_shards = self.self_attn.shard(config.devices)

        # Create MLP or MoE layer
        self.mlp = self._get_mlp(config, layer_idx, linear_cls, ep_manager)

        if self.use_dp:
            if hasattr(self.mlp, "_ep_batch_manager") and ep_manager:
                self.mlp.sharding_strategy = ShardingStrategy.expert_parallel(
                    num_devices
                )
            else:
                self.mlp.sharding_strategy = ShardingStrategy.replicate(
                    num_devices
                )
        else:
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

        # Allreduce for combining sharded outputs (only used in TP mode)
        self.allreduce = Allreduce(num_accelerators=num_devices)
        self.residual_multiplier = config.residual_multiplier

    def _get_mlp(
        self,
        config: Qwen3Config,
        layer_idx: int,
        linear_cls: Callable[..., Linear],
        ep_manager: EPBatchManager | None = None,
    ) -> MLP | MoE:
        """Get MLP or MoE layer based on config and layer index."""
        use_moe = (
            config.num_experts > 0
            and layer_idx not in config.mlp_only_layers
            and (layer_idx + 1) % config.decoder_sparse_step == 0
        )

        if use_moe:
            ep_size = (
                config.ep_config.n_gpus_per_node * config.ep_config.n_nodes
                if config.ep_config is not None
                else 1
            )
            moe_cls = MoEQuantized if config.quant_config is not None else MoE
            return moe_cls(
                devices=config.devices,
                hidden_dim=config.hidden_size,
                num_experts=config.num_experts,
                num_experts_per_token=config.num_experts_per_tok,
                moe_dim=config.moe_intermediate_size,
                gate_cls=Qwen3MoEGate,
                dtype=config.dtype,
                quant_config=config.quant_config,
                ep_size=ep_size,
                ep_batch_manager=ep_manager,
            )
        else:
            return MLP(
                config.dtype,
                config.model_quantization_encoding,
                config.hidden_size,
                config.intermediate_size,
                config.devices,
                linear_cls,
                quant_config=config.quant_config,
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
        """Forward pass through the block."""
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

        # Allreduce attention outputs (TP mode only)
        if not self.use_dp and len(self.devices) > 1:
            attn_outs = self.allreduce(attn_outs, signal_buffers)

        # Residual connection
        hs = [x + attn_out for x, attn_out in zip(xs, attn_outs, strict=True)]
        if self.residual_multiplier != 1.0:
            hs = [h * self.residual_multiplier for h in hs]

        # Apply post-attention layer norm
        norm_outs = forward_sharded_layers(
            self.post_attention_layernorm_shards, hs
        )

        # MLP/MoE
        if self.use_dp:
            mlp_outs = forward_moe_sharded_layers(self.mlp_shards, norm_outs)
        else:
            mlp_outs = forward_sharded_layers(self.mlp_shards, norm_outs)
            if len(self.devices) > 1:
                mlp_outs = self.allreduce(mlp_outs, signal_buffers)

        # Residual connection
        hs = [h + mlp_out for h, mlp_out in zip(hs, mlp_outs, strict=True)]

        return hs


def _dp_logits_postprocess(
    h: list[TensorValue],
    input_row_offsets: list[TensorValue],
    return_n_logits: TensorValue,
    norm_shards: Sequence[Callable[[TensorValue], TensorValue]],
    lm_head: Callable[
        [list[TensorValue], Sequence[BufferValue]], Sequence[TensorValue]
    ],
    signal_buffers: list[BufferValue],
    devices: list[DeviceRef],
    return_logits: ReturnLogits,
) -> tuple[TensorValue, ...]:
    """Logits postprocessing for DP mode.

    In DP mode each device has hidden states for its own batch shard. We gather
    last-token hidden states from all devices before the vocab-parallel LM head.
    """
    last_token_per_dev: list[TensorValue] = []
    for dev_idx in range(len(devices)):
        last_token_indices = input_row_offsets[dev_idx][1:] - 1
        last_token_h = ops.gather(h[dev_idx], last_token_indices, axis=0)
        last_token_per_dev.append(last_token_h)

    last_token_distributed = ops.allgather(last_token_per_dev, signal_buffers)

    norm_last_token = forward_sharded_layers(
        norm_shards, last_token_distributed
    )
    last_logits = ops.cast(
        lm_head(norm_last_token, signal_buffers)[0],
        DType.float32,
    )

    return (last_logits,)


class Qwen3(DistributedLogitsPostprocessMixin, Module):
    """Unified Qwen3 model supporting single-GPU, TP, and DP+EP inference."""

    def __init__(
        self, config: Qwen3Config, ep_manager: EPBatchManager | None = None
    ) -> None:
        super().__init__()
        self.config = config
        self.devices = config.devices
        self.num_devices = len(config.devices)
        self.use_dp = config.data_parallel_degree > 1
        self.ep_manager = ep_manager

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

        linear_cls = functools.partial(Linear, quant_config=config.quant_config)

        # Create transformer layers
        self.layers = LayerList(
            [
                Qwen3TransformerBlock(
                    config=config,
                    layer_idx=i,
                    rope=rope,
                    create_norm=create_norm,
                    linear_cls=linear_cls,
                    ep_manager=ep_manager,
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
        self.embedding_multiplier = config.embedding_multiplier

    def __call__(
        self,
        tokens: TensorValueLike,
        kv_collections: list[PagedCacheValues],
        return_n_logits: TensorValue,
        input_row_offsets: TensorValue,
        signal_buffers: list[BufferValue],
        host_input_row_offsets: TensorValue | None = None,
        data_parallel_splits: TensorValue | None = None,
    ) -> tuple[TensorValue, ...]:
        """Forward pass through the model.

        Args:
            tokens: Input token IDs.
            kv_collections: KV cache per device.
            return_n_logits: Number of logits to return.
            input_row_offsets: Row offsets for ragged batching.
            signal_buffers: Signal buffers for allreduce.
            host_input_row_offsets: CPU-side row offsets (required for DP mode).
            data_parallel_splits: Batch splits per device (required for DP mode).
        """
        # Get embeddings
        h = self.embed_tokens(tokens, signal_buffers)

        if self.embedding_multiplier != 1.0:
            h = [hi * self.embedding_multiplier for hi in h]

        # Distribute RoPE frequencies and row offsets to all devices
        freqs_cis = [self.rope.freqs_cis.to(device) for device in self.devices]
        input_row_offsets_list = ops.distributed_broadcast(
            input_row_offsets.to(self.devices[0]), signal_buffers
        )

        # In DP mode, split batch across devices
        if self.use_dp and data_parallel_splits is not None:
            assert host_input_row_offsets is not None
            h, input_row_offsets_list = split_batch_replicated(
                self.devices,
                h,
                input_row_offsets_list,
                host_input_row_offsets.cast(DType.int64),
                data_parallel_splits,
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

        if self.use_dp:
            return _dp_logits_postprocess(
                h,
                input_row_offsets_list,
                return_n_logits,
                norm_shards=self.norm_shards,
                lm_head=self.lm_head,
                signal_buffers=signal_buffers,
                devices=self.devices,
                return_logits=self.return_logits,
            )

        return self._postprocess_logits(
            h, input_row_offsets_list, return_n_logits, signal_buffers
        )

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

        kv_inputs = kv_params.get_symbolic_inputs()

        base_inputs: list[TensorType | BufferType] = [
            tokens_type,
            input_row_offsets_type,
            return_n_logits_type,
        ]

        # DP mode needs additional inputs for batch splitting
        if self.use_dp:
            host_input_row_offsets_type = TensorType(
                DType.uint32,
                shape=["input_row_offsets_len"],
                device=DeviceRef.CPU(),
            )
            data_parallel_splits_type = TensorType(
                DType.int64,
                shape=[self.num_devices + 1],
                device=DeviceRef.CPU(),
            )
            base_inputs.extend(
                [host_input_row_offsets_type, data_parallel_splits_type]
            )

        # Signal buffers
        signals = Signals(devices=self.devices)
        signal_buffer_types = signals.input_types()

        # KV cache inputs
        flattened_kv_types = kv_inputs.flatten()

        # EP inputs
        ep_input_types: list[TensorType | BufferType] = []
        if self.ep_manager is not None:
            ep_input_types = list(self.ep_manager.input_types())

        return tuple(
            base_inputs
            + signal_buffer_types
            + flattened_kv_types
            + ep_input_types
        )
