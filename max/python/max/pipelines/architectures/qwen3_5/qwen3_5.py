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
"""Qwen3.5 hybrid attention model (linear + full attention layers)."""

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
from max.nn.embedding import VocabParallelEmbedding
from max.nn.kv_cache import KVCacheParamInterface, PagedCacheValues
from max.nn.layer import LayerList, Module
from max.nn.linear import MLP, ColumnParallelLinear, Linear
from max.nn.norm import RMSNorm
from max.nn.rotary_embedding import Llama3RotaryEmbedding
from max.nn.transformer import forward_sequential_layers
from max.nn.transformer.distributed_transformer import (
    DistributedLogitsPostprocessMixin,
)
from max.pipelines.lib.vlm_utils import merge_multimodal_embeddings

from .layers.attention import Qwen3_5Attention
from .layers.gated_deltanet import GatedDeltaNet
from .layers.visual_transformer import VisionTransformer
from .model_config import Qwen3_5Config


class Qwen3_5FullAttentionBlock(Module):
    """Full-attention transformer block (KV cache path)."""

    def __init__(
        self,
        config: Qwen3_5Config,
        layer_idx: int,
        rope: Llama3RotaryEmbedding,
        create_norm: Callable[..., RMSNorm],
        linear_cls: Callable[..., Linear],
    ) -> None:
        super().__init__()
        self.self_attn = Qwen3_5Attention(
            num_attention_heads=config.num_attention_heads,
            num_key_value_heads=config.num_key_value_heads,
            hidden_size=config.hidden_size,
            head_dim=config.kv_params.head_dim,
            kv_params=config.kv_params,
            layer_idx=layer_idx,
            dtype=config.dtype,
            rope=rope,
            linear_cls=linear_cls,
            devices=config.devices,
            scale=config.attention_multiplier,
            partial_rotary_factor=config.partial_rotary_factor,
            has_bias=config.attention_bias,
            norm_dtype=config.norm_dtype or config.dtype,
            norm_eps=config.rms_norm_eps or 1e-6,
        )
        self.mlp = MLP(
            config.dtype,
            config.model_quantization_encoding,
            config.hidden_size,
            config.intermediate_size,
            config.devices,
            linear_cls,
        )
        self.input_layernorm = create_norm()
        self.post_attention_layernorm = create_norm()

    def __call__(
        self,
        x: TensorValue,
        layer_idx: TensorValue,
        kv_blocks: BufferValue,
        cache_lengths: TensorValue,
        lookup_table: TensorValue,
        max_lengths: TensorValue,
        attention_dispatch_metadata: TensorValue,
        freqs_cis: TensorValue,
        input_row_offsets: TensorValue,
    ) -> TensorValue:
        kv_collection = PagedCacheValues(
            kv_blocks=kv_blocks,
            cache_lengths=cache_lengths,
            lookup_table=lookup_table,
            max_lengths=max_lengths,
            attention_dispatch_metadata=attention_dispatch_metadata,
        )
        residual = x
        h = self.input_layernorm(x)
        h = self.self_attn(
            layer_idx, h, kv_collection, freqs_cis, input_row_offsets
        )
        h = residual + h
        residual = h
        h = self.post_attention_layernorm(h)
        h = self.mlp(h)
        return residual + h


class Qwen3_5LinearAttentionBlock(Module):
    """Linear-attention transformer block (Gated DeltaNet path)."""

    def __init__(
        self,
        config: Qwen3_5Config,
        create_norm: Callable[..., RMSNorm],
        linear_cls: Callable[..., Linear],
    ) -> None:
        super().__init__()
        self.linear_attn = GatedDeltaNet(
            hidden_size=config.hidden_size,
            num_key_heads=config.linear_num_key_heads,
            num_value_heads=config.linear_num_value_heads,
            key_head_dim=config.linear_key_head_dim,
            value_head_dim=config.linear_value_head_dim,
            conv_kernel_size=config.linear_conv_kernel_dim,
            dtype=config.dtype,
            device=config.devices[0],
            rms_norm_eps=config.rms_norm_eps or 1e-6,
            ssm_dtype=config.mamba_ssm_dtype,
        )
        self.mlp = MLP(
            config.dtype,
            config.model_quantization_encoding,
            config.hidden_size,
            config.intermediate_size,
            config.devices,
            linear_cls,
        )
        self.input_layernorm = create_norm()
        self.post_attention_layernorm = create_norm()

    def __call__(
        self,
        x: TensorValue,
        conv_pool: BufferValue,
        recurrent_pool: BufferValue,
        slot_idx: TensorValue,
        input_row_offsets: TensorValue,
    ) -> TensorValue:
        residual = x
        h = self.input_layernorm(x)
        h = self.linear_attn(
            h,
            conv_pool=conv_pool,
            recurrent_pool=recurrent_pool,
            slot_idx=slot_idx,
            input_row_offsets=input_row_offsets,
        )
        h = residual + h
        residual = h
        h = self.post_attention_layernorm(h)
        h = self.mlp(h)
        return residual + h


class Qwen3_5(DistributedLogitsPostprocessMixin, Module):
    """Qwen3.5 hybrid attention model.

    This model uses a mix of full attention (with KV cache) and linear
    attention (Gated DeltaNet) layers. Every full_attention_interval-th
    layer uses full attention, and the rest use linear attention.
    """

    def __init__(self, config: Qwen3_5Config) -> None:
        super().__init__()
        self.config = config
        self.devices = config.devices
        self.num_devices = len(config.devices)

        if config.model_quantization_encoding == QuantizationEncoding.GPTQ:
            raise NotImplementedError("GPTQ Qwen3.5 is not implemented yet")
        if config.model_quantization_encoding is not None:
            raise NotImplementedError("GGUFQ Qwen3.5 is not implemented yet")

        # Create RoPE embedding for full attention layers
        # Only the partial rotary dimension gets rotation
        rotary_dim = int(
            config.kv_params.head_dim * config.partial_rotary_factor
        )
        rope = Llama3RotaryEmbedding(
            dim=config.hidden_size,
            n_heads=config.num_attention_heads,
            theta=config.rope_theta,
            max_seq_len=config.max_seq_len,
            head_dim=rotary_dim,
            interleaved=config.interleaved_rope_weights,
            scaling_params=config.rope_scaling_params,
        )
        self.rope = rope

        # Norm factory (uses (1 + weight) offset for Qwen3.5)
        if config.norm_method != "rms_norm" or config.rms_norm_eps is None:
            raise ValueError(
                "Qwen3.5 requires RMSNorm. Set norm_method='rms_norm' "
                "and provide rms_norm_eps."
            )

        create_norm = functools.partial(
            RMSNorm,
            config.hidden_size,
            dtype=config.norm_dtype or DType.float32,
            eps=config.rms_norm_eps,
            weight_offset=1.0,
            multiply_before_cast=False,
        )

        linear_cls = functools.partial(Linear, quant_config=config.quant_config)

        self.layer_types = config.layer_types
        self.linear_layer_indices = [
            i
            for i, lt in enumerate(config.layer_types)
            if lt == "linear_attention"
        ]

        layers: list[Module] = []
        for i, lt in enumerate(config.layer_types):
            if lt == "full_attention":
                layers.append(
                    Qwen3_5FullAttentionBlock(
                        config=config,
                        layer_idx=i,
                        rope=rope,
                        create_norm=create_norm,
                        linear_cls=linear_cls,
                    )
                )
            else:
                layers.append(
                    Qwen3_5LinearAttentionBlock(
                        config=config,
                        create_norm=create_norm,
                        linear_cls=linear_cls,
                    )
                )
        self.layers = LayerList(layers)

        # Final norm (replicated across devices)
        self.norm = create_norm()
        self.norm.sharding_strategy = ShardingStrategy.replicate(
            self.num_devices
        )
        self.norm_shards = self.norm.shard(config.devices)

        # Embedding and output layers
        embedding_dtype = config.dtype
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

        # Linear attention state dimensions
        self._conv_dim = (
            config.linear_key_head_dim * config.linear_num_key_heads * 2
            + config.linear_value_head_dim * config.linear_num_value_heads
        )
        self._conv_kernel_size = config.linear_conv_kernel_dim
        self._num_v_heads = config.linear_num_value_heads
        self._key_head_dim = config.linear_key_head_dim
        self._value_head_dim = config.linear_value_head_dim

        # Vision encoder (only present in multimodal checkpoints)
        self.vision_encoder: VisionTransformer | None = (
            VisionTransformer(config=config.vision_config)
            if config.vision_config is not None
            else None
        )

    def __call__(
        self,
        tokens: TensorValueLike,
        kv_collections: list[PagedCacheValues],
        return_n_logits: TensorValue,
        input_row_offsets: TensorValue,
        signal_buffers: list[BufferValue],
        slot_idx: TensorValue,
        conv_pools: list[BufferValue],
        recurrent_pools: list[BufferValue],
        image_embeddings: TensorValue | None = None,
        image_token_indices: TensorValue | None = None,
    ) -> tuple[TensorValue, ...]:
        """Forward pass through the hybrid model.

        The conv and recurrent state pools are mutable graph inputs;
        per-linear-layer the slot-indexed SSM kernels read and write them in
        place at slot ``slot_idx[batch_item]``. There are no per-layer state
        graph outputs — the only graph outputs are the logits.

        Args:
            tokens: Input token IDs.
            kv_collections: KV cache per device.
            return_n_logits: Number of logits to return.
            input_row_offsets: Row offsets for ragged batching.
            signal_buffers: Signal buffers for allreduce.
            slot_idx: Per-batch slot indices into the linear-attention pools,
                shape ``[batch_size]`` uint32.
            conv_pools: Per-linear-layer mutable conv state pools,
                shape ``[max_slots, conv_dim, K-1]``.
            recurrent_pools: Per-linear-layer mutable recurrent state pools,
                shape ``[max_slots, num_v_heads, key_dim, val_dim]``.
            image_embeddings: Vision encoder output to merge into token embeddings.
                Shape [vision_merged_seq_len, hidden_size]. None for text-only.
            image_token_indices: Scatter indices for placing image embeddings in
                the token sequence. Shape [vision_merged_seq_len]. None for text-only.

        Returns:
            Tuple of (logits,).
        """
        # Get embeddings — unwrap immediately; this model is single-GPU only.
        h_list = self.embed_tokens(tokens, signal_buffers)
        h: TensorValue = h_list[0] if isinstance(h_list, list) else h_list

        if image_embeddings is not None and image_token_indices is not None:
            # TODO: multi-device — merge must be applied per shard with a
            # matching sharded image_embeddings.
            h = merge_multimodal_embeddings(
                h, image_embeddings, image_token_indices
            )

        # Place RoPE frequencies and row offsets on device
        freqs_cis = self.rope.freqs_cis.to(self.devices[0])
        input_row_offsets = input_row_offsets.to(self.devices[0])

        kv_collection = kv_collections[0]
        # ``forward_sequential_layers`` only introspects ``Value`` and
        # ``Sequence[Value]``, so the dataclass is unpacked into positional
        # args below.
        assert kv_collection.kv_scales is None, (
            "Qwen3.5 does not support quantized KV cache"
        )
        assert kv_collection.draft_attention_dispatch_metadata is None, (
            "Qwen3.5 does not support eagle speculation"
        )
        assert kv_collection.attention_dispatch_metadata is not None
        attention_dispatch_metadata = kv_collection.attention_dispatch_metadata
        kv_cache_idx = 0
        linear_state_idx = 0

        def inputs_for_layer(
            idx: int, hs: list[TensorValue]
        ) -> list[Value[Any] | Sequence[Value[Any]]]:
            nonlocal kv_cache_idx, linear_state_idx
            hidden = hs[0]
            if self.layer_types[idx] == "full_attention":
                # ``layer_idx`` is the sequential index within the KV cache
                # (0-based across full-attention layers only), distinct from
                # the absolute layer index. The KV cache is only allocated for
                # full-attention layers.
                layer_idx_tensor = ops.constant(
                    kv_cache_idx, DType.uint32, device=DeviceRef.CPU()
                )
                kv_cache_idx += 1
                return [
                    hidden,
                    layer_idx_tensor,
                    kv_collection.kv_blocks,
                    kv_collection.cache_lengths,
                    kv_collection.lookup_table,
                    kv_collection.max_lengths,
                    attention_dispatch_metadata,
                    freqs_cis,
                    input_row_offsets,
                ]
            vals: list[Value[Any] | Sequence[Value[Any]]] = [
                hidden,
                conv_pools[linear_state_idx],
                recurrent_pools[linear_state_idx],
                slot_idx,
                input_row_offsets,
            ]
            linear_state_idx += 1
            return vals

        full_attn_indices = [
            i for i, lt in enumerate(self.layer_types) if lt == "full_attention"
        ]
        groups: list[list[int]] = [
            g for g in (full_attn_indices, self.linear_layer_indices) if g
        ]

        h_list = forward_sequential_layers(
            list(self.layers),
            inputs_for_layer=inputs_for_layer,
            initial_hidden_states=[h],
            subgraph_layer_groups=(
                groups if self.config.use_subgraphs else None
            ),
            name_for_subgraph=lambda g: f"qwen3_5_{self.layer_types[groups[g][0]]}_block",
            weight_prefix_for_layer=lambda i: f"layers.{i}.",
        )
        h = h_list[0]

        logits = self._postprocess_logits(
            [h], [input_row_offsets], return_n_logits, signal_buffers
        )
        return tuple(logits)

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

        # Signal buffer types
        signals = Signals(devices=self.devices)
        signal_buffer_types = signals.input_types()

        # Flatten KV types for all devices
        flattened_kv_types = kv_inputs.flatten()

        # Linear-attention state pools. Pools are mutable ``BufferType`` graph
        # inputs in the model's native dtype (typically bf16); the slot-indexed
        # SSM kernels mutate them in place at slot ``slot_idx[batch_item]``.
        # ``slot_idx`` is a single per-step ``[batch_size]`` uint32 tensor.
        num_linear_layers = len(self.linear_layer_indices)
        state_dtype = self.config.dtype
        slot_idx_type = TensorType(
            DType.uint32, shape=["batch_size"], device=device_ref
        )
        conv_pool_types: list[TensorType | BufferType] = [
            BufferType(
                state_dtype,
                shape=[
                    "max_slots",
                    self._conv_dim,
                    self._conv_kernel_size - 1,
                ],
                device=device_ref,
            )
            for _ in range(num_linear_layers)
        ]
        recurrent_pool_types: list[TensorType | BufferType] = [
            BufferType(
                state_dtype,
                shape=[
                    "max_slots",
                    self._num_v_heads,
                    self._key_head_dim,
                    self._value_head_dim,
                ],
                device=device_ref,
            )
            for _ in range(num_linear_layers)
        ]

        vision_types: list[TensorType | BufferType] = []
        if self.vision_encoder is not None:
            vision_types = [
                TensorType(
                    self.config.dtype,
                    shape=["vision_merged_seq_len", self.config.hidden_size],
                    device=device_ref,
                ),
                TensorType(
                    DType.int32,
                    shape=["total_image_tokens"],
                    device=device_ref,
                ),
            ]

        return tuple(
            base_inputs
            + signal_buffer_types
            + flattened_kv_types
            + [slot_idx_type]
            + conv_pool_types
            + recurrent_pool_types
            + vision_types
        )
