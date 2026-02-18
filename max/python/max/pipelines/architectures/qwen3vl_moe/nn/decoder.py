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
"""Build a Qwen3VL decoder model that uses paged kv-caching."""

from __future__ import annotations

import functools
from collections.abc import Callable
from typing import Any

from max.dtype import DType
from max.graph import (
    BufferValue,
    DeviceRef,
    ShardingStrategy,
    TensorValue,
    TensorValueLike,
    ops,
)
from max.nn.legacy.comm.allreduce import Allreduce
from max.nn.legacy.embedding import VocabParallelEmbedding
from max.nn.legacy.float8_config import Float8Config
from max.nn.legacy.kv_cache import PagedCacheValues
from max.nn.legacy.layer import LayerList, Module
from max.nn.legacy.linear import MLP, ColumnParallelLinear
from max.nn.legacy.norm import RMSNorm
from max.nn.legacy.transformer.distributed_transformer import (
    DistributedLogitsPostprocessMixin,
    forward_sharded_layers,
)
from max.pipelines.architectures.internvl.embedding_utils import (
    merge_multimodal_embeddings,
)

from ..model_config import Qwen3VLConfig
from .moe import Qwen3VLMoE, Qwen3VLMoEGate
from .text_attention import Qwen3VLMoEDecoderAttentionWithRope
from .text_rotary import Qwen3VLTextRotaryEmbedding


class Qwen3VLMoeTextDecoderLayer(Module):
    def __init__(
        self,
        rope: Qwen3VLTextRotaryEmbedding,
        config: Qwen3VLConfig,
        dtype: DType,
        layer_idx: int,
        float8_config: Float8Config | None = None,
    ):
        super().__init__()
        # Create Multi-head Latent Attention layer.
        head_dim = config.llm_config.kv_params.head_dim
        if config.llm_config.rms_norm_eps is None:
            raise ValueError(
                "rms_norm_eps cannot be None for model that uses RMSNorm."
            )
        mla_kwargs: dict[str, Any] = dict(
            rope=rope,
            num_attention_heads=config.llm_config.num_attention_heads,
            num_key_value_heads=config.llm_config.num_key_value_heads,
            hidden_size=config.llm_config.hidden_size,
            kv_params=config.llm_config.kv_params,
            devices=config.devices,
            dtype=dtype,
            has_bias=False,
            rms_norm_eps=config.llm_config.rms_norm_eps,
            scale=head_dim**-0.5,
            float8_config=float8_config,
        )

        self.self_attn = Qwen3VLMoEDecoderAttentionWithRope(**mla_kwargs)
        self.self_attn.sharding_strategy = ShardingStrategy.tensor_parallel(
            len(config.devices)
        )
        self.self_attn_shards = self.self_attn.shard(config.devices)

        # Create a shardable MLP or MoE layer
        self.mlp = self._get_mlp(config, layer_idx, float8_config)
        self.mlp_shards = self.mlp.shard(config.devices)

        norm_dtype = config.llm_config.norm_dtype or config.dtype

        self.input_layernorm = RMSNorm(
            dim=config.llm_config.hidden_size,
            dtype=norm_dtype,
            eps=config.llm_config.rms_norm_eps,
            multiply_before_cast=False,
        )
        num_devices = len(config.devices)
        self.input_layernorm.sharding_strategy = ShardingStrategy.replicate(
            num_devices
        )
        self.input_layernorm_shards = self.input_layernorm.shard(config.devices)

        self.post_attention_layernorm = RMSNorm(
            dim=config.llm_config.hidden_size,
            dtype=norm_dtype,
            eps=config.llm_config.rms_norm_eps,
            multiply_before_cast=False,
        )
        self.post_attention_layernorm.sharding_strategy = (
            ShardingStrategy.replicate(num_devices)
        )
        self.post_attention_layernorm_shards = (
            self.post_attention_layernorm.shard(config.devices)
        )

        self.devices = config.devices
        self.allreduce = Allreduce(num_accelerators=len(config.devices))

    def __call__(
        self,
        layer_idx: TensorValue,
        xs: list[TensorValue],
        kv_collections: list[PagedCacheValues],
        freqs_cis: list[TensorValue],
        input_row_offsets: list[TensorValue],
        signal_buffers: list[BufferValue],
    ) -> list[TensorValue]:
        """Forward pass through the decoder layer.

        Args:
            layer_idx: Layer index for KV cache.
            xs: Per-device hidden states.
            kv_collections: Per-device KV cache collections.
            freqs_cis: Per-device RoPE frequency tables.
            input_row_offsets: Per-device input row offsets.
            signal_buffers: Per-device signal buffers for synchronization.

        Returns:
            Per-device updated hidden states.
        """
        # Apply input layer norm
        norm_xs = forward_sharded_layers(self.input_layernorm_shards, xs)

        # Self-attention
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
        attn_outs = self.allreduce(attn_out, signal_buffers)

        # Residual connection
        hs = [x + attn_out for x, attn_out in zip(xs, attn_outs, strict=True)]

        # Apply post-attention layer norm
        norm_outs = forward_sharded_layers(
            self.post_attention_layernorm_shards, hs
        )

        # MLP or MoE
        mlp_outs = forward_sharded_layers(self.mlp_shards, norm_outs)
        mlp_outs = self.allreduce(mlp_outs, signal_buffers)

        # Residual connection
        hs = [h + mlp_out for h, mlp_out in zip(hs, mlp_outs, strict=True)]

        return hs

    def _get_mlp(
        self,
        config: Qwen3VLConfig,
        layer_idx: int,
        float8_config: Float8Config | None,
    ) -> MLP | Qwen3VLMoE:
        """Helper function to return a mixture of experts layer or traditional multi-layer perceptron layer
        for the TransformerBlock's mlp depending on the layer idx.

        Args:
            config: Configuration object containing model parameters
            layer_idx: Layer index
            float8_config: Configuration for FP8 quantization.

        Returns:
            List of MLP shards or MoE modules depending on the layer index and config
        """
        if (layer_idx not in config.mlp_only_layers) and (
            config.num_experts > 0
            and (layer_idx + 1) % config.decoder_sparse_step == 0
        ):
            moe_kwargs: dict[str, Any] = dict(
                devices=config.devices,
                hidden_dim=config.llm_config.hidden_size,
                num_experts=config.num_experts,
                num_experts_per_token=config.num_experts_per_tok,
                moe_dim=config.moe_intermediate_size,
                gate_cls=Qwen3VLMoEGate,
                dtype=config.dtype,
                float8_config=float8_config,
            )
            moe = Qwen3VLMoE(**moe_kwargs)
            moe.sharding_strategy = ShardingStrategy.tensor_parallel(
                len(config.devices)
            )
            return moe
        else:
            mlp = MLP(
                dtype=config.dtype,
                quantization_encoding=None,
                hidden_dim=config.llm_config.hidden_size,
                feed_forward_length=config.llm_config.intermediate_size,
                has_bias=False,
                activation_function="silu",
                devices=config.devices,
                float8_config=float8_config,
            )
            mlp.sharding_strategy = ShardingStrategy.tensor_parallel(
                len(config.devices)
            )
            return mlp


class Qwen3VLMoEDecoder(DistributedLogitsPostprocessMixin, Module):
    """Qwen3VL MoE decoder model with support for vision-language tasks.

    This decoder implements the Qwen3VL MoE architecture with:
    - Multi-axis rotary position embeddings (mrope) for 3D position encoding
    - DeepStack visual feature integration at specific layers
    """

    def __init__(self, config: Qwen3VLConfig) -> None:
        super().__init__()
        self.devices = config.devices

        rope = Qwen3VLTextRotaryEmbedding(
            dim=config.llm_config.hidden_size,
            n_heads=config.llm_config.num_attention_heads,
            theta=config.llm_config.rope_theta,
            max_seq_len=config.llm_config.max_seq_len,
            dtype=DType.bfloat16,
            mrope_section=config.mrope_section,
            head_dim=config.llm_config.kv_params.head_dim,
            interleaved=config.llm_config.interleaved_rope_weights,
            scaling_params=config.llm_config.rope_scaling_params,
        )

        # Select norm layer class
        create_norm: Callable[..., Module]
        if config.llm_config.norm_method == "rms_norm":
            if config.llm_config.rms_norm_eps is None:
                raise ValueError(
                    "rms_norm_eps cannot be None for model that uses RMSNorm."
                )
            create_norm = functools.partial(
                RMSNorm,
                config.llm_config.hidden_size,
                config.llm_config.norm_dtype or config.dtype,
                config.llm_config.rms_norm_eps,
                multiply_before_cast=False,
            )
        else:
            raise ValueError(
                f"Unsupported norm method: {config.llm_config.norm_method}"
            )

        # Extract float8_config from the nested llm_config
        float8_config = config.llm_config.float8_config

        # Create decoder layers
        layers = [
            Qwen3VLMoeTextDecoderLayer(
                rope=rope,
                config=config,
                dtype=config.dtype,
                layer_idx=i,
                float8_config=float8_config,
            )
            for i in range(config.llm_config.num_hidden_layers)
        ]

        embedding_output_dtype = config.dtype
        if (
            config.llm_config.float8_config
            and config.llm_config.float8_config.embedding_output_dtype
        ):
            embedding_output_dtype = (
                config.llm_config.float8_config.embedding_output_dtype
            )

        embedding_layer = VocabParallelEmbedding(
            config.llm_config.vocab_size,
            config.llm_config.hidden_size,
            embedding_output_dtype,
            config.devices,
            quantization_encoding=None,
        )

        output = ColumnParallelLinear(
            config.llm_config.hidden_size,
            config.llm_config.vocab_size,
            embedding_output_dtype,
            config.devices,
            quantization_encoding=None,
            tied_weight=(
                embedding_layer.weight
                if config.llm_config.tie_word_embeddings
                else None
            ),
        )

        if config.llm_config.tie_word_embeddings:
            output.set_shared_weight("weight", embedding_layer.weight)

        self.dim = config.llm_config.hidden_size
        self.n_heads = config.llm_config.num_attention_heads
        self.layers = LayerList(layers)
        self.norm = create_norm()
        self.norm.sharding_strategy = ShardingStrategy.replicate(
            len(config.devices)
        )
        self.norm_shards = self.norm.shard(config.devices)

        self.lm_head = output
        self.embed_tokens = embedding_layer
        self.kv_params = config.llm_config.kv_params
        self.rope = rope
        self.return_logits = config.llm_config.return_logits

    def _deepstack_process(
        self,
        hidden_states: TensorValue,
        visual_embeds: TensorValue,
        image_token_indices: TensorValue,
    ) -> TensorValue:
        """Add deepstack visual embeddings to hidden states at visual positions.

        This method implements the DeepStack feature where visual embeddings
        are added to the hidden states at positions corresponding to visual tokens.
        Equivalent to: hidden_states[scatter_indices, :] += visual_embeds

        Args:
            hidden_states: Hidden states (output of a decoder layer) tensor of shape (seq_len, hidden_dim).
            visual_embeds: deepstack visual embeddings. tensor of shape (visual_seqlen, hidden_dim).
            image_token_indices: Per-device image token indices for image embeddings.

        Returns:
            Updated hidden states with visual embeddings added at visual positions. tensor of shape (seq_len, hidden_dim).
        """
        # Ensure visual_embeds has the same dtype as hidden_states
        if visual_embeds.dtype != hidden_states.dtype:
            visual_embeds = ops.cast(visual_embeds, hidden_states.dtype)

        # FIXME(SERVOPT-924): Fuse the scatter_nd and add op into a single kernel.
        # This will improve perf and reduce size of activations.

        # Create a tensor of zeros with the same shape as hidden_states
        zeros = ops.constant(0, hidden_states.dtype, hidden_states.device)
        zeros_like = ops.broadcast_to(zeros, hidden_states.shape)

        # Write the visual_embeds to the appropriate positions in the zeros tensor
        visual_hidden_states = merge_multimodal_embeddings(
            inputs_embeds=zeros_like,
            multimodal_embeddings=visual_embeds,
            image_token_indices=image_token_indices,
        )

        # Add the visual_hidden_states to the hidden_states
        return hidden_states + visual_hidden_states

    def __call__(
        self,
        tokens: TensorValueLike,
        return_n_logits: TensorValue,
        image_embeddings: list[TensorValue],
        image_token_indices: list[TensorValue],
        position_ids: TensorValue,
        mrope_section: list[int],
        kv_collections: list[PagedCacheValues],
        input_row_offsets: list[TensorValue],
        signal_buffers: list[BufferValue],
        deepstack_visual_embeds: list[list[TensorValue]],
    ) -> tuple[TensorValue, ...]:
        """Forward pass of the Qwen3VL MoE decoder.

        Args:
            tokens: Input token IDs.
            return_n_logits: Number of logits to return.
            image_embeddings: List of image embeddings per device.
            image_token_indices: Per-device scatter indices for image embeddings.
            position_ids: 3D position IDs for RoPE.
            mrope_section: MRoPE section configuration.
            kv_collections: Per-device KV cache collections.
            input_row_offsets: Per-device input row offsets.
            signal_buffers: Per-device signal buffers for synchronization.
            deepstack_visual_embeds: List of deepstack visual embeddings,
                one per layer that needs deepstack processing.
                deepstack_visual_embeds[layer_idx] is the visual embeddings for the layer_idx-th layer. one per device.

        Returns:
            Tuple of logits tensors.
        """
        h = self.embed_tokens(tokens, signal_buffers)

        # Merge image embeddings into text embeddings
        h = [
            merge_multimodal_embeddings(
                inputs_embeds=h_device,
                multimodal_embeddings=image_embeddings_device,
                image_token_indices=image_token_indices_device,
            )
            for h_device, image_embeddings_device, image_token_indices_device in zip(
                h,
                image_embeddings,
                image_token_indices,
                strict=True,
            )
        ]

        # Create position embeddings shared across the decoder layers
        freqs_cis_value = self.rope.freqs_cis_position_ids(position_ids)
        freqs_cis = [freqs_cis_value.to(device) for device in self.devices]

        # Process through decoder layers
        for layer_idx, layer in enumerate(self.layers):
            # Create layer index tensor
            layer_idx_tensor = ops.constant(
                layer_idx, DType.uint32, device=DeviceRef.CPU()
            )

            # Forward through layer
            h = layer(
                layer_idx_tensor,
                h,
                kv_collections,
                freqs_cis,
                input_row_offsets,
                signal_buffers,
            )

            # Apply deepstack processing if needed
            if layer_idx < len(deepstack_visual_embeds):
                # Apply deepstack to each device's hidden states
                visual_embeds = deepstack_visual_embeds[layer_idx]
                h = [
                    self._deepstack_process(
                        hidden_states=h_device,
                        visual_embeds=visual_embeds_device,
                        image_token_indices=image_token_indices_device,
                    )
                    for h_device, visual_embeds_device, image_token_indices_device in zip(
                        h,
                        visual_embeds,
                        image_token_indices,
                        strict=True,
                    )
                ]

        return self._postprocess_logits(
            h, input_row_offsets, return_n_logits, signal_buffers
        )
