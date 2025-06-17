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
"""Implements the InternVL multimodal model."""

from __future__ import annotations

from collections.abc import Iterable, Sequence

from max.dtype import DType
from max.graph import (
    BufferValue,
    DeviceRef,
    ShardingStrategy,
    TensorValue,
    TensorValueLike,
    Weight,
    ops,
)
from max.graph.ops.resize import InterpolationMode
from max.graph.type import Dim, StaticDim
from max.nn import (
    ColumnParallelLinear,
    DistributedAttentionWithRope,
    DistributedMLP,
    DistributedRMSNorm,
    LayerList,
    Linear,
    Llama3RotaryEmbedding,
    Module,
    Shardable,
    VocabParallelEmbedding,
)
from max.nn.kv_cache import (
    FetchPagedKVCacheCollection,
    PagedKVCacheCollection,
)

from .model_config import InternVLConfig


def distribute_value(
    v: TensorValue, devices: Sequence[DeviceRef]
) -> list[TensorValue]:
    """Distributes a tensor value across multiple devices.

    Args:
        v: The tensor value to distribute.
        devices: The list of devices to distribute the tensor across.

    Returns:
        A list of tensor values, one per device.
    """
    return [v.to(device) for device in devices]


class InternVLDecoderLayer(Module):
    """Represents a single decoder layer in the InternVL language model.

    This layer follows the Qwen2 architecture, which is similar to Llama3 but
    includes attention bias. Each layer contains:

    - Self-attention with RoPE (Rotary Position Embeddings)
    - Feed-forward network (MLP)
    - RMS normalization before attention and MLP
    - Residual connections
    """

    def __init__(
        self,
        layer_idx: int,
        rope: Llama3RotaryEmbedding,
        config: InternVLConfig,
    ):
        """Initializes a decoder layer.

        Args:
            layer_idx: The index of this layer in the model.
            rope: The rotary position embedding module.
            config: The InternVL configuration containing model parameters.
        """
        super().__init__()
        llm_config = config.llm_config
        devices = config.devices

        self.self_attn = DistributedAttentionWithRope(
            stacked_qkv=llm_config.stacked_qkv,
            scale=llm_config.attention_multiplier,
            clip_qkv=llm_config.clip_qkv,
            num_attention_heads=llm_config.num_attention_heads,
            num_key_value_heads=llm_config.num_key_value_heads,
            hidden_size=llm_config.hidden_size,
            kv_params=llm_config.kv_params,
            dtype=llm_config.dtype,
            rope=rope,
            linear_cls=Linear,
            devices=devices,
            has_bias=True,  # Qwen2 uses attention bias
        )

        self.mlp = DistributedMLP(
            llm_config.dtype,
            quantization_encoding=None,
            hidden_dim=llm_config.hidden_size,
            feed_forward_length=llm_config.intermediate_size,
            devices=devices,
            linear_cls=Linear,
        )

        self.input_layernorm = DistributedRMSNorm(
            dim=llm_config.hidden_size,
            dtype=llm_config.dtype,
            eps=llm_config.rms_norm_eps,
            devices=devices,
        )
        self.post_attention_layernorm = DistributedRMSNorm(
            dim=llm_config.hidden_size,
            dtype=llm_config.dtype,
            eps=llm_config.rms_norm_eps,
            devices=devices,
        )

    def __call__(
        self,
        layer_idx: TensorValue,
        xs: Sequence[TensorValue],
        signal_buffers: Sequence[BufferValue],
        kv_collections: Sequence[PagedKVCacheCollection],
        input_row_offsets: Sequence[TensorValue],
    ) -> list[TensorValue]:
        """Processes input through the decoder layer.

        Args:
            layer_idx: The index of this layer in the model.
            xs: The input hidden states, one per device.
            signal_buffers: Communication buffers for distributed execution.
            kv_collections: Key-value cache collections for each device.
            input_row_offsets: Offsets for flattened input sequences.

        Returns:
            The output hidden states after attention and MLP, one per device.
        """
        attn_outs = self.self_attn(
            layer_idx,
            self.input_layernorm(xs),
            signal_buffers,
            kv_collections,
            input_row_offsets,
        )

        # Add residual.
        hs = [x + attn_out for x, attn_out in zip(xs, attn_outs)]

        mlp_outs = self.mlp(self.post_attention_layernorm(hs), signal_buffers)

        # Add residual.
        hs = [h + mlp_out for h, mlp_out in zip(hs, mlp_outs)]

        return hs


class InternVLLanguageModel(Module):
    """Implements the language model component of InternVL.

    This model is based on Qwen2.5 architecture and is designed to process
    text tokens and generate language outputs. It supports multimodal inputs
    through embedding merging (implemented separately).

    The model consists of:
    - Token embeddings
    - Multiple decoder layers with attention and feed-forward networks
    - RoPE position embeddings
    - Final layer normalization and output projection

    Note:
        This implementation assumes:
        - Multiple device execution (distributed)
        - Paged KV cache only
        - No quantization
        - No tied embeddings
        - Last token logits only (no variable logits)
    """

    def __init__(self, config: InternVLConfig):
        """Initializes the InternVL language model.

        Args:
            config: The InternVL configuration containing model parameters,
                including the language model config, devices, and other settings.

        Raises:
            ValueError: If tied embeddings are requested (not supported).
        """
        super().__init__()
        llm_config = config.llm_config
        self.devices = config.devices

        if config.llm_config.tie_word_embeddings:
            raise ValueError("tied embeddings unsupported by InternVL")

        # Create RoPE embeddings.
        self.rope = Llama3RotaryEmbedding(
            dim=llm_config.hidden_size,
            n_heads=llm_config.num_attention_heads,
            theta=llm_config.rope_theta,
            max_seq_len=llm_config.max_seq_len,
            interleaved=llm_config.interleaved_rope_weights,
            scaling_params=llm_config.rope_scaling_params,
            device=self.devices[0],
        )

        # Create decoder layers.
        self.layers = LayerList(
            [
                InternVLDecoderLayer(layer_idx, self.rope, config)
                for layer_idx in range(llm_config.num_hidden_layers)
            ]
        )

        self.norm = DistributedRMSNorm(
            dim=llm_config.hidden_size,
            dtype=llm_config.dtype,
            eps=llm_config.rms_norm_eps,
            devices=self.devices,
        )

        self.embed_tokens = VocabParallelEmbedding(
            llm_config.vocab_size,
            llm_config.hidden_size,
            llm_config.dtype,
            self.devices,
            quantization_encoding=None,
        )

        self.lm_head = ColumnParallelLinear(
            llm_config.hidden_size,
            llm_config.vocab_size,
            llm_config.dtype,
            devices=self.devices,
            quantization_encoding=None,
        )

        # Always assume paged KV cache for InternVL.
        self.kv_collection_constructor = FetchPagedKVCacheCollection(
            llm_config.kv_params,
            num_layers=llm_config.num_hidden_layers,
        )

    def __call__(
        self,
        tokens: TensorValueLike,
        signal_buffers: Iterable[BufferValue],
        kv_cache_inputs_per_dev: Sequence[tuple[TensorValue, ...]],
        return_n_logits: TensorValue,
        input_row_offsets: TensorValue,
    ) -> tuple[TensorValue, ...]:
        """Executes the language model forward pass.

        Args:
            tokens: Input token IDs to process.
            signal_buffers: Communication buffers for distributed execution.
            kv_cache_inputs_per_dev: KV cache inputs for each device.
            return_n_logits: Number of logits to return (unused, always returns last).
            input_row_offsets: Offsets for flattened input sequences.

        Returns:
            A tuple containing the output logits for the last token positions.
        """
        # Get embeddings.
        h = self.embed_tokens(tokens, signal_buffers)

        # Create KV cache collections.
        kv_collections = [
            self.kv_collection_constructor(*kv_cache_inputs)
            for kv_cache_inputs in kv_cache_inputs_per_dev
        ]

        # Distribute input row offsets
        input_row_offsets_ = distribute_value(input_row_offsets, self.devices)

        # Run through decoder layers
        for idx, layer in enumerate(self.layers):
            h = layer(
                ops.constant(idx, DType.uint32, device=DeviceRef.CPU()),
                h,
                signal_buffers,
                kv_collections,
                input_row_offsets_,
            )

        # Get last token logits only (no variable logits support)
        h0 = h[0]
        last_token_indices = input_row_offsets[1:] - 1
        last_token_h = ops.gather(h0, last_token_indices, axis=0)
        last_token_distributed = distribute_value(last_token_h, self.devices)
        last_logits = ops.cast(
            self.lm_head(self.norm(last_token_distributed))[0], DType.float32
        )

        return (last_logits,)


class InternVisionEmbeddings(Module, Shardable):
    def __init__(
        self, config: InternVLConfig, device: DeviceRef | None = None
    ) -> None:
        self.config = config
        self.devices = config.devices
        self.embed_dim = config.vision_config.hidden_size
        self.image_size = config.vision_config.image_size
        self.patch_size = config.vision_config.patch_size
        self.dtype = config.vision_config.dtype

        # Calculate patch dimensions
        # Note: in_dim matches Conv2D flattening order (C*H*W)
        self.patch_embedding = Linear(
            in_dim=3 * self.patch_size * self.patch_size,
            out_dim=self.embed_dim,
            dtype=self.dtype,
            device=device if device else DeviceRef.CPU(),
            has_bias=True,
        )

        self.num_patches = (self.image_size // self.patch_size) ** 2
        self.num_positions = self.num_patches + 1

        self.class_embedding = Weight(
            "class_embedding",
            dtype=self.dtype,
            shape=(1, 1, self.embed_dim),
            device=DeviceRef.CPU() if not device else device,
        )

        self.position_embedding = Weight(
            "position_embedding",
            dtype=self.dtype,
            shape=(1, self.num_positions, self.embed_dim),
            device=DeviceRef.CPU() if not device else device,
        )

    @property
    def sharding_strategy(self) -> ShardingStrategy | None:
        """Get the embedding sharding strategy."""
        return self.patch_embedding.sharding_strategy

    @sharding_strategy.setter
    def sharding_strategy(self, strategy: ShardingStrategy) -> None:
        """Set the sharding strategy for the patch, class, and position
        embeddings.

        Args:
            strategy: The strategy describing the embeddings' sharding.
        """
        if not strategy.is_replicate:
            raise ValueError(
                "only replicate is supported for InternVisionEmbeddings, "
                "currently"
            )

        self.patch_embedding.sharding_strategy = strategy
        self.class_embedding.sharding_strategy = strategy
        self.position_embedding.sharding_strategy = strategy

    def shard(
        self, shard_idx: int, device: DeviceRef
    ) -> InternVisionEmbeddings:
        """Creates a sharded view of this Linear layer for a specific device.

        Args:
            shard_idx: The index of the shard (0 to num_devices-1).
            device: The device where this shard should reside.

        Returns:
            A sharded Linear instance.
        """
        # This should be set unconditionally in the constructor.
        assert self.sharding_strategy

        # Create the new sharded embedding.
        sharded = InternVisionEmbeddings(self.config, device)

        # Shard the embedding fields.
        sharded.patch_embedding = self.patch_embedding.shard(shard_idx, device)
        sharded.class_embedding = self.class_embedding.shard(shard_idx, device)
        sharded.position_embedding = self.position_embedding.shard(
            shard_idx, device
        )

        return sharded

    def _get_position_embedding(self, H: Dim, W: Dim) -> TensorValue:
        """Get position embeddings, interpolating if needed for different resolutions.

        Args:
            H: Height in patches (can be int or symbolic Dim)
            W: Width in patches (can be int or symbolic Dim)
        """
        # For static dimensions, check if we need interpolation.
        if isinstance(H, StaticDim) and isinstance(W, StaticDim):
            h_int = int(H)
            w_int = int(W)
            if self.num_patches == h_int * w_int:
                return self.position_embedding

        # Otherwise, interpolate position embeddings.
        # Split class token and patch position embeddings.
        class_pos_embed = self.position_embedding[:, :1, :]
        patch_pos_embed = self.position_embedding[:, 1:, :]

        # Reshape patch position embeddings to spatial layout.
        orig_size = int(self.num_patches**0.5)
        patch_pos_embed = ops.reshape(
            patch_pos_embed, [1, orig_size, orig_size, self.embed_dim]
        )

        # Permute to NCHW format for interpolation.
        patch_pos_embed = ops.permute(patch_pos_embed, [0, 3, 1, 2])

        # Interpolate using bicubic.
        # resize expects full shape (N, C, H, W).
        patch_pos_embed = ops.resize(
            patch_pos_embed,
            shape=[1, self.embed_dim, H, W],
            interpolation=InterpolationMode.BICUBIC,
        )

        # Permute back to NHWC and reshape.
        patch_pos_embed = ops.permute(patch_pos_embed, [0, 2, 3, 1])
        patch_pos_embed = ops.reshape(
            patch_pos_embed, [1, H * W, self.embed_dim]
        )

        # Concatenate class token and interpolated patch embeddings
        return ops.concat([class_pos_embed, patch_pos_embed], axis=1)

    def __call__(self, pixel_values: TensorValue) -> TensorValue:
        """Compute embeddings for input pixel values.

        Args:
            pixel_values: Input image tensor of shape (batch, height, width, channels).

        Returns:
            Embeddings tensor of shape (batch, num_positions, embed_dim).
        """
        # pixel_values is in BHWC format
        batch_size = pixel_values.shape[0]
        img_height = pixel_values.shape[1]
        img_width = pixel_values.shape[2]

        # Calculate number of patches
        height = img_height // self.patch_size
        width = img_width // self.patch_size

        # 1. Reshape to extract patches
        # From (B, H, W, C) to (B, H/P, P, W/P, P, C)
        # Rebind `pixel_values` to be an explicit multiple of the `patch_size`.
        # This is asserting that at runtime the `img_height` and `img_width`
        # will both be divisible by `patch_size`.
        pixel_values_rebind = ops.rebind(
            pixel_values,
            [
                batch_size,
                self.patch_size * (img_height // self.patch_size),
                self.patch_size * (img_width // self.patch_size),
                3,
            ],
        )
        pixel_values = ops.reshape(
            pixel_values_rebind,
            [
                batch_size,
                height,
                self.patch_size,
                width,
                self.patch_size,
                3,
            ],
        )

        # 2. Permute to group patch pixels together
        # From (B, H/P, P, W/P, P, C) to (B, H/P, W/P, P, P, C)
        pixel_values = ops.permute(pixel_values, [0, 1, 3, 2, 4, 5])

        # 2.5. Permute within each patch from HWC to CHW to match Conv2D weight layout
        # From (B, H/P, W/P, P, P, C) to (B, H/P, W/P, C, P, P)
        pixel_values = ops.permute(pixel_values, [0, 1, 2, 5, 3, 4])

        # 3. Reshape to (B, num_patches, channels * patch_size * patch_size)
        pixel_values = ops.reshape(
            pixel_values,
            [
                batch_size,
                height * width,
                3 * self.patch_size * self.patch_size,
            ],
        )

        # 4. Apply linear transformation
        pixel_values = pixel_values.cast(self.patch_embedding.weight.dtype)
        patch_embeds = self.patch_embedding(pixel_values)

        # 3. Add class token
        class_embeds = self.class_embedding.broadcast_to(
            (batch_size, 1, self.embed_dim)
        )
        embeddings = ops.concat([class_embeds, patch_embeds], axis=1)

        # 4. Add position embeddings
        # Handle both static and symbolic dimensions
        position_embedding = self._get_position_embedding(height, width)
        embeddings = embeddings + position_embedding

        return embeddings


class InternVLVisionModel(Module):
    """The InternVL vision model for processing images."""

    def __init__(self, config: InternVLConfig) -> None:
        super().__init__()
        self.config = config

        self.embeddings = InternVisionEmbeddings(config)
        self.embeddings.sharding_strategy = ShardingStrategy.replicate(
            len(config.devices)
        )

        self.embeddings_list = [
            self.embeddings.shard(n, dev)
            for n, dev in enumerate(config.devices)
        ]

    def __call__(
        self, pixel_values: Sequence[TensorValue]
    ) -> Sequence[TensorValue]:
        """Process pixel values to image embeddings.

        Args:
            pixel_values: Input pixel values tensor.

        Returns:
            Image embeddings tensor.
        """
        hidden_states = [
            # Call one forward per device -- embeddings are replicated.
            embed(pixels)
            for embed, pixels in zip(self.embeddings_list, pixel_values)
        ]

        # TODO: Implement vision encoder
        # 1. Process pixel values through InternViT encoder
        # 2. Apply multimodal projector

        return hidden_states
