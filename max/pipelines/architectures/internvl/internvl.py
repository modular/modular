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
"""Implements the InternVL multimodal model architecture."""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

from max.dtype import DType
from max.graph import (
    BufferValue,
    DeviceRef,
    ShardingStrategy,
    TensorValue,
    Weight,
    ops,
)
from max.graph.ops.allgather import allgather
from max.graph.ops.resize import InterpolationMode
from max.graph.type import Dim, StaticDim
from max.graph.weight import _compute_shard_range
from max.nn import (
    Allreduce,
    ColumnParallelLinear,
    DistributedAttentionWithRope,
    DistributedMLP,
    DistributedRMSNorm,
    LayerList,
    LayerNorm,
    Linear,
    Llama3RotaryEmbedding,
    Module,
    RMSNorm,
    Shardable,
    VocabParallelEmbedding,
)
from max.nn.attention.mask_config import MHAMaskVariant
from max.nn.kernels import flash_attention_gpu
from max.nn.kv_cache import (
    FetchPagedKVCacheCollection,
    PagedKVCacheCollection,
)

from .embedding_utils import merge_multimodal_embeddings
from .layers.attention import (
    InternVLMultiheadAttention,
    compute_heads_per_device,
)
from .model_config import InternVLConfig


@dataclass
class DeviceAttentionParams:
    """Parameters for attention computation on a specific device."""

    device_heads: int
    head_start: int
    head_dim: int


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
        rope: Llama3RotaryEmbedding,
        config: InternVLConfig,
    ) -> None:
        """Initializes a decoder layer.

        Args:
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

    def __init__(
        self, config: InternVLConfig, image_context_token_id: int
    ) -> None:
        """Initializes the InternVL language model.

        Args:
            config: The InternVL configuration containing model parameters,
                including the language model config, devices, and other settings.
            image_context_token_id: Token ID for image context tokens.

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
                InternVLDecoderLayer(self.rope, config)
                for _ in range(llm_config.num_hidden_layers)
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

        # Store image context token ID.
        self.image_context_token_id = image_context_token_id

    def __call__(
        self,
        tokens: TensorValue,
        signal_buffers: Iterable[BufferValue],
        kv_cache_inputs_per_dev: Sequence[tuple[TensorValue, ...]],
        return_n_logits: TensorValue,
        input_row_offsets: TensorValue,
        image_embeddings: Sequence[TensorValue],
        image_token_indices: TensorValue,
    ) -> tuple[TensorValue, ...]:
        """Executes the language model forward pass.

        Args:
            tokens: Input token IDs to process.
            signal_buffers: Communication buffers for distributed execution.
            kv_cache_inputs_per_dev: KV cache inputs for each device.
            return_n_logits: Number of logits to return (unused, always returns last).
            input_row_offsets: Offsets for flattened input sequences.
            image_embeddings: Image embeddings to merge into text embeddings,
                one per device. Can be empty tensors for text-only inputs.

        Returns:
            A tuple containing the output logits for the last token positions.
        """
        # Get embeddings.
        h = self.embed_tokens(tokens, signal_buffers)

        # Merge image embeddings into text embeddings.
        # Let the kernel handle the no-image embeddings case.
        # And use the first device's image embeddings since they're replicated.
        h0_merged = merge_multimodal_embeddings(
            inputs_embeds=h[0],
            multimodal_embeddings=image_embeddings[0],
            image_token_indices=image_token_indices,
        )

        # Distribute merged embeddings to all devices
        h = distribute_value(h0_merged, self.devices)

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
    """Implements patch embeddings for the InternVL vision model.

    This module converts input images into patch embeddings by:
    - Dividing images into fixed-size patches
    - Projecting each patch to an embedding vector
    - Adding learnable class and position embeddings

    The module supports sharding across multiple devices for distributed execution.
    """

    def __init__(
        self, config: InternVLConfig, device: DeviceRef | None = None
    ) -> None:
        """Initializes the vision embeddings module.

        Args:
            config: The InternVL configuration containing vision model parameters.
            device: The device to place weights on. Defaults to CPU if not specified.
        """
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
        """Gets position embeddings, interpolating if needed for different resolutions.

        Args:
            H: Height in patches (can be int or symbolic Dim).
            W: Width in patches (can be int or symbolic Dim).

        Returns:
            Position embeddings tensor of shape [1, H*W+1, embed_dim].
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
        """Computes embeddings for input pixel values.

        Args:
            pixel_values: Input tensor of pre-extracted patches of shape
                         (batch_size, num_patches_h, num_patches_w, channels, patch_size, patch_size).

        Returns:
            Embeddings tensor of shape (batch_size, num_positions, embed_dim).
        """
        # Extract dimensions from input shape.
        (
            batch_size,
            num_patches_h,
            num_patches_w,
            channels,
            patch_size_h,
            patch_size_w,
        ) = pixel_values.shape
        assert channels == 3
        assert patch_size_h == self.patch_size
        assert patch_size_w == self.patch_size

        # Check that we have static dimensions for height and width.
        if not isinstance(num_patches_h, StaticDim) or not isinstance(
            num_patches_w, StaticDim
        ):
            raise ValueError(
                f"InternVisionEmbeddings requires static image dimensions, "
                f"got {num_patches_h=}, {num_patches_w=}"
            )

        # Reshape pre-extracted patches to (batch_size, num_patches, channels * patch_size * patch_size).
        # The patches are already extracted by the tokenizer, so we just need to reshape them.
        pixel_values = ops.reshape(
            pixel_values,
            [
                batch_size,
                num_patches_h * num_patches_w,
                channels * self.patch_size * self.patch_size,
            ],
        )

        # Apply linear transformation directly
        pixel_values = pixel_values.cast(self.patch_embedding.weight.dtype)
        patch_embeds = self.patch_embedding(pixel_values)

        # 3. Add class token
        class_embeds = self.class_embedding.broadcast_to(
            (batch_size, 1, self.embed_dim)
        )
        embeddings = ops.concat([class_embeds, patch_embeds], axis=1)

        # 4. Add position embeddings.
        position_embedding = self._get_position_embedding(
            num_patches_h, num_patches_w
        )
        embeddings = embeddings + position_embedding

        return embeddings


class InternVLVisionMLP(Module, Shardable):
    """Multi-layer perceptron (MLP) for the InternVL vision model.

    A simple 2-layer feed-forward network used within the
    vision transformer encoder layers. The MLP consists of:

    - First linear projection expanding hidden size to intermediate size
    - GELU activation function
    - Second linear projection back to hidden size

    This follows the standard transformer FFN architecture but uses GELU
    activation. Supports tensor parallelism through the Shardable interface.
    """

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        dtype: DType,
        device: DeviceRef,
        has_bias: bool = True,
    ) -> None:
        super().__init__()
        self.fc1 = Linear(
            in_dim=hidden_size,
            out_dim=intermediate_size,
            dtype=dtype,
            device=device,
            has_bias=has_bias,
        )
        self.fc2 = Linear(
            in_dim=intermediate_size,
            out_dim=hidden_size,
            dtype=dtype,
            device=device,
            has_bias=has_bias,
        )

    @property
    def sharding_strategy(self) -> ShardingStrategy | None:
        """Get the MLP sharding strategy."""
        return self.fc1.sharding_strategy

    @sharding_strategy.setter
    def sharding_strategy(self, strategy: ShardingStrategy) -> None:
        """Set the sharding strategy for the MLP layers.

        Args:
            strategy: The sharding strategy to apply.
        """
        if strategy.is_replicate:
            # For replicate strategy, both layers use the same strategy
            self.fc1.sharding_strategy = strategy
            self.fc2.sharding_strategy = strategy
        else:
            # For tensor parallel: fc1 expands (rowwise), fc2 reduces (columnwise)
            self.fc1.sharding_strategy = ShardingStrategy.rowwise(
                strategy.num_devices
            )
            self.fc2.sharding_strategy = ShardingStrategy.columnwise(
                strategy.num_devices
            )

    def shard(self, shard_idx: int, device: DeviceRef) -> InternVLVisionMLP:
        """Creates a sharded view of this MLP for a specific device.

        Args:
            shard_idx: The index of the shard (0 to num_devices-1).
            device: The device where this shard should reside.

        Returns:
            A sharded InternVLVisionMLP instance.
        """
        # Create new MLP instance with the same configuration
        sharded = InternVLVisionMLP(
            hidden_size=int(self.fc1.weight.shape[1]),  # in_dim
            intermediate_size=int(self.fc1.weight.shape[0]),  # out_dim
            dtype=self.fc1.weight.dtype,
            device=device,
            has_bias=self.fc1.bias is not None,
        )

        # Shard the linear layers
        sharded.fc1 = self.fc1.shard(shard_idx, device)
        sharded.fc2 = self.fc2.shard(shard_idx, device)

        return sharded

    def __call__(self, x: TensorValue) -> TensorValue:
        """Simple forward: fc1 -> GELU -> fc2"""
        x = self.fc1(x)
        x = ops.gelu(x)
        x = self.fc2(x)
        return x


class InternVLMLP1(Module, Shardable):
    """Multimodal projector (mlp1) for InternVL vision model.

    This module projects vision features from the vision encoder to the
    dimensionality expected by the language model. It consists of:
    - Layer normalization
    - Two linear layers with GELU activation

    The module supports sharding across multiple devices for distributed execution.
    """

    def __init__(
        self, config: InternVLConfig, device: DeviceRef | None = None
    ) -> None:
        """Initializes the multimodal projector.

        Args:
            config: The InternVL configuration containing model parameters.
            device: The device to place weights on. Defaults to CPU if not specified.
        """
        super().__init__()
        self.config = config
        vit_hidden_size = config.vision_config.hidden_size
        llm_hidden_size = config.llm_config.hidden_size
        downsample_ratio = config.downsample_ratio

        # Use CPU as default if no device specified.
        device = device or DeviceRef.CPU()

        # Calculate input size after pixel shuffle.
        mlp_input_size = int(vit_hidden_size * (1 / downsample_ratio) ** 2)

        self.layer_norm = LayerNorm(
            dims=mlp_input_size,
            device=device,
            dtype=config.vision_config.dtype,
            eps=1e-6,
            use_bias=True,
        )

        self.fc1 = Linear(
            in_dim=mlp_input_size,
            out_dim=llm_hidden_size,
            dtype=config.vision_config.dtype,
            device=device,
            has_bias=True,
        )

        self.fc2 = Linear(
            in_dim=llm_hidden_size,
            out_dim=llm_hidden_size,
            dtype=config.vision_config.dtype,
            device=device,
            has_bias=True,
        )

    @property
    def sharding_strategy(self) -> ShardingStrategy | None:
        """Get the MLP sharding strategy."""
        return self.fc1.sharding_strategy

    @sharding_strategy.setter
    def sharding_strategy(self, strategy: ShardingStrategy) -> None:
        """Set the sharding strategy for the MLP layers.

        Args:
            strategy: The strategy describing the MLP's sharding.
        """
        # For tensor parallel mlp1:
        # - LayerNorm operates on replicated inputs, so weights are replicated
        # - fc1 expands (rowwise sharding)
        # - fc2 reduces (columnwise sharding)

        # Set sharding strategy for all weights
        # LayerNorm weights should be replicated since inputs are replicated
        self.layer_norm.weight.sharding_strategy = ShardingStrategy.replicate(
            strategy.num_devices
        )
        if self.layer_norm.bias is not None:
            self.layer_norm.bias.sharding_strategy = ShardingStrategy.replicate(
                strategy.num_devices
            )

        # Linear layer weights use tensor parallel sharding
        self.fc1.sharding_strategy = ShardingStrategy.rowwise(
            strategy.num_devices
        )
        self.fc2.sharding_strategy = ShardingStrategy.columnwise(
            strategy.num_devices
        )

    def shard(self, shard_idx: int, device: DeviceRef) -> InternVLMLP1:
        """Creates a sharded view of this MLP for a specific device.

        Args:
            shard_idx: The index of the shard (0 to num_devices-1).
            device: The device where this shard should reside.

        Returns:
            A sharded InternVLMLP1 instance.
        """
        # This should be set unconditionally in the constructor.
        assert self.sharding_strategy

        # Create the new sharded MLP.
        sharded = InternVLMLP1(self.config, device)

        # Shard the weights.
        sharded.layer_norm.weight = self.layer_norm.weight.shard(
            shard_idx, device
        )
        if self.layer_norm.bias is not None:
            sharded.layer_norm.bias = self.layer_norm.bias.shard(
                shard_idx, device
            )

        sharded.fc1 = self.fc1.shard(shard_idx, device)
        sharded.fc2 = self.fc2.shard(shard_idx, device)

        return sharded

    def __call__(self, x: TensorValue) -> TensorValue:
        """Applies the multimodal projection to input embeddings.

        Args:
            x: Input tensor of shape [sequence_length, input_dim].

        Returns:
            Projected tensor of shape [sequence_length, llm_hidden_size].
        """
        x = self.layer_norm(x)
        x = self.fc1(x)
        x = ops.gelu(x)
        x = self.fc2(x)
        return x


class InternVisionEncoderLayer(Module):
    """Single encoder layer for the InternVL vision transformer.

    Each encoder layer implements the standard transformer architecture with
    some InternVL-specific modifications:

    - Multi-head self-attention with optional QK normalization
    - Layer scaling (learnable per-layer scaling factors)
    - Feed-forward network (MLP) with GELU activation
    - Pre-normalization using either LayerNorm or RMSNorm
    - Residual connections around both attention and MLP blocks

    This layer supports tensor parallelism through the Shardable interface.
    """

    def __init__(self, config: InternVLConfig) -> None:
        """Initializes an encoder layer.

        Args:
            config: The InternVL configuration containing model parameters.
        """
        super().__init__()
        self.config = config
        self.devices = config.devices
        self.embed_dim = config.vision_config.hidden_size
        self.intermediary_size = config.vision_config.intermediate_size
        self.norm_type = config.vision_config.norm_type

        # Use custom simple MLP instead of SwiGLU MLP
        # For sharding, we'll create per-device instances later
        default_device = self.devices[0] if self.devices else DeviceRef.CPU()
        self.mlp = InternVLVisionMLP(
            hidden_size=self.embed_dim,
            intermediate_size=self.intermediary_size,
            dtype=config.llm_config.dtype,
            device=default_device,
            has_bias=True,
        )

        layer_norm_eps = config.vision_config.layer_norm_eps

        self.norm1: RMSNorm | LayerNorm
        self.norm2: RMSNorm | LayerNorm
        if self.norm_type == "rms_norm":
            self.norm1 = RMSNorm(
                dim=self.embed_dim,
                dtype=config.llm_config.dtype,
                eps=layer_norm_eps,
                multiply_before_cast=False,  # Match PyTorch behavior: cast first, then multiply
            )
            self.norm2 = RMSNorm(
                dim=self.embed_dim,
                dtype=config.llm_config.dtype,
                eps=layer_norm_eps,
                multiply_before_cast=False,  # Match PyTorch behavior: cast first, then multiply
            )
        else:  # layer_norm
            self.norm1 = LayerNorm(
                dims=self.embed_dim,
                device=default_device,
                dtype=config.llm_config.dtype,
                eps=layer_norm_eps,
                use_bias=True,
            )
            self.norm2 = LayerNorm(
                dims=self.embed_dim,
                device=default_device,
                dtype=config.llm_config.dtype,
                eps=layer_norm_eps,
                use_bias=True,
            )

        self.ls1 = Weight(
            "ls1",
            config.llm_config.dtype,
            [self.embed_dim],
            device=DeviceRef.CPU(),
        )
        self.ls2 = Weight(
            "ls2",
            config.llm_config.dtype,
            [self.embed_dim],
            device=DeviceRef.CPU(),
        )

        # Use InternVL-specific attention with QK normalization.
        vision_config = config.vision_config
        # Create attention on first device: sharding handles distributing it.
        head_dim = (
            vision_config.hidden_size // vision_config.num_attention_heads
        )
        self.attn = InternVLMultiheadAttention(
            num_attention_heads=vision_config.num_attention_heads,
            hidden_size=vision_config.hidden_size,
            head_dim=head_dim,
            devices=[default_device],
            dtype=config.llm_config.dtype,
            qk_normalization=vision_config.qk_normalization,
            layer_norm_eps=vision_config.layer_norm_eps,
            qkv_has_bias=vision_config.qkv_bias,
            o_proj_has_bias=vision_config.o_proj_bias,
            stacked_qkv=True,
        )

        # Set attention sharding strategy for tensor parallelism
        # Use stacked_qkv strategy with head information for proper sharding
        self.attn.sharding_strategy = ShardingStrategy.stacked_qkv(
            len(self.devices),
            num_heads=vision_config.num_attention_heads,
            head_dim=head_dim,
        )

        # Create per-device attention instances using the shard method
        self.attn_per_device = [
            self.attn.shard(n, device) for n, device in enumerate(self.devices)
        ]

        # Set sharding strategies for weights
        self.norm1.weight.sharding_strategy = ShardingStrategy.replicate(
            len(self.devices)
        )
        self.norm2.weight.sharding_strategy = ShardingStrategy.replicate(
            len(self.devices)
        )
        if hasattr(self.norm1, "bias") and self.norm1.bias is not None:
            self.norm1.bias.sharding_strategy = ShardingStrategy.replicate(
                len(self.devices)
            )
        if hasattr(self.norm2, "bias") and self.norm2.bias is not None:
            self.norm2.bias.sharding_strategy = ShardingStrategy.replicate(
                len(self.devices)
            )

        # Set MLP sharding strategies for tensor parallelism
        self.mlp.fc1.sharding_strategy = ShardingStrategy.rowwise(
            len(self.devices)
        )
        self.mlp.fc2.sharding_strategy = ShardingStrategy.columnwise(
            len(self.devices)
        )

        # Create per-device norm instances.
        self.norm1_per_device: list[RMSNorm | LayerNorm] = []
        self.norm2_per_device: list[RMSNorm | LayerNorm] = []
        for n, device in enumerate(self.devices):
            norm1_copy: RMSNorm | LayerNorm
            norm2_copy: RMSNorm | LayerNorm
            if self.norm_type == "rms_norm":
                norm1_copy = RMSNorm(
                    dim=self.embed_dim,
                    dtype=config.llm_config.dtype,
                    eps=layer_norm_eps,
                    multiply_before_cast=False,
                )
                norm2_copy = RMSNorm(
                    dim=self.embed_dim,
                    dtype=config.llm_config.dtype,
                    eps=layer_norm_eps,
                    multiply_before_cast=False,
                )
            else:
                norm1_copy = LayerNorm(
                    dims=self.embed_dim,
                    device=device,
                    dtype=config.llm_config.dtype,
                    eps=layer_norm_eps,
                    use_bias=True,
                )
                norm2_copy = LayerNorm(
                    dims=self.embed_dim,
                    device=device,
                    dtype=config.llm_config.dtype,
                    eps=layer_norm_eps,
                    use_bias=True,
                )

            # Shard weights to device.
            norm1_copy.weight = self.norm1.weight.shard(n, device)
            norm2_copy.weight = self.norm2.weight.shard(n, device)
            if (
                isinstance(norm1_copy, LayerNorm)
                and isinstance(self.norm1, LayerNorm)
                and self.norm1.bias is not None
            ):
                norm1_copy.bias = self.norm1.bias.shard(n, device)
            if (
                isinstance(norm2_copy, LayerNorm)
                and isinstance(self.norm2, LayerNorm)
                and self.norm2.bias is not None
            ):
                norm2_copy.bias = self.norm2.bias.shard(n, device)

            self.norm1_per_device.append(norm1_copy)
            self.norm2_per_device.append(norm2_copy)

        # Create per-device MLP instances.
        self.mlp_per_device = []
        for n, device in enumerate(self.devices):
            mlp_copy = InternVLVisionMLP(
                hidden_size=self.embed_dim,
                intermediate_size=self.intermediary_size,
                dtype=config.llm_config.dtype,
                device=device,
                has_bias=True,
            )
            # Shard the linear layers
            mlp_copy.fc1 = self.mlp.fc1.shard(n, device)
            mlp_copy.fc2 = self.mlp.fc2.shard(n, device)
            self.mlp_per_device.append(mlp_copy)

        # Create allreduce for tensor parallel attention and MLP
        self.allreduce = Allreduce(num_accelerators=len(self.devices))

    def _device_attention_params(
        self, *, device_idx: int
    ) -> DeviceAttentionParams:
        """Get attention parameters for a specific device.

        Returns:
            DeviceAttentionParams with device-specific attention parameters
        """
        num_heads = self.config.vision_config.num_attention_heads
        device_heads = compute_heads_per_device(
            total_heads=num_heads,
            device_idx=device_idx,
            num_devices=len(self.devices),
        )
        head_start, _ = _compute_shard_range(
            num_heads, device_idx, len(self.devices)
        )
        head_dim = self.embed_dim // num_heads
        return DeviceAttentionParams(device_heads, head_start, head_dim)

    def _compute_flash_attention(
        self,
        q: TensorValue,
        k: TensorValue,
        v: TensorValue,
        device_heads: int,
        head_dim: int,
    ) -> TensorValue:
        """Compute flash attention with proper reshaping.

        Args:
            q, k, v: Query, key, value tensors
            device_heads: Number of heads for this device
            head_dim: Dimension per head

        Returns:
            Attention output tensor
        """
        batch_size, seq_len = q.shape[0], q.shape[1]

        # Reshape for multi-head attention
        q = q.reshape((batch_size, seq_len, device_heads, head_dim))
        k = k.reshape((batch_size, seq_len, device_heads, head_dim))
        v = v.reshape((batch_size, seq_len, device_heads, head_dim))

        # Apply flash attention
        attn_out = flash_attention_gpu(
            q,
            k,
            v,
            mask_variant=MHAMaskVariant.NULL_MASK,
            scale=1.0 / math.sqrt(head_dim),
        )

        # Reshape back to partial embedding dimension
        return attn_out.reshape((batch_size, seq_len, device_heads * head_dim))

    def _layer_scaling(
        self,
        tensors: Sequence[TensorValue],
        scale_weight: TensorValue,
        devices: Sequence[DeviceRef],
    ) -> list[TensorValue]:
        """Apply layer scaling to tensors on their respective devices."""
        scale_tensors = [scale_weight.to(device) for device in devices]
        return [t * s for t, s in zip(tensors, scale_tensors)]

    def _split_qkv_per_device(
        self, qkv_outs: Sequence[TensorValue]
    ) -> tuple[list[TensorValue], list[TensorValue], list[TensorValue]]:
        """Split QKV tensors into Q, K, V components per device.

        Returns:
            Tuple of (q_partials, k_partials, v_partials)
        """
        q_partials, k_partials, v_partials = [], [], []

        for i, qkv in enumerate(qkv_outs):
            params = self._device_attention_params(device_idx=i)

            # Calculate sizes for Q, K, V based on device heads
            split_size = params.device_heads * params.head_dim

            # Split into Q, K, V (partial embeddings)
            q, k, v = ops.split(
                qkv, [split_size, split_size, split_size], axis=-1
            )
            q_partials.append(q)
            k_partials.append(k)
            v_partials.append(v)

        return q_partials, k_partials, v_partials

    def __call__(
        self, xs: Sequence[TensorValue], signal_buffers: Sequence[BufferValue]
    ) -> list[TensorValue]:
        """Process input through the encoder layer.

        Args:
            xs: The input hidden states, one per device.
            signal_buffers: Communication buffers for distributed execution.

        Returns:
            The output hidden states after attention and MLP, one per device.
        """
        # Store original inputs for residual connections
        original_hidden_states = xs

        # 1. Apply first normalization per device (using per-device instances)
        # x: [1025, 3200]
        norm1_outs = [norm(x) for x, norm in zip(xs, self.norm1_per_device)]

        # 2. Apply QKV projection per device (rowwise)
        qkv_outs = []
        for i, norm_out in enumerate(norm1_outs):
            attn = self.attn_per_device[i]
            # Rowwise QKV projection - each device computes partial QKV
            qkv = norm_out @ attn.wqkv.T
            if attn.wqkv_bias is not None:
                qkv += attn.wqkv_bias
            qkv_outs.append(qkv)

        # Split QKV into components per device
        q_partials, k_partials, v_partials = self._split_qkv_per_device(
            qkv_outs
        )

        # Handle QK normalization case
        if self.config.vision_config.qk_normalization:
            # Allgather Q and K only (not V) for QK normalization
            q_complete = allgather(q_partials, axis=-1)
            k_complete = allgather(k_partials, axis=-1)
            # V stays partial - no need to allgather

            # Process attention with QK normalization
            attn_outs = []
            batch_size, seq_len, _ = q_complete[0].shape
            num_heads = self.config.vision_config.num_attention_heads
            head_dim = self.embed_dim // num_heads

            for i in range(len(self.devices)):
                params = self._device_attention_params(device_idx=i)
                device_heads, head_start, head_dim = (
                    params.device_heads,
                    params.head_start,
                    params.head_dim,
                )

                # Apply QK normalization on complete Q and K.
                q_normalized = self.attn.q_norm(q_complete[i])
                k_normalized = self.attn.k_norm(k_complete[i])

                # Reshape and slice to get device-specific heads
                q_normalized = q_normalized.reshape(
                    (batch_size, seq_len, num_heads, head_dim)
                )
                k_normalized = k_normalized.reshape(
                    (batch_size, seq_len, num_heads, head_dim)
                )

                q_device = q_normalized[
                    :, :, head_start : head_start + device_heads, :
                ]
                k_device = k_normalized[
                    :, :, head_start : head_start + device_heads, :
                ]

                # Get partial V for this device
                v_device = v_partials[i]

                # Compute attention (handles reshaping internally)
                attn_out = self._compute_flash_attention(
                    q_device.reshape((batch_size, seq_len, -1)),
                    k_device.reshape((batch_size, seq_len, -1)),
                    v_device,
                    device_heads,
                    head_dim,
                )

                # Apply output projection (columnwise) on device-specific portion
                attn_out = self.attn_per_device[i].o_proj(attn_out)

                attn_outs.append(attn_out)
        else:
            # No QK normalization - process with partial embeddings directly
            attn_outs = []
            for i in range(len(self.devices)):
                params = self._device_attention_params(device_idx=i)

                # Compute attention with partial Q, K, V
                attn_out = self._compute_flash_attention(
                    q_partials[i],
                    k_partials[i],
                    v_partials[i],
                    params.device_heads,
                    params.head_dim,
                )

                # Apply output projection (columnwise)
                attn_out = self.attn_per_device[i].o_proj(attn_out)
                attn_outs.append(attn_out)

        # Allreduce output projection results
        attn_outs = self.allreduce(attn_outs, signal_buffers)

        # 3. Apply layer scaling and first residual
        attn_outs_scaled = self._layer_scaling(
            attn_outs, self.ls1, [x.device for x in xs]
        )

        # First residual connection
        hidden_states = [
            out + orig
            for out, orig in zip(attn_outs_scaled, original_hidden_states)
        ]

        # 4. Apply second normalization per device
        norm2_outs = [
            norm(hidden)
            for norm, hidden in zip(self.norm2_per_device, hidden_states)
        ]

        # 5. Apply MLP with reshaping using per-device MLPs
        mlp_outs = []
        for mlp, norm_out in zip(self.mlp_per_device, norm2_outs):
            batch_size, seq_len, hidden_dim = norm_out.shape
            # Reshape to 2D for MLP
            mlp_out_2d = mlp(
                norm_out.reshape((batch_size * seq_len, hidden_dim))
            )
            # Reshape back to 3D
            mlp_outs.append(
                mlp_out_2d.reshape((batch_size, seq_len, hidden_dim))
            )

        # Apply allreduce for tensor parallel MLP
        mlp_outs = self.allreduce(mlp_outs, signal_buffers)

        # 6. Apply second layer scaling and residual
        mlp_outs_scaled = self._layer_scaling(
            mlp_outs, self.ls2, [x.device for x in mlp_outs]
        )

        # Second residual connection
        outputs = [
            out + hidden for out, hidden in zip(mlp_outs_scaled, hidden_states)
        ]

        return outputs


class InternVLVisionModel(Module):
    """Vision transformer model for processing images in InternVL.

    This implements the vision encoder component of InternVL, which processes
    input images and produces visual embeddings that can be consumed by the
    language model. The architecture follows a standard Vision Transformer (ViT)
    design with InternVL-specific enhancements:

    Key components:
    - Patch embedding layer that converts image patches to embeddings
    - Positional embeddings with interpolation support for varying resolutions
    - Stack of transformer encoder layers
    - Optional final layer normalization (when not using mean pooling, occurs in larger variants)

    Note:
        Currently limited to single-device execution for the vision component,
        multi-gpu coming shortly.
    """

    def __init__(self, config: InternVLConfig) -> None:
        super().__init__()
        self.config = config
        self.devices = config.devices

        default_device = (
            config.llm_config.devices[0]
            if config.llm_config.devices
            else DeviceRef.CPU()
        )

        self.embeddings = InternVisionEmbeddings(config)
        self.embeddings.sharding_strategy = ShardingStrategy.replicate(
            len(config.devices)
        )
        self.embeddings_list = [
            self.embeddings.shard(n, dev)
            for n, dev in enumerate(config.devices)
        ]

        # Store downsample_ratio and ps_version for pixel_shuffle
        self.downsample_ratio = config.downsample_ratio
        self.ps_version: str = getattr(config, "ps_version", "v2")

        if config.downsample_ratio != 0.5:
            raise ValueError(
                "InternVLVisionModel only supports downsample ratio of 0.5"
            )

        if self.ps_version != "v2":
            raise ValueError("InternVLVisionModel only supports ps_version v2")

        # Create encoder layers
        self.encoder_layers = LayerList(
            [
                InternVisionEncoderLayer(config)
                for _ in range(config.vision_config.num_hidden_layers)
            ]
        )

        # Add final layer normalization to match PyTorch reference implementation
        # In PyTorch: self.layernorm = nn.Identity() if config.use_mean_pooling else nn.LayerNorm(...)
        vision_config = config.vision_config

        # Only create layernorm if use_mean_pooling is False
        self.layernorm: RMSNorm | LayerNorm | None
        if not getattr(vision_config, "use_mean_pooling", False):
            if vision_config.norm_type == "rms_norm":
                self.layernorm = RMSNorm(
                    dim=vision_config.hidden_size,
                    dtype=config.llm_config.dtype,
                    eps=vision_config.layer_norm_eps,
                    multiply_before_cast=False,  # Match PyTorch behavior: cast first, then multiply
                )
            else:  # layer_norm
                self.layernorm = LayerNorm(
                    dims=vision_config.hidden_size,
                    device=default_device,
                    dtype=config.llm_config.dtype,
                    eps=vision_config.layer_norm_eps,
                )

            # Set replicate sharding strategy for layernorm
            self.layernorm.weight.sharding_strategy = (
                ShardingStrategy.replicate(len(self.devices))
            )
            if (
                hasattr(self.layernorm, "bias")
                and self.layernorm.bias is not None
            ):
                self.layernorm.bias.sharding_strategy = (
                    ShardingStrategy.replicate(len(self.devices))
                )

            # Create per-device layernorm instances
            self.layernorm_per_device: list[RMSNorm | LayerNorm] = []
            for n, device in enumerate(self.devices):
                ln_copy: RMSNorm | LayerNorm
                if vision_config.norm_type == "rms_norm":
                    ln_copy = RMSNorm(
                        dim=vision_config.hidden_size,
                        dtype=config.llm_config.dtype,
                        eps=vision_config.layer_norm_eps,
                        multiply_before_cast=False,
                    )
                else:
                    ln_copy = LayerNorm(
                        dims=vision_config.hidden_size,
                        device=device,
                        dtype=config.llm_config.dtype,
                        eps=vision_config.layer_norm_eps,
                    )
                ln_copy.weight = self.layernorm.weight.shard(n, device)
                if (
                    isinstance(ln_copy, LayerNorm)
                    and isinstance(self.layernorm, LayerNorm)
                    and self.layernorm.bias is not None
                ):
                    ln_copy.bias = self.layernorm.bias.shard(n, device)
                self.layernorm_per_device.append(ln_copy)
        else:
            # Use identity (no-op) when mean pooling is used
            self.layernorm = None
            self.layernorm_per_device = []

        # Initialize the multimodal projector (mlp1).
        self.mlp1 = InternVLMLP1(config)
        # Set tensor parallel sharding strategy for mlp1
        # This will properly shard fc1/fc2 while keeping layer_norm replicated
        self.mlp1.sharding_strategy = ShardingStrategy.rowwise(
            len(config.devices)
        )

        # Create sharded mlp1 instances for each device.
        self.mlp1_list = [
            self.mlp1.shard(n, dev) for n, dev in enumerate(config.devices)
        ]

        # Create allreduce for tensor parallel MLP1.
        self.allreduce = Allreduce(num_accelerators=len(self.devices))

    def pixel_shuffle(self, x: TensorValue, h: int, w: int) -> TensorValue:
        """Pixel shuffle operation for downsampling vision features.

        Args:
            x: Input tensor of shape [batch, height, width, channels]
            h: Height dimension (as int, not Dim)
            w: Width dimension (as int, not Dim)

        NOTE: this assumes a downsampling factor of 2x (`scale_factor == 0.5`).

        Returns:
            Shuffled tensor
        """
        # The constructor checked this is v2 already.
        assert self.ps_version != "v1"

        batch_size = x.shape[0]
        c = x.shape[3]

        # Common case: downsample by 2x.
        # [N, H, W, C] -> [N, H, W/2, C*2].
        x = ops.reshape(x, [batch_size, h, w // 2, -1])

        # Permute: [N, H, W/2, C*2] -> [N, W/2, H, C*2].
        x = ops.permute(x, [0, 2, 1, 3])

        # Reshape: [N, W/2, H, C*2] -> [N, H/2, W/2, C*4].
        x = ops.reshape(x, [batch_size, h // 2, w // 2, -1])

        # For ps_version v2, do another permute.
        x = ops.permute(x, [0, 2, 1, 3])

        return x

    def __call__(
        self,
        pixel_values: Sequence[TensorValue],
        signal_buffers: Sequence[BufferValue],
    ) -> Sequence[TensorValue]:
        """Process pixel values to image embeddings.

        Args:
            pixel_values: Input pixel values tensor, one per device.
            signal_buffers: Communication buffers for distributed execution.

        Returns:
            Image embeddings tensor, one per device, flattened for language model.
        """
        # Get vision embeddings from each device
        vit_embeds = [
            embed(pixels)
            for embed, pixels in zip(self.embeddings_list, pixel_values)
        ]

        # Pass through encoder layers on all devices
        hidden_states_list = vit_embeds
        for encoder_layer in self.encoder_layers:
            # Process the list of tensors through the encoder layer
            hidden_states_list = encoder_layer(
                hidden_states_list, signal_buffers
            )

        # Apply layernorm if it exists (when use_mean_pooling is False)
        if self.layernorm is not None:
            vit_embeds_processed = [
                self.layernorm_per_device[i](hidden_states_list[i])
                for i in range(len(hidden_states_list))
            ]
        else:
            vit_embeds_processed = hidden_states_list

        # Remove CLS token (first token) from each device's embeddings.
        # Shape: [batch, num_positions, embed_dim] -> [batch, num_positions-1, embed_dim]
        vit_embeds_no_cls = [
            embeds[:, 1:, :] for embeds in vit_embeds_processed
        ]

        # For spatial operations, we need to know the grid size
        # Calculate from the number of patches (assuming square images)
        # Use the first device's shape as reference.
        batch_size, num_patches, embed_dim = vit_embeds_no_cls[0].shape[:3]

        # For static shapes, compute grid dimensions.
        # This assumes num_patches is a perfect square.
        h = int(int(num_patches) ** 0.5)
        w = int(int(num_patches) ** 0.5)

        # Reshape to spatial format: [batch, h*w, embed_dim] -> [batch, h, w, embed_dim]
        spatial_embeds = [
            ops.reshape(embeds, [batch_size, h, w, embed_dim])
            for embeds in vit_embeds_no_cls
        ]

        # Apply pixel shuffle for downsampling
        shuffled_embeds = [
            self.pixel_shuffle(embeds, h, w) for embeds in spatial_embeds
        ]

        # Reshape back to sequence format
        # After pixel shuffle with scale=0.5, dimensions are halved
        new_h = h // 2
        new_w = w // 2
        new_embed_dim = int(embed_dim * 4)  # C becomes C*4 after pixel shuffle

        seq_embeds = [
            ops.reshape(embeds, [batch_size, new_h * new_w, new_embed_dim])
            for embeds in shuffled_embeds
        ]

        # Apply mlp1 projection (includes layer norm, fc1, gelu, fc2).
        mlp_out = [
            mlp(embeds) for mlp, embeds in zip(self.mlp1_list, seq_embeds)
        ]
        mlp_out = self.allreduce(mlp_out, signal_buffers)

        # Flatten for language model
        # Shape: [batch, seq_len, hidden_dim] -> [batch * seq_len, hidden_dim]
        flattened = [
            ops.reshape(
                embeds,
                [
                    batch_size * new_h * new_w,
                    self.config.llm_config.hidden_size,
                ],
            )
            for embeds in mlp_out
        ]

        # Return the flattened embeddings for all devices
        return flattened
