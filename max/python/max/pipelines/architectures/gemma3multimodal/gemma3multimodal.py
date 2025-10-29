from __future__ import annotations
from typing import Iterable, Literal
from collections.abc import Sequence
import functools

from .model_config import Gemma3ForConditionalGenerationConfig
from .image_processing import Gemma3ImageProcessor

from max.dtype import DType

from max.graph import BufferValue, ShardingStrategy, TensorValue, ops, DeviceRef, Weight, Dim, StaticDim
from max.graph.ops.resize import InterpolationMode

from max.nn import MLP, LayerList, LayerNorm, Linear
from max.nn.kv_cache import PagedCacheValues
from max.nn import ColumnParallelLinear, MLP, LayerList, Module, ReturnLogits, Linear
from max.nn.rotary_embedding import (
    Llama3RopeScalingParams,
    Llama3RotaryEmbedding,
)
from max.pipelines.architectures.gemma3.layers.attention import Gemma3Attention
from max.pipelines.architectures.gemma3.layers.rms_norm import Gemma3RMSNorm
from max.pipelines.architectures.gemma3.layers.scaled_word_embedding import ScaledWordEmbedding
from max.pipelines.architectures.gemma3.layers.transformer_block import Gemma3TransformerBlock

from .attention import Attention
from .model_config import SiglipVisionConfig

# taken from gemma3
class Gemma3LanguageModel(Module):
    """The Gemma3 Multi-Modal model's text component"""

    def __init__(self, config: Gemma3ForConditionalGenerationConfig) -> None:
        super().__init__()
        text_config = config.text_config
        self.devices = config.devices
        # Use scaling_params for both cases (with and without scaling)
        scaling_params = (
            Llama3RopeScalingParams(
                factor=text_config.rope_scaling.factor,
                low_freq_factor=1e38,  # This degenerates to linear scaling
                high_freq_factor=1e38,
                orig_max_position=text_config.max_position_embeddings,
            )
            if text_config.rope_scaling is not None
            else None
        )

        rope_global = Llama3RotaryEmbedding(
            dim=text_config.hidden_size,
            n_heads=text_config.num_attention_heads,
            theta=text_config.rope_theta,
            max_seq_len=text_config.max_position_embeddings,
            device=config.devices[0],
            head_dim=text_config.head_dim,
            interleaved=False,
            scaling_params=scaling_params,
        )

        # rope_local doesn't use scaling
        rope_local = Llama3RotaryEmbedding(
            dim=text_config.hidden_size,
            n_heads=text_config.num_attention_heads,
            theta=text_config.rope_local_base_freq,
            max_seq_len=text_config.max_position_embeddings,
            device=config.devices[0],
            head_dim=text_config.head_dim,
            interleaved=False,
            scaling_params=None,  # No scaling
        )

        embedding_output_dtype = config.dtype
        if config.float8_config and config.float8_config.embedding_output_dtype:
            embedding_output_dtype = config.float8_config.embedding_output_dtype

        self.embed_tokens = ScaledWordEmbedding(
            text_config.vocab_size,
            text_config.hidden_size,
            embedding_output_dtype,
            config.devices,
            embed_scale=text_config.hidden_size**0.5,
        )

        self.norm = Gemma3RMSNorm(
            text_config.hidden_size,
            DType.bfloat16,
            text_config.rms_norm_eps,
        )
        self.norm.sharding_strategy = ShardingStrategy.replicate(
            len(config.devices)
        )
        self.norm_shards = self.norm.shard(config.devices)

        self.lm_head = ColumnParallelLinear(
            text_config.hidden_size,
            text_config.vocab_size,
            dtype=config.dtype,
            devices=config.devices,
            tied_weight=(
                self.embed_tokens.weight if config.tie_word_embeddings else None
            ),
        )

        create_norm = functools.partial(
            Gemma3RMSNorm,
            text_config.hidden_size,
            DType.bfloat16,
            eps=text_config.rms_norm_eps,
        )

        layers = [
            Gemma3TransformerBlock(
                attention=Gemma3Attention(
                    rope_global=rope_global,
                    rope_local=rope_local,
                    num_attention_heads=text_config.num_attention_heads,
                    num_key_value_heads=text_config.num_key_value_heads,
                    hidden_size=text_config.hidden_size,
                    kv_params=config.kv_params,
                    layer_idx=i,
                    dtype=config.dtype,
                    devices=config.devices,
                    qk_norm_eps=text_config.rms_norm_eps,
                    local_window_size=text_config.sliding_window,
                    float8_config=config.float8_config,
                ),
                mlp=MLP(
                    dtype=config.dtype,
                    quantization_encoding=None,
                    hidden_dim=text_config.hidden_size,
                    feed_forward_length=text_config.intermediate_size,
                    devices=config.devices,
                    activation_function=text_config.hidden_activation,
                    float8_config=config.float8_config,
                ),
                input_layernorm=create_norm(),
                post_attention_layernorm=create_norm(),
                pre_feedforward_layernorm=create_norm(),
                post_feedforward_layernorm=create_norm(),
                devices=config.devices,
            )
            for i in range(text_config.num_hidden_layers)
        ]

        self.dim = text_config.hidden_size
        self.n_heads = text_config.num_attention_heads
        self.layers = LayerList(layers)
        self.norm = self.norm
        self.lm_head = self.lm_head
        self.embed_tokens = self.embed_tokens
        self.kv_params = config.kv_params
        self.return_logits = config.return_logits

    def __call__(
        self,
        tokens: TensorValue,
        signal_buffers: Sequence[BufferValue],
        kv_collections: Sequence[PagedCacheValues],
        return_n_logits: TensorValue,
        input_row_offsets: Sequence[TensorValue],
        image_embeddings: Sequence[TensorValue],
        **kwargs,
    ) -> tuple[TensorValue, ...]:
        h = self.embed_tokens(tokens, signal_buffers)

        # Create KV cache collections per device

        # Run through transformer layers
        for idx, layer in enumerate(self.layers):
            layer_idx_tensor = ops.constant(
                idx, DType.uint32, device=self.devices[0]
            )
            h = layer(
                layer_idx_tensor,
                h,
                signal_buffers,
                kv_collections,
                input_row_offsets=input_row_offsets,
                **kwargs,
            )

        last_token_indices = [offsets[1:] - 1 for offsets in input_row_offsets]
        last_token_h = []
        if h:
            last_token_h = [
                ops.gather(h_device, indices, axis=0)
                for h_device, indices in zip(h, last_token_indices, strict=True)
            ]
        last_logits = ops.cast(
            # Take only the device 0 logits to device-to-host transfer.
            self.lm_head(
                [
                    self.norm_shards[i](last_token_h[i])
                    for i in range(len(last_token_h))
                ],
                signal_buffers,
            )[0],
            DType.float32,
        )

        logits = None
        offsets = None

        if self.return_logits == ReturnLogits.VARIABLE and h:
            # Create range and gather indices for variable logits
            return_range = ops.range(
                start=return_n_logits[0],
                stop=0,
                step=-1,
                out_dim="return_n_logits_range",
                dtype=DType.int64,
                device=self.devices[0],
            )
            last_indices = [
                ops.reshape(
                    ops.unsqueeze(row_offset[1:], -1) - return_range,
                    shape=(-1,),
                )
                for row_offset in input_row_offsets
            ]

            # Gather, normalize, and get logits
            variable_tokens = [
                self.norm_shards[i](ops.gather(h_device, indices, axis=0))
                for i, (h_device, indices) in enumerate(
                    zip(h, last_indices, strict=True)
                )
            ]
            logits = ops.cast(
                self.lm_head(variable_tokens, signal_buffers)[0], DType.float32
            )
            offsets = ops.range(
                0,
                last_indices[0].shape[0] + return_n_logits[0],
                return_n_logits[0],
                out_dim="logit_offsets",
                dtype=DType.int64,
                device=self.devices[0],
            )

        elif self.return_logits == ReturnLogits.ALL and h:
            # Apply normalization to all hidden states and get all logits
            all_normalized = [
                self.norm_shards[i](h_device) for i, h_device in enumerate(h)
            ]
            logits = ops.cast(
                self.lm_head(all_normalized, signal_buffers)[0], DType.float32
            )
            offsets = input_row_offsets[0]

        if offsets is not None:
            assert logits is not None
            return (last_logits, logits, offsets)

        return (last_logits,)

# class Gemma3MLP1/projector:
#   https://entron.github.io/posts/Try-Gemma3-using-Hugging-Face-Part-1/#multi_modal_projector
#   may be useful.  uses a Gemma3RMSNorm layer and more
#   def __init__(Module):
#        super.__init__()
#        mm_input_projection_weight (nn.Paramter)
#        mm_soft_emb_norm = Gemma3RMSNorm(...)
#        patches_per_image
#        tokens_per_side
#        kernel_size
#        avg_pool

#   def forward(vision_outputs):
#       get shape of vision_outputs
#       reshape them with transpose() and reshape() and contiguous()
#       pool the outputs with avg_pool, then flatten() and transpose()
#       run through mm_soft_emb_norm
#       use matmul(normed_vision_outputs, self.mm_input_projection_weight)
#       return
        return projected_vision_outputs.type_as(vision_outputs)


# borrowed from InternVL
class Gemma3VisionEmbeddings:
    """implements patch embeddings as per Siglip (?)"""
    def __init__(self, config: Gemma3ForConditionalGenerationConfig, device: DeviceRef | None = None) -> None:
        """initializes the vision embeddings module"""
        self.config = config
        self.devices = config.devices
        self.embed_dim = config.vision_config.hidden_size
        self.image_size = config.vision_config.image_size
        self.patch_size = config.vision_config.patch_size
        self.dtype = config.dtype

        # Calculate patch dimensions
        # Note: in_dim matches Conv2d flattening order (C*H*W)
        self.patch_embedding = Linear(
            in_dim=3 * self.patch_size * self.patch_size,
            out_dim=self.embed_dim,
            dtype=self.dtype,
            device=device if device else DeviceRef.CPU(),
            has_bias=True,
        )

        self.num_patches = (self.image_size // self.patch_size) ** 2
        self.num_positions = self.num_patches

        self.position_embedding = Weight(
            "position_embedding",
            dtype=self.dtype,
            shape=(1, self.num_positions, self.embed_dim),
            device=device if device else DeviceRef.CPU(),
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
                "only replicate is supported for Gemma3VisionEmbeddings, "
                "currently"
            )

        self.patch_embedding.sharding_strategy = strategy
        self.position_embedding.sharding_strategy = strategy

    def shard(
        self, devices: Iterable[DeviceRef]
    ) -> list[Gemma3VisionEmbeddings]:
        """Creates sharded views of this vision embeddings across multiple devices.

        Args:
            devices: Iterable of devices to place the shards on.

        Returns:
            List of sharded Gemma3VisionEmbeddings instances, one for each device.
        """
        # This should be set unconditionally in the constructor.
        assert self.sharding_strategy

        # Get sharded weights
        patch_embedding_shards = self.patch_embedding.shard(devices)
        position_embedding_shards = self.position_embedding.shard(devices)

        shards = []
        for device, patch_shard, pos_shard in zip(
            devices,
            patch_embedding_shards,
            position_embedding_shards,
            strict=True,
        ):
            # Create the new sharded embedding.
            sharded = Gemma3VisionEmbeddings(self.config, device)

            # Assign the sharded weights.
            sharded.patch_embedding = patch_shard
            sharded.position_embedding = pos_shard

            shards.append(sharded)

        return shards

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
        return patch_pos_embed

    def __call__(self, pixel_values: TensorValue, patch_attention_mask) -> TensorValue:
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
                f"Gemma3VisionEmbeddings requires static image dimensions, "
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

        # Add position embeddings.
        position_embedding = self._get_position_embedding(
            num_patches_h, num_patches_w
        )
        patch_embeds = patch_embeds + position_embedding

        return patch_embeds


class Gemma3VisionEncoderLayer(Module):
    """Single transformer layer in the SigLIP vision encoder."""
    
    def __init__(self, config: Gemma3ForConditionalGenerationConfig):
        vision_config = config.vision_config
        text_config = config.text_config

        self.embed_dim = vision_config.hidden_size
        
        # Pre-attention layer norm
        self.layer_norm1 = LayerNorm(
            self.embed_dim,
            eps=vision_config.layer_norm_eps,
            device=config.devices[0],
            dtype=config.dtype
        )
        
        # Self-attention
        self.self_attn = Attention(
            n_heads=vision_config.num_attention_heads,
            device=config.devices[0],
            dtype=config.dtype,
            dim=vision_config.hidden_size,
            head_dim=text_config.head_dim,
        )
        
        # Pre-MLP layer norm
        self.layer_norm2 = LayerNorm(
            self.embed_dim,
            eps=vision_config.layer_norm_eps,
            device=config.devices[0],
            dtype=config.dtype
        )
        
        # MLP (Feed-Forward Network)
        self.mlp = MLP(
            dtype=config.dtype,
            hidden_dim=vision_config.intermediate_size,
            quantization_encoding=None,
            activation_function="gelu", # it doesn't like vision_config.hidden_act (gelu_pytorch_tanh),
            devices=config.devices,
            feed_forward_length=text_config.intermediate_size, # Size of dimension used to project the inputs TODO ????
            float8_config=config.float8_config,
            has_bias=text_config.attention_bias,
        )
    
    def __call__(
        self,
        hidden_states: TensorValue,
    ) -> TensorValue:
        # Self-attention with residual
        residual = hidden_states
        hidden_states = self.layer_norm1(hidden_states)
        hidden_states = self.self_attn(hidden_states)
        hidden_states = residual + hidden_states
        
        # MLP with residual
        residual = hidden_states
        hidden_states = self.layer_norm2(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        
        return hidden_states


class Gemma3VisionEncoder(Module):
    """SigLIP vision encoder with 27 transformer layers."""
    
    def __init__(self, config: Gemma3ForConditionalGenerationConfig):
        self.layers = LayerList([
            Gemma3VisionEncoderLayer(config)
            for _ in range(config.vision_config.num_hidden_layers)
        ])
    
    def __call__(
        self,
        hidden_states: TensorValue,
    ) -> TensorValue:
        # Pass through all layers
        for layer in self.layers:
            hidden_states = layer(hidden_states)
        return hidden_states


class Gemma3VisionModel(Module):
    def __init__(self, config: Gemma3ForConditionalGenerationConfig) -> None:
        super().__init__()
        self.config = config
        vision_config = config.vision_config
        text_config = config.text_config
        
        # Vision embeddings (you already have this)
        self.embeddings = Gemma3VisionEmbeddings(config)
        
        # Vision encoder (NEW - need to implement)
        self.encoder = Gemma3VisionEncoder(config)
        
        # Post-encoder layer norm (NEW)
        self.post_layernorm = LayerNorm(
            vision_config.hidden_size,
            eps=vision_config.layer_norm_eps,
            device=config.devices[0],
            dtype=config.dtype
        )
        
        # Vision-to-language projection (NEW)
        # Projects from vision hidden_size to language hidden_size
        self.mlp1 = MLP(
            dtype=config.dtype,
            hidden_dim=vision_config.intermediate_size,
            quantization_encoding=None,
            activation_function="gelu", # it doesn't like vision_config.hidden_act (gelu_pytorch_tanh),
            devices=config.devices,
            feed_forward_length=text_config.intermediate_size, # Size of dimension used to project the inputs TODO ????
            float8_config=config.float8_config,
            has_bias=text_config.attention_bias,
        )

    # from huggingface/gemma3/modeling_gemma3.py
    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        inputs_embeds=None,
        cache_position=None,
        position_ids=None,
        pixel_values=None,
        attention_mask=None,
        token_type_ids=None,
        use_cache=True,
        logits_to_keep=None,
        labels=None,
        **kwargs,
    ):
        pass

    def __call__(
        self,
        pixel_values: Sequence[TensorValue],
        signal_buffers: Sequence[BufferValue],
    ) -> Sequence[TensorValue]:
        """Process pixel values to image embeddings"""
        # TODO pinched from idefics3
        # 1. Convert patches to embeddings
        hidden_states = self.embeddings(pixel_values[0])
        # Shape: [batch, num_patches, vision_hidden_size]
        
        # 2. Pass through vision encoder (27 transformer layers)
        hidden_states = self.encoder(hidden_states)
        # Shape: [batch, num_patches, vision_hidden_size]
        
        # 3. Post-encoder normalization
        hidden_states = self.post_layernorm(hidden_states)
        # Shape: [batch, num_patches, vision_hidden_size]
        
        # 4. Project to language model dimension
        image_features = self.mlp1(hidden_states)
        # Shape: [batch, num_patches, language_hidden_size]
        
        # Return one output per device (for multi-GPU)
        return [image_features for _ in range(len(self.config.devices))]