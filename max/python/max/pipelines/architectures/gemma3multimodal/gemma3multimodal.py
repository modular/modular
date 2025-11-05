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

from __future__ import annotations

import functools
import logging
from collections.abc import Iterable, Sequence

from max.driver import Tensor
from max.dtype import DType
from max.experimental.functional import matmul
from max.graph import (
    BufferValue,
    DeviceRef,
    ShardingStrategy,
    TensorValue,
    Weight,
    ops,
)
from max.graph.ops import avg_pool2d
from max.nn import (
    MLP,
    ColumnParallelLinear,
    Conv2d,
    LayerList,
    LayerNorm,
    Linear,
    Module,
    ReturnLogits,
)
from max.nn.embedding import Embedding
from max.nn.kv_cache import PagedCacheValues
from max.nn.rotary_embedding import (
    Llama3RopeScalingParams,
    Llama3RotaryEmbedding,
)
from max.pipelines.architectures.gemma3.layers.attention import Gemma3Attention
from max.pipelines.architectures.gemma3.layers.rms_norm import Gemma3RMSNorm
from max.pipelines.architectures.gemma3.layers.scaled_word_embedding import (
    ScaledWordEmbedding,
)
from max.pipelines.architectures.gemma3.layers.transformer_block import (
    Gemma3TransformerBlock,
)
from max.pipelines.architectures.internvl.embedding_utils import (
    merge_multimodal_embeddings,
)

from .attention import Gemma3VisionAttention
from .model_config import Gemma3ForConditionalGenerationConfig

logger = logging.getLogger("max.pipelines")


# ✅ taken from gemma3
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
        image_token_indices: TensorValue,
        **kwargs,
    ) -> tuple[TensorValue, ...]:
        h = self.embed_tokens(tokens, signal_buffers)

        # Replace image placeholder tokens with vision embeddings
        h = [
            merge_multimodal_embeddings(
                inputs_embeds=h_device,
                multimodal_embeddings=img_embed,
                image_token_indices=img_tok_indices,
            )
            for h_device, img_embed, img_tok_indices in zip(
                h, image_embeddings, image_token_indices, strict=True
            )
        ]

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


# ✅ working, based on HF
class Gemma3MultiModalProjector(Module):
    def __init__(self, config: Gemma3ForConditionalGenerationConfig):
        super().__init__()

        self.devices = config.devices

        self.mm_input_projection_weight = Weight(
            "mm_input_projection_weight",
            dtype=config.dtype,
            shape=(
                config.vision_config.hidden_size,
                config.text_config.hidden_size,
            ),
            device=self.devices[0],
        )

        self.mm_soft_emb_norm = Gemma3RMSNorm(
            config.vision_config.hidden_size,
            eps=config.vision_config.layer_norm_eps,
            dtype=config.dtype,
        )

        self.patches_per_image = int(
            config.vision_config.image_size // config.vision_config.patch_size
        )  # 64

        self.tokens_per_side = int(
            config.mm_tokens_per_image**0.5
        )  # 256 ** 05 = 16
        self.kernel_size = (
            self.patches_per_image // self.tokens_per_side
        )  # 64 / 16 = 4

    # TODO | are these in the right order?  pool -> mm_soft_emb_norm??
    # TODO | i think should be other way based on the vision tower
    def __call__(self, vision_outputs: Tensor):
        batch_size, _, seq_length = (
            vision_outputs.shape
        )  # TensorValue shape: ['batch_size', 4096, 1152]

        reshaped_vision_outputs = vision_outputs.transpose(
            1, 2
        )  # TensorValue shape: ['batch_size', 1152, 4096]

        # TODO not sure if shape is correct.
        reshaped_vision_outputs = reshaped_vision_outputs.reshape(
            (
                batch_size,
                self.patches_per_image,
                self.patches_per_image,
                seq_length,
            )
        )  # TensorValue shape: ['batch_size', 64, 64, 1152]

        normed_vision_outputs = self.mm_soft_emb_norm(
            reshaped_vision_outputs
        )  # TensorValue shape: ['batch_size', 64, 64, 1152]

        # [N, H, W, C]
        pooled_vision_outputs = avg_pool2d(
            input=normed_vision_outputs,
            kernel_size=(self.kernel_size, self.kernel_size),  # (4,4)
            stride=self.kernel_size,  # 4
        )
        pooled_vision_outputs = pooled_vision_outputs.flatten(2)
        pooled_vision_outputs = vision_outputs.transpose(1, 2)

        projected_vision_outputs = matmul(
            normed_vision_outputs, self.mm_input_projection_weight
        )  # ['batch_size', 1152, 4096]
        return projected_vision_outputs  # .type_as(vision_outputs)


# ⚠️ borrowed from InternVL/Idefics
class Gemma3VisionEmbeddings(Module):
    """Implements patch embeddings for SigLIP vision model."""

    def __init__(
        self,
        config: Gemma3ForConditionalGenerationConfig,
        device: DeviceRef | None = None,
    ) -> None:
        """Initializes the vision embeddings module."""
        super().__init__()
        self.config = config
        self.devices = config.devices
        self.num_channels = config.vision_config.num_channels  # 3
        self.embed_dim = config.vision_config.hidden_size  # 1152
        self.image_size = config.vision_config.image_size  # 896
        self.patch_size = config.vision_config.patch_size  # 14
        self.dtype = config.dtype

        # Calculate patch dimensions
        # Note: in_dim matches Conv2d flattening order (C*H*W)
        # self.patch_embedding = Linear(
        #     in_dim=3 * self.patch_size * self.patch_size,
        #     out_dim=self.embed_dim,
        #     dtype=self.dtype,
        #     device=device,
        #     has_bias=True,
        # )
        # TODO above is internvl, below is idefics
        self.patch_embedding = Conv2d(
            in_channels=self.num_channels,
            out_channels=self.embed_dim,  # 1152
            kernel_size=self.patch_size,  # 14
            stride=self.patch_size,  # 14
            padding=0,  # "valid" padding
            has_bias=True,
            dtype=self.dtype,
            device=device,
        )

        self.num_patches = (
            self.image_size // self.patch_size
        ) ** 2  # 4096 = (896 // 14)^2

        self.position_embedding = Embedding(
            vocab_size=self.num_patches,
            hidden_dim=self.embed_dim,
            dtype=self.dtype,
            device=device,
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

    def __call__(self, pixel_values: TensorValue) -> TensorValue:
        """Computes embeddings for input pixel values.

        Args:
            pixel_values: Input tensor of shape (batch_size, channels, height, width).

        Returns:
            Embeddings tensor of shape (batch_size, num_positions, embed_dim).
        """
        logger.info(
            f"*** Gemma3VisionEmbeddings PV shape: {pixel_values.shape}"
        )
        batch_size = pixel_values.shape[0]
        max_im_h = pixel_values.shape[2]
        max_im_w = pixel_values.shape[3]

        # Convert input from NCHW to NHWC format for MAX Conv2d
        # pixel_values: [batch_size, channels, height, width] -> [batch_size, height, width, channels]
        pixel_values_nhwc = ops.permute(pixel_values, [0, 2, 3, 1])

        # Apply patch embedding (Conv2d with stride=patch_size extracts patches)
        # Output will be in NHWC format: [batch_size, out_height, out_width, out_channels]
        patch_embeds_nhwc = self.patch_embedding(pixel_values_nhwc)

        # Convert output back to NCHW format: [batch_size, out_channels, out_height, out_width]
        patch_embeds = ops.permute(patch_embeds_nhwc, [0, 3, 1, 2])

        # Flatten spatial dimensions and transpose to [batch_size, num_patches, embed_dim]
        # patch_embeds shape: [batch_size, embed_dim, num_patches_h, num_patches_w]
        embeddings = ops.flatten(
            patch_embeds, start_dim=2
        )  # [batch_size, embed_dim, num_patches]
        embeddings = ops.transpose(
            embeddings, 1, 2
        )  # [batch_size, num_patches, embed_dim]

        max_nb_patches_h = max_im_h // self.patch_size
        max_nb_patches_w = max_im_w // self.patch_size
        total_patches = max_nb_patches_h * max_nb_patches_w

        # Create position IDs: [0, 1, 2, ..., total_patches-1] for each batch
        # Generate 2D tensor with shape [batch_size, total_patches]
        position_ids = ops.range(
            start=0,
            stop=self.num_patches,
            step=1,
            out_dim=total_patches,
            device=self.config.devices[0],
            dtype=self.config.dtype,
        )  # [total_patches]
        position_ids = ops.unsqueeze(position_ids, 0)  # [1, total_patches]
        position_ids = ops.tile(
            position_ids, [batch_size, 1]
        )  # [batch_size, total_patches]

        # Get position embeddings for the position IDs
        position_embeds = self.position_embedding(
            position_ids
        )  # [batch_size, total_patches, embed_dim]

        # Add position embeddings to patch embeddings
        embeddings = embeddings + position_embeds

        return embeddings


# ✅ working, based on HF
class Gemma3VisionMLP(Module):
    def __init__(self, config: Gemma3ForConditionalGenerationConfig):
        super().__init__()
        self.hidden_size = config.vision_config.hidden_size
        self.intermediate_size = config.vision_config.intermediate_size

        self.fc1 = Linear(
            self.hidden_size,
            self.intermediate_size,
            dtype=config.dtype,
            device=config.devices[0],
            has_bias=False,
        )

        self.fc2 = Linear(
            self.intermediate_size,
            self.hidden_size,
            dtype=config.dtype,
            device=config.devices[0],
            has_bias=False,
        )

    def __call__(self, x: TensorValue):
        x = self.fc1(x)
        x = ops.gelu(x)
        x = self.fc2(x)
        return x


# ✅ based on HF
class Gemma3VisionEncoderLayer(Module):
    def __init__(
        self, config: Gemma3ForConditionalGenerationConfig, layer_idx: int
    ):
        vision_config = config.vision_config

        self.embed_dim = vision_config.hidden_size

        # Pre-attention layer norm ((1152,), eps=1e-06)
        self.layer_norm1 = LayerNorm(
            self.embed_dim,
            eps=vision_config.layer_norm_eps,
            device=config.devices[0],
            dtype=config.dtype,
        )

        # Self-attention
        self.self_attn = Gemma3VisionAttention(
            config=config,
            layer_idx=layer_idx,
        )

        # post-attention layer norm ((1152,), eps=1e-06)
        self.layer_norm2 = LayerNorm(
            self.embed_dim,
            eps=vision_config.layer_norm_eps,
            device=config.devices[0],
            dtype=config.dtype,
        )

        # MLP (Feed-Forward Network) - simple GELUTanhfc1/fc2 style
        self.mlp = Gemma3VisionMLP(config)

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


# ✅ construct lots of EncoderLayers/run through them
class Gemma3VisionEncoder(Module):
    """SigLIP vision encoder with 27 transformer layers."""

    def __init__(self, config: Gemma3ForConditionalGenerationConfig):
        super().__init__()
        self.layers = LayerList(
            [
                Gemma3VisionEncoderLayer(config, layer_idx)
                for layer_idx, _ in enumerate(
                    range(config.vision_config.num_hidden_layers)
                )
            ]
        )

    def __call__(
        self,
        hidden_states: TensorValue,
    ) -> TensorValue:
        # Pass through all layers
        for layer in self.layers:
            hidden_states = layer(hidden_states)
        return hidden_states


# ⚠️ not sure if `prepare_inputs_for_generation` is required
class Gemma3VisionModel(Module):
    def __init__(self, config: Gemma3ForConditionalGenerationConfig) -> None:
        super().__init__()
        self.config = config
        vision_config = config.vision_config

        # Vision embeddings
        self.embeddings = Gemma3VisionEmbeddings(
            config, device=config.devices[0]
        )

        # Vision encoder (27 transformer layers)
        self.encoder = Gemma3VisionEncoder(config)

        # Post-encoder layer norm
        self.post_layernorm = LayerNorm(
            vision_config.hidden_size,
            eps=vision_config.layer_norm_eps,
            device=config.devices[0],
            dtype=config.dtype,
        )

        # Multimodal projector to bridge vision and text spaces
        self.projector = Gemma3MultiModalProjector(config)

    # from huggingface/gemma3/modeling_gemma3.py
    # TODO is this required?
    # def prepare_inputs_for_generation(
    #     self,
    # ):
    #     print("*** CALL PREPARE INPUTS FOR GENERATION ***")

    def __call__(
        self,
        pixel_values: Sequence[TensorValue],
        signal_buffers: Sequence[BufferValue],
    ) -> Sequence[TensorValue]:
        """Process pixel values to image embeddings.

        Args:
            pixel_values: List of image tensors [batch, channels, height, width] (one per device)
            signal_buffers: Communication buffers for distributed execution

        Returns:
            List of projected image embeddings [pooled_tokens, text_hidden_size] (one per device)
        """
        # convert to patches, run through the encoder layers, then normalize and project into multimodal space
        hidden_states = self.embeddings(pixel_values[0])

        hidden_states = self.encoder(hidden_states)

        hidden_states = self.post_layernorm(hidden_states)

        image_embeddings = self.projector(hidden_states)

        # Replicate to all devices
        return [image_embeddings for _ in range(len(self.config.devices))]
