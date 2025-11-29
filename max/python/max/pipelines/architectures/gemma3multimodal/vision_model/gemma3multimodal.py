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
from collections.abc import Sequence

from max.dtype import DType
from max.graph import (
    BufferValue,
    ShardingStrategy,
    TensorValue,
    ops,
)
from max.nn import (
    MLP,
    ColumnParallelLinear,
    LayerList,
    LayerNorm,
    Module,
    ReturnLogits,
)
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

from ..model_config import Gemma3ForConditionalGenerationConfig
from .embedding import Gemma3VisionEmbeddings
from .encoding import Gemma3VisionEncoder
from .projection import Gemma3MultiModalProjector

logger = logging.getLogger("max.pipelines")


class Gemma3LanguageModel(Module):
    """The Gemma3 Multi-Modal model's text component, shared with Gemma3"""

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
        return_n_logits: TensorValue,
        input_row_offsets: Sequence[TensorValue],
        image_embeddings: Sequence[TensorValue],
        image_token_indices: Sequence[TensorValue],
        signal_buffers: Sequence[BufferValue],
        kv_collections: Sequence[PagedCacheValues],
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


class Gemma3VisionModel(Module):
    """The Gemma3 Multi-Modal model's vision component"""

    def __init__(self, config: Gemma3ForConditionalGenerationConfig) -> None:
        """Initializes the necessary components for processing vision inputs and
        projecting into language space, with multi-device functionality."""
        super().__init__()
        self.config = config
        self.devices = config.devices
        vision_config = config.vision_config

        # Vision embeddings, sharded for multi-device setups
        self.embeddings = Gemma3VisionEmbeddings(config, device=config.devices)
        self.embeddings.sharding_strategy = ShardingStrategy.replicate(
            len(config.devices)
        )
        self.embeddings_list = self.embeddings.shard(config.devices)

        # Vision encoder (27 transformer layers)
        self.encoder = Gemma3VisionEncoder(config)

        # Post-encoder layer norm
        self.post_layernorm = LayerNorm(
            vision_config.hidden_size,
            eps=vision_config.layer_norm_eps,
            devices=[config.devices],
            dtype=config.dtype,
        )
        self.post_layernorm.weight.sharding_strategy = (
            ShardingStrategy.replicate(len(config.devices))
        )
        if self.post_layernorm.bias is not None:
            self.post_layernorm.bias.sharding_strategy = (
                ShardingStrategy.replicate(len(config.devices))
            )

        # Shard post_layernorm across devices
        post_layernorm_weight_shards = self.post_layernorm.weight.shard(
            config.devices
        )
        post_layernorm_bias_shards = (
            self.post_layernorm.bias.shard(config.devices)
            if self.post_layernorm.bias is not None
            else [None] * len(config.devices)
        )

        self.post_layernorm_list = []
        for device, weight_shard, bias_shard in zip(
            config.devices,
            post_layernorm_weight_shards,
            post_layernorm_bias_shards,
            strict=True,
        ):
            ln = LayerNorm(
                vision_config.hidden_size,
                eps=vision_config.layer_norm_eps,
                devices=[device],
                dtype=config.dtype,
            )
            ln.weight = weight_shard
            if bias_shard is not None:
                ln.bias = bias_shard
            self.post_layernorm_list.append(ln)

        # Multimodal projector to project vision embeddings to language space
        self.projector = Gemma3MultiModalProjector(
            config, device=config.devices[0]
        )
        self.projector.sharding_strategy = ShardingStrategy.replicate(
            len(config.devices)
        )
        self.projector_list = self.projector.shard(config.devices)

    def __call__(
        self,
        pixel_values: Sequence[TensorValue],
        signal_buffers: Sequence[BufferValue],
    ) -> Sequence[TensorValue]:
        """Processes vision inputs through the Gemma3 vision tower and produces a
        sequence of image embeddings"""
        hidden_states = [
            embed(pixels)
            for embed, pixels in zip(
                self.embeddings_list, pixel_values, strict=True
            )
        ]

        # Pass through encoder layers
        hidden_states = self.encoder(hidden_states, signal_buffers)

        # Apply post-encoder layer norm
        if isinstance(hidden_states, list):
            hidden_states = [
                layer(states)
                for layer, states in zip(
                    self.post_layernorm_list, hidden_states, strict=True
                )
            ]
        else:
            hidden_states = self.post_layernorm_list[0](hidden_states)

        # Project to language model hidden size
        if isinstance(hidden_states, list):
            image_embeddings_list = [
                projector(states)
                for projector, states in zip(
                    self.projector_list, hidden_states, strict=True
                )
            ]
        else:
            image_embeddings_list = [self.projector_list[0](hidden_states)]

        return image_embeddings_list
