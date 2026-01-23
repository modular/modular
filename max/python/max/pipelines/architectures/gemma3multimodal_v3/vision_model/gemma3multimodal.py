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

from attention import Gemma3Attention
from max import functional as F
from max.dtype import DType
from max.graph import DeviceRef
from max.nn import (
    MLP,
    LayerList,
    LayerNorm,
    ReturnLogits,
)
from max.nn.kv_cache import PagedCacheValues
from max.nn.module_v3 import Embedding, Linear, Module
from max.nn.rotary_embedding import (
    Llama3RopeScalingParams,
    Llama3RotaryEmbedding,
)
from max.pipelines.architectures.internvl.embedding_utils import (
    merge_multimodal_embeddings,
)
from max.tensor import Tensor
from rms_norm import Gemma3RMSNorm
from transformer_block import Gemma3TransformerBlock

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

        self.embed_tokens = Embedding(
            config.vocab_size,
            dim=config.hidden_size,
        )

        self.norm = Gemma3RMSNorm(
            text_config.hidden_size,
            DType.bfloat16,
            text_config.rms_norm_eps,
        )

        self.lm_head = Linear(
            text_config.hidden_size,
            text_config.vocab_size,
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
        tokens: Tensor,
        return_n_logits: Tensor,
        input_row_offsets: Tensor,
        image_embeddings: Tensor,
        image_token_indices: Tensor,
        kv_collections: Sequence[PagedCacheValues],
    ) -> tuple[Tensor, ...]:
        h = self.embed_tokens(tokens)

        # Replace image placeholder tokens with vision embeddings
        h = merge_multimodal_embeddings(
            inputs_embeds=h,
            multimodal_embeddings=image_embeddings,
            image_token_indices=image_token_indices,
        )

        # Run through transformer layers
        for idx, layer in enumerate(self.layers):
            layer_idx_tensor = F.constant(
                idx, DType.uint32, device=self.devices[0]
            )
            h = layer(
                layer_idx_tensor,
                h,
                kv_collections,
                input_row_offsets=input_row_offsets,
            )

        last_token_indices = input_row_offsets[1:] - 1
        last_token_h = F.gather(h, last_token_indices, axis=0)
        last_logits = F.cast(
            self.lm_head(self.norm(last_token_h)),
            DType.float32,
        )

        logits = None
        offsets = None

        if self.return_logits == ReturnLogits.VARIABLE and h:
            # Create range and gather indices for variable logits
            return_range = F.arange(
                start=return_n_logits[0],
                stop=0,
                step=-1,
                out_dim="return_n_logits_range",
                dtype=DType.int64,
                device=self.devices[0],
            )
            last_indices = F.reshape(
                F.unsqueeze(input_row_offsets[1:], -1) - return_range,
                shape=(-1,),
            )

            # Gather, normalize, and get logits
            variable_tokens = self.norm(F.gather(h, last_indices, axis=0))
            logits = F.cast(self.lm_head(variable_tokens), DType.float32)
            offsets = F.arange(
                0,
                last_indices.shape[0] + return_n_logits[0],
                return_n_logits[0],
                out_dim="logit_offsets",
                dtype=DType.int64,
                device=self.devices[0],
            )

        elif self.return_logits == ReturnLogits.ALL and h:
            # Apply normalization to all hidden states and get all logits
            all_normalized = self.norm(h)
            logits = F.cast(self.lm_head(all_normalized), DType.float32)
            offsets = input_row_offsets

        if offsets is not None:
            assert logits is not None
            return (last_logits, logits, offsets)

        return (last_logits,)


class Gemma3VisionModel(Module):
    """The Gemma3 Multi-Modal model's vision component"""

    def __init__(
        self, config: Gemma3ForConditionalGenerationConfig, device: DeviceRef
    ) -> None:
        """Initializes the necessary components for processing vision inputs and
        projecting into language space, with multi-device functionality."""
        super().__init__()
        self.config = config
        self.devices = config.devices
        vision_config = config.vision_config
        vision_dtype = DType.bfloat16

        # Vision embeddings, sharded for multi-device setups
        self.embeddings = Gemma3VisionEmbeddings(
            config, device=config.devices[0]
        )

        # Vision encoder (27 transformer layers)
        self.encoder = Gemma3VisionEncoder(config)

        # Post-encoder layer norm
        self.post_layernorm = LayerNorm(
            vision_config.hidden_size,
            eps=vision_config.layer_norm_eps,
            devices=[device],
            dtype=vision_dtype,
        )

        # Multimodal projector to project vision embeddings to language space
        self.projector = Gemma3MultiModalProjector(
            config, device=config.devices[0]
        )

    def __call__(
        self,
        pixel_values: Tensor,
    ) -> Tensor:
        """Processes vision inputs through the Gemma3 vision tower and produces a
        sequence of image embeddings"""
        hidden_states: Tensor | Sequence[Tensor] = self.embeddings(pixel_values)

        # Pass through encoder layers
        hidden_states = self.encoder(hidden_states)

        # Apply post-encoder layer norm
        hidden_states = self.post_layernorm(hidden_states)

        # Project image embeddings to language model hidden size and return
        return self.projector(hidden_states)
