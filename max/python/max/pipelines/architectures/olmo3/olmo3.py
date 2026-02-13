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

"""Implements the Olmo3 model."""

from __future__ import annotations

import functools
from collections.abc import Sequence

from max import functional as F
from max.dtype import DType
from max.graph import BufferValue, TensorValue
from max.kv_cache import PagedKVCacheManager
from max.nn import Module
from max.nn.embedding import Embedding
from max.nn.legacy.attention import MHAMaskVariant
from max.nn.legacy.kv_cache import PagedCacheValues
from max.nn.linear import Linear
from max.nn.sequential import ModuleList
from max.tensor import Tensor

from ..common_layers.mlp import MLP
from ..common_layers.rotary_embedding import (
    RotaryEmbedding,
    YarnRotaryEmbedding,
    YarnScalingParams,
)
from .layers.attention import Olmo3Attention
from .layers.rms_norm import Olmo3RMSNorm
from .layers.transformer_block import Olmo3TransformerBlock
from .model_config import Olmo3Config


class Olmo3TextModel(
    Module[[Tensor, PagedCacheValues, Tensor, Tensor], tuple[Tensor]]
):
    """The Olmo3 language model.

    Decoder-only Transformer with standard MLP feed-forward,
    rotary embeddings (YARN), and mixed attention (full + sliding window).

    Olmo3 includes Q and K normalization after Q/K projections.
    """

    def __init__(self, config: Olmo3Config) -> None:
        super().__init__()
        self.devices = config.devices

        if config.rope_scaling is not None:
            if not isinstance(config.rope_scaling, YarnScalingParams):
                raise ValueError(
                    "Only YARN scaling is supported for Olmo3 models"
                )
            yarn_scaling_params: YarnScalingParams = config.rope_scaling
        else:
            yarn_scaling_params = YarnScalingParams(
                factor=32.0,
                beta_fast=32.0,
                beta_slow=1.0,
                original_max_position_embeddings=4096,
                truncate=False,
            )

        # Create YARN RoPE for full attention layers
        yarn_rope = YarnRotaryEmbedding(
            dim=config.hidden_size,
            n_heads=config.num_attention_heads,
            theta=config.rope_theta,
            max_seq_len=config.max_position_embeddings,
            device=config.devices[0].to_device(),
            head_dim=config.head_dim,
            interleaved=False,
            scaling_params=yarn_scaling_params,
        )

        # Create basic RoPE for sliding attention layers
        basic_rope = RotaryEmbedding(
            dim=config.hidden_size,
            n_heads=config.num_attention_heads,
            theta=config.rope_theta,
            max_seq_len=config.max_position_embeddings,
            device=config.devices[0].to_device(),
            head_dim=config.head_dim,
            interleaved=False,
        )
        self.embed_tokens = Embedding(
            config.vocab_size,
            dim=config.hidden_size,
        )

        self.norm = Olmo3RMSNorm(
            config.hidden_size,
            config.rms_norm_eps,
        )

        if config.tie_word_embeddings:
            self.lm_head = None
        else:
            self.lm_head = Linear(
                in_dim=config.hidden_size,
                out_dim=config.vocab_size,
                bias=False,
            )

        create_norm = functools.partial(
            Olmo3RMSNorm,
            config.hidden_size,
            eps=config.rms_norm_eps,
        )

        layers = []
        for i in range(config.num_hidden_layers):
            if i < len(config.layer_types):
                layer_type = config.layer_types[i]
            else:
                layer_type = "full_attention"
            mask_variant = (
                MHAMaskVariant.SLIDING_WINDOW_CAUSAL_MASK
                if layer_type == "sliding_attention"
                else MHAMaskVariant.CAUSAL_MASK
            )
            # Use basic RoPE for sliding attention, YARN RoPE for full attention
            layer_rope = (
                basic_rope if layer_type == "sliding_attention" else yarn_rope
            )
            layers.append(
                Olmo3TransformerBlock(
                    attention=Olmo3Attention(
                        rope=layer_rope,
                        num_attention_heads=config.num_attention_heads,
                        num_key_value_heads=config.num_key_value_heads,
                        hidden_size=config.hidden_size,
                        kv_params=config.kv_params,
                        layer_idx=i,
                        local_window_size=config.sliding_window,
                        has_bias=config.attention_bias,
                        mask_variant=mask_variant,
                        use_qk_norm=config.use_qk_norm,
                        qk_norm_eps=config.qk_norm_eps,
                    ),
                    mlp=MLP(
                        hidden_dim=config.hidden_size,
                        feed_forward_length=config.intermediate_size,
                        bias=config.attention_bias,
                        activation_function=config.hidden_activation,
                    ),
                    post_attention_layernorm=create_norm(),
                    post_feedforward_layernorm=create_norm(),
                )
            )

        self.dim = config.hidden_size
        self.n_heads = config.num_attention_heads
        self.layers = ModuleList(layers)
        self.kv_params = config.kv_params
        self.return_logits = config.return_logits
        self.tie_word_embeddings = config.tie_word_embeddings

    def forward(
        self,
        tokens: Tensor,
        kv_collection: PagedCacheValues,
        return_n_logits: Tensor,
        input_row_offsets: Tensor,
    ) -> tuple[Tensor]:
        h = self.embed_tokens(tokens)
        for idx, layer in enumerate(self.layers):
            layer_idx_tensor = F.constant(idx, DType.uint32, device=h.device)
            h = layer(
                layer_idx_tensor,
                h,
                kv_collection,
                input_row_offsets,
            )

        last_token_indices = input_row_offsets[1:] - 1
        last_token_h = F.gather(h, last_token_indices, axis=0)
        last_token_h = self.norm(last_token_h)

        if self.tie_word_embeddings:
            last_logits = F.cast(
                last_token_h @ self.embed_tokens.weight.T,
                DType.float32,
            )
        else:
            assert self.lm_head is not None
            last_logits = F.cast(
                self.lm_head(last_token_h),
                DType.float32,
            )

        return (last_logits,)


class Olmo3(Module[[Tensor, Tensor, Tensor], tuple[Tensor]]):
    """The Olmo3 model."""

    def __init__(
        self,
        config: Olmo3Config,
        kv_manager: PagedKVCacheManager,
    ) -> None:
        super().__init__()
        self.language_model = Olmo3TextModel(config)
        self.config = config
        self.kv_manager = kv_manager

    def forward(
        self,
        tokens: Tensor,
        return_n_logits: Tensor,
        input_row_offsets: Tensor,
        *variadic_args,
    ) -> tuple[Tensor]:
        kv_collection = _unflatten_kv_inputs(
            self.config, self.kv_manager, variadic_args
        )
        return self.language_model(
            tokens, kv_collection[0], return_n_logits, input_row_offsets
        )


def _unflatten_kv_inputs(
    config: Olmo3Config,
    kv_manager: PagedKVCacheManager,
    kv_inputs_flat: Sequence[Tensor],
) -> list[PagedCacheValues]:
    kv_params = config.kv_params
    n_devices = kv_params.n_devices
    fetch_types = kv_manager.params.get_symbolic_inputs()[0]
    len_of_kv_tuple_per_dev = len(list(fetch_types))
    kv_caches_per_dev: list[PagedCacheValues] = []
    for i in range(n_devices):
        start_idx = i * len_of_kv_tuple_per_dev

        kv_block = kv_inputs_flat[start_idx]
        cache_lengths = kv_inputs_flat[start_idx + 1]
        lookup_table = kv_inputs_flat[start_idx + 2]
        max_lengths = kv_inputs_flat[start_idx + 3]

        kv_caches_per_dev.append(
            PagedCacheValues(
                kv_blocks=BufferValue(kv_block),
                cache_lengths=TensorValue(cache_lengths),
                lookup_table=TensorValue(lookup_table),
                max_lengths=TensorValue(max_lengths),
            )
        )
    return kv_caches_per_dev
