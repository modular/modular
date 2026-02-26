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

"""Implements the GPT OSS model."""

from __future__ import annotations

import functools
from collections.abc import Sequence

from max.dtype import DType
from max.experimental import functional as F
from max.experimental.tensor import Tensor
from max.graph import BufferValue, TensorValue
from max.nn.attention import MHAMaskVariant
from max.nn.kv_cache import KVCacheParamInterface, PagedCacheValues
from max.nn.module_v3 import Module
from max.nn.module_v3.embedding import Embedding
from max.nn.module_v3.linear import Linear
from max.nn.module_v3.sequential import ModuleList

from ..common_layers.rotary_embedding import (
    YarnRotaryEmbedding,
    YarnScalingParams,
)
from .layers.attention import GptOssAttention
from .layers.moe import GptOssMoE
from .layers.rms_norm import GptOssRMSNorm
from .layers.transformer_block import GptOssTransformerBlock
from .model_config import GptOssConfig


class GptOssTextModel(
    Module[[Tensor, PagedCacheValues, Tensor, Tensor], tuple[Tensor, ...]]
):
    """The GPT OSS language model.

    Decoder-only Transformer with MoE feed-forward, rotary embeddings (YARN),
    and mixed attention (full + sliding window).
    """

    def __init__(self, config: GptOssConfig) -> None:
        super().__init__()
        self.devices = config.devices

        # Create YARN scaling params if configured
        assert config.rope_scaling is not None, (
            "RoPE scaling is required for GPT-OSS models"
        )
        assert isinstance(config.rope_scaling, YarnScalingParams), (
            "Only YARN scaling is supported for GPT-OSS models"
        )
        yarn_scaling_params: YarnScalingParams = config.rope_scaling

        # RoPE with YARN scaling for full and window attention layers
        rope = YarnRotaryEmbedding(
            dim=config.hidden_size,
            n_heads=config.num_attention_heads,
            theta=config.rope_theta,
            max_seq_len=config.max_position_embeddings,
            device=config.devices[0].to_device(),
            head_dim=config.head_dim,
            interleaved=False,
            scaling_params=yarn_scaling_params,
        )
        self.embed_tokens = Embedding(
            config.vocab_size,
            dim=config.hidden_size,
        )

        self.norm = GptOssRMSNorm(
            config.hidden_size,
            config.rms_norm_eps,
        )

        self.lm_head = Linear(
            in_dim=config.hidden_size,
            out_dim=config.vocab_size,
            bias=False,
        )

        create_norm = functools.partial(
            GptOssRMSNorm,
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
            layers.append(
                GptOssTransformerBlock(
                    attention=GptOssAttention(
                        rope=rope,
                        num_attention_heads=config.num_attention_heads,
                        num_key_value_heads=config.num_key_value_heads,
                        hidden_size=config.hidden_size,
                        kv_params=config.kv_params,
                        layer_idx=i,
                        local_window_size=config.sliding_window,
                        has_bias=config.attention_bias,
                        mask_variant=mask_variant,
                    ),
                    mlp=GptOssMoE(config),
                    input_layernorm=create_norm(),
                    post_attention_layernorm=create_norm(),
                )
            )

        self.dim = config.hidden_size
        self.n_heads = config.num_attention_heads
        self.layers = ModuleList(layers)
        self.kv_params = config.kv_params
        self.return_logits = config.return_logits

    def forward(
        self,
        tokens: Tensor,
        kv_collection: PagedCacheValues,
        return_n_logits: Tensor,
        input_row_offsets: Tensor,
    ) -> tuple[Tensor, ...]:
        h = self.embed_tokens(tokens)
        # Run through transformer layers
        for idx, layer in enumerate(self.layers):
            layer_idx_tensor = F.constant(idx, DType.uint32, device=h.device)
            h = layer(
                layer_idx_tensor,
                h,
                kv_collection,
                input_row_offsets=input_row_offsets,
            )

        # Get last token logits only (no variable logits support).
        last_token_indices = input_row_offsets[1:] - 1
        last_token_h = F.gather(h, last_token_indices, axis=0)
        last_logits = F.cast(
            # Take only the device 0 logits to device-to-host transfer.
            self.lm_head(self.norm(last_token_h)),
            DType.float32,
        )

        # For now, simplified to return last token only
        # TODO: Handle VARIABLE and ALL logits cases for distributed processing
        return (last_logits,)


class GptOss(Module[..., tuple[Tensor, ...]]):
    """The GPT OSS model."""

    def __init__(
        self,
        config: GptOssConfig,
        kv_params: KVCacheParamInterface,
    ) -> None:
        super().__init__()
        self.language_model = GptOssTextModel(config)
        self.config = config
        self.kv_params = kv_params

    def forward(
        self,
        tokens: Tensor,
        return_n_logits: Tensor,
        input_row_offsets: Tensor,
        *variadic_args,
    ) -> tuple[Tensor, ...]:
        kv_collection = _unflatten_kv_inputs(
            self.config, self.kv_params, variadic_args
        )
        return self.language_model(
            tokens, kv_collection[0], return_n_logits, input_row_offsets
        )


def _unflatten_kv_inputs(
    config: GptOssConfig,
    kv_params: KVCacheParamInterface,
    kv_inputs_flat: Sequence[Tensor],
) -> list[PagedCacheValues]:
    kv_params = config.kv_params
    n_devices = kv_params.n_devices
    fetch_types = kv_params.get_symbolic_inputs()[0]
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
