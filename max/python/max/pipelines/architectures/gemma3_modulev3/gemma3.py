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

"""Implements the Gemma3 model using the ModuleV3 API."""

from __future__ import annotations

import functools

from max.dtype import DType
from max.experimental import functional as F
from max.experimental.nn import Module
from max.experimental.nn.common_layers.mlp import MLP
from max.experimental.nn.linear import Linear
from max.experimental.nn.sequential import ModuleList
from max.experimental.tensor import Tensor
from max.graph import TensorValue, ops
from max.nn.kv_cache import (
    KVCacheParamInterface,
    PagedCacheValues,
    unflatten_ragged_attention_inputs,
)
from max.nn.rotary_embedding import Llama3RopeScalingParams
from max.nn.transformer import ReturnLogits

from .layers.attention import Gemma3Attention
from .layers.rms_norm import Gemma3RMSNorm
from .layers.rotary_embedding import Llama3RotaryEmbedding
from .layers.scaled_word_embedding import ScaledEmbedding
from .layers.transformer_block import Gemma3TransformerBlock
from .model_config import Gemma3Config


class Gemma3TextModel(
    Module[[Tensor, PagedCacheValues, Tensor, Tensor], tuple[Tensor, ...]]
):
    """The Gemma3 language model (single-device ModuleV3 implementation)."""

    def __init__(self, config: Gemma3Config) -> None:
        super().__init__()
        # Use scaling_params for both cases (with and without scaling)
        scaling_params = (
            Llama3RopeScalingParams(
                factor=config.rope_scaling.factor,
                low_freq_factor=1e38,  # degenerates to linear scaling
                high_freq_factor=1e38,
                orig_max_position=config.max_position_embeddings,
            )
            if config.rope_scaling is not None
            else None
        )

        device = config.devices[0].to_device()

        rope_global = Llama3RotaryEmbedding(
            dim=config.hidden_size,
            n_heads=config.num_attention_heads,
            theta=config.rope_theta,
            max_seq_len=config.max_position_embeddings,
            device=device,
            head_dim=config.head_dim,
            interleaved=False,
            scaling_params=scaling_params,
        )

        rope_local = Llama3RotaryEmbedding(
            dim=config.hidden_size,
            n_heads=config.num_attention_heads,
            theta=config.rope_local_base_freq,
            max_seq_len=config.max_position_embeddings,
            device=device,
            head_dim=config.head_dim,
            interleaved=False,
            scaling_params=None,
        )

        self.embed_tokens = ScaledEmbedding(
            config.vocab_size,
            dim=config.hidden_size,
            embed_scale=config.hidden_size**0.5,
        )

        self.norm = Gemma3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        self.tie_word_embeddings = config.tie_word_embeddings
        if config.tie_word_embeddings:
            self.lm_head = None
        else:
            self.lm_head = Linear(
                in_dim=config.hidden_size,
                out_dim=config.vocab_size,
                bias=False,
            )

        create_norm = functools.partial(
            Gemma3RMSNorm,
            config.hidden_size,
            eps=config.rms_norm_eps,
        )

        layers = [
            Gemma3TransformerBlock(
                attention=Gemma3Attention(
                    rope_global=rope_global,
                    rope_local=rope_local,
                    num_attention_heads=config.num_attention_heads,
                    num_key_value_heads=config.num_key_value_heads,
                    hidden_size=config.hidden_size,
                    kv_params=config.kv_params,
                    layer_idx=i,
                    sliding_window_pattern=config.sliding_window_pattern,
                    has_bias=config.attention_bias,
                    qk_norm_eps=config.rms_norm_eps,
                    local_window_size=config.sliding_window,
                ),
                mlp=MLP(
                    hidden_dim=config.hidden_size,
                    feed_forward_length=config.intermediate_size,
                    activation_function=config.hidden_activation,
                ),
                input_layernorm=create_norm(),
                post_attention_layernorm=create_norm(),
                pre_feedforward_layernorm=create_norm(),
                post_feedforward_layernorm=create_norm(),
            )
            for i in range(config.num_hidden_layers)
        ]

        self.layers = ModuleList(layers)
        self.kv_params = config.kv_params
        self.return_logits = config.return_logits

    def _compute_logits(self, h: Tensor) -> Tensor:
        """Compute logits from hidden states, handling weight tying."""
        if self.tie_word_embeddings:
            return F.cast(h @ self.embed_tokens.weight.T, DType.float32)
        assert self.lm_head is not None
        return F.cast(self.lm_head(h), DType.float32)

    def forward(
        self,
        tokens: Tensor,
        kv_collection: PagedCacheValues,
        return_n_logits: Tensor,
        input_row_offsets: Tensor,
    ) -> tuple[Tensor, ...]:
        h = self.embed_tokens(tokens)

        for idx, layer in enumerate(self.layers):
            layer_idx_tensor = F.constant(idx, DType.uint32, device=h.device)
            h = layer(
                layer_idx_tensor,
                h,
                kv_collection,
                input_row_offsets=input_row_offsets,
            )

        # Gather last tokens and compute last-token logits.
        last_h = F.gather(h, input_row_offsets[1:] - 1, axis=0)
        last_logits = self._compute_logits(self.norm(last_h))

        logits = None
        offsets = None

        if self.return_logits == ReturnLogits.VARIABLE:
            return_n_logits_range = ops.range(
                return_n_logits[0],
                0,
                -1,
                out_dim="return_n_logits_range",
                device=h.device,
                dtype=DType.int64,
            )
            offsets = (
                F.unsqueeze(input_row_offsets[1:], -1) - return_n_logits_range
            )
            last_indices = F.reshape(offsets, shape=(-1,))
            last_tokens = F.gather(h, last_indices, axis=0)
            logits = self._compute_logits(self.norm(last_tokens))
            offsets = ops.range(
                0,
                TensorValue(last_indices.shape[0]) + return_n_logits[0],
                return_n_logits[0],
                out_dim="logit_offsets",
                device=h.device,
                dtype=DType.int64,
            )
        elif self.return_logits == ReturnLogits.ALL:
            logits = self._compute_logits(self.norm(h))
            offsets = input_row_offsets

        ret_val: tuple[Tensor, ...] = (last_logits,)
        if offsets is not None:
            assert logits is not None
            ret_val += (logits, offsets)

        return ret_val


class Gemma3(Module[..., tuple[Tensor, ...]]):
    """The Gemma3 model (ModuleV3 wrapper).

    Top-level wrapper that unflattens the variadic KV cache arguments
    and delegates to :class:`Gemma3TextModel`.
    """

    def __init__(
        self,
        config: Gemma3Config,
        kv_params: KVCacheParamInterface,
    ) -> None:
        super().__init__()
        self.language_model = Gemma3TextModel(config)
        self.config = config
        self.kv_params = kv_params

    def forward(
        self,
        tokens: Tensor,
        return_n_logits: Tensor,
        input_row_offsets: Tensor,
        *variadic_args,
    ) -> tuple[Tensor, ...]:
        kv_collections = unflatten_ragged_attention_inputs(
            variadic_args, n_devices=self.kv_params.n_devices
        )
        return self.language_model(
            tokens, kv_collections[0], return_n_logits, input_row_offsets
        )
