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

"""UMT5 encoder implementation aligned with Hugging Face UMT5 behavior."""

from __future__ import annotations

import copy
import math
from collections.abc import Callable

import numpy as np
from max import functional as F
from max.driver import Device
from max.dtype import DType
from max.graph import Dim, TensorType
from max.nn import Embedding, Linear, Module
from max.nn.sequential import ModuleList
from max.tensor import Tensor

from .model_config import UMT5ConfigBase


def _get_act_fn(name: str) -> Callable[[Tensor], Tensor]:
    if name == "relu":
        return F.relu
    if name == "gelu_new":
        return lambda x: F.gelu(x, approximate="tanh")
    if name == "gelu":
        return F.gelu
    raise ValueError(
        f"Unsupported UMT5 dense activation '{name}'. "
        "Expected one of: relu, gelu, gelu_new."
    )


class UMT5LayerNorm(Module[..., Tensor]):
    def __init__(
        self,
        hidden_size: int,
        eps: float = 1e-6,
        dtype: DType = DType.float32,
    ):
        super().__init__()
        self.weight = Tensor.ones([hidden_size])
        self.variance_epsilon = eps
        self.dtype = dtype

    def forward(self, hidden_states: Tensor) -> Tensor:
        hidden_states_f32 = F.cast(hidden_states, DType.float32)
        variance = F.mean(F.pow(hidden_states_f32, 2), axis=-1)
        hidden_states = hidden_states * F.rsqrt(
            variance + self.variance_epsilon
        )

        if self.dtype in (DType.float16, DType.bfloat16):
            hidden_states = F.cast(hidden_states, self.dtype)
        return self.weight * hidden_states


class UMT5DenseActDense(Module[..., Tensor]):
    def __init__(self, config: UMT5ConfigBase):
        super().__init__()
        self.wi = Linear(config.d_model, config.d_ff, bias=False)
        self.wo = Linear(config.d_ff, config.d_model, bias=False)
        if config.dense_act_fn is None:
            raise ValueError("UMT5 dense_act_fn is not initialized.")
        self.act_fn = _get_act_fn(config.dense_act_fn)

    def forward(self, hidden_states: Tensor) -> Tensor:
        hidden_states = self.wi(hidden_states)
        hidden_states = self.act_fn(hidden_states)
        hidden_states = self.wo(hidden_states)
        return hidden_states


class UMT5DenseGatedActDense(Module[..., Tensor]):
    def __init__(self, config: UMT5ConfigBase):
        super().__init__()
        self.wi_0 = Linear(config.d_model, config.d_ff, bias=False)
        self.wi_1 = Linear(config.d_model, config.d_ff, bias=False)
        self.wo = Linear(config.d_ff, config.d_model, bias=False)
        if config.dense_act_fn is None:
            raise ValueError("UMT5 dense_act_fn is not initialized.")
        self.act_fn = _get_act_fn(config.dense_act_fn)

    def forward(self, hidden_states: Tensor) -> Tensor:
        hidden_gated = self.act_fn(self.wi_0(hidden_states))
        hidden_linear = self.wi_1(hidden_states)
        hidden_states = hidden_gated * hidden_linear
        hidden_states = self.wo(hidden_states)
        return hidden_states


class UMT5LayerFF(Module[..., Tensor]):
    def __init__(self, config: UMT5ConfigBase):
        super().__init__()
        if config.is_gated_act:
            self.DenseReluDense: UMT5DenseGatedActDense | UMT5DenseActDense = (
                UMT5DenseGatedActDense(config)
            )
        else:
            self.DenseReluDense = UMT5DenseActDense(config)
        self.layer_norm = UMT5LayerNorm(
            config.d_model,
            eps=config.layer_norm_epsilon,
            dtype=config.dtype,
        )

    def forward(self, hidden_states: Tensor) -> Tensor:
        forwarded_states = self.layer_norm(hidden_states)
        forwarded_states = self.DenseReluDense(forwarded_states)
        return hidden_states + forwarded_states


class UMT5Attention(Module[..., tuple[Tensor, Tensor]]):
    def __init__(
        self,
        config: UMT5ConfigBase,
        has_relative_attention_bias: bool = False,
        layer_idx: int | None = None,
    ):
        super().__init__()
        del layer_idx
        self.is_decoder = config.is_decoder
        self.has_relative_attention_bias = has_relative_attention_bias
        self.relative_attention_num_buckets = (
            config.relative_attention_num_buckets
        )
        self.relative_attention_max_distance = (
            config.relative_attention_max_distance
        )
        self.d_model = config.d_model
        self.key_value_proj_dim = config.d_kv
        self.n_heads = config.num_heads
        self.dropout = config.dropout_rate
        self.inner_dim = self.n_heads * self.key_value_proj_dim
        self.dtype = config.dtype

        self.q = Linear(self.d_model, self.inner_dim, bias=False)
        self.k = Linear(self.d_model, self.inner_dim, bias=False)
        self.v = Linear(self.d_model, self.inner_dim, bias=False)
        self.o = Linear(self.inner_dim, self.d_model, bias=False)

        if self.has_relative_attention_bias:
            self.relative_attention_bias = Embedding(
                self.relative_attention_num_buckets,
                dim=self.n_heads,
            )

    def _relative_position_bucket(self, relative_position: Tensor) -> Tensor:
        relative_buckets = Tensor.constant(
            0, dtype=DType.int32, device=relative_position.device
        )
        num_buckets = self.relative_attention_num_buckets
        max_distance = self.relative_attention_max_distance

        if not self.is_decoder:
            num_buckets = num_buckets // 2
            is_positive = F.greater(relative_position, 0)
            relative_buckets = relative_buckets + (
                F.cast(is_positive, DType.int32) * num_buckets
            )
            relative_position = F.abs(relative_position)
        else:
            relative_position = -F.min(relative_position, 0)

        max_exact = num_buckets // 2
        is_small = F.greater(max_exact, relative_position)

        scale = (num_buckets - max_exact) / math.log(max_distance / max_exact)
        rel_pos_float = F.cast(relative_position, DType.float32)
        val_log = F.log(rel_pos_float / float(max_exact))
        relative_position_if_large = max_exact + F.cast(
            val_log * scale, DType.int32
        )
        relative_position_if_large = F.min(
            relative_position_if_large, num_buckets - 1
        )

        return relative_buckets + F.where(
            is_small, relative_position, relative_position_if_large
        )

    def compute_bias(
        self,
        query_length: int | Dim,
        key_length: int | Dim,
        device: Device,
        cache_position: Tensor | None = None,
    ) -> Tensor:
        if cache_position is None:
            context_position = F.arange(
                0, query_length, step=1, dtype=DType.int32, device=device
            )
        else:
            context_position = cache_position
        context_position = F.unsqueeze(context_position, 1)

        memory_position = F.arange(
            0, key_length, step=1, dtype=DType.int32, device=device
        )
        memory_position = F.unsqueeze(memory_position, 0)
        relative_position = memory_position - context_position

        relative_position_bucket = self._relative_position_bucket(
            relative_position
        )
        values = self.relative_attention_bias(relative_position_bucket)
        values = F.permute(values, (2, 0, 1))
        values = F.unsqueeze(values, 0)
        return values

    def forward(
        self,
        hidden_states: Tensor,
        encoder_hidden_states: Tensor | None = None,
        past_key_values: Tensor | None = None,
        attention_mask: Tensor | None = None,
        cache_position: Tensor | None = None,
        output_attentions: bool = False,
    ) -> tuple[Tensor, Tensor]:
        del output_attentions
        if past_key_values is not None:
            raise NotImplementedError(
                "UMT5 cache is not implemented for MAX yet."
            )

        batch_size, seq_length = hidden_states.shape[:2]
        source_states = (
            encoder_hidden_states
            if encoder_hidden_states is not None
            else hidden_states
        )
        source_seq_length = source_states.shape[1]

        query_states = self.q(hidden_states)
        key_states = self.k(source_states)
        value_states = self.v(source_states)

        query_states = F.reshape(
            query_states,
            (batch_size, seq_length, self.n_heads, self.key_value_proj_dim),
        )
        key_states = F.reshape(
            key_states,
            (
                batch_size,
                source_seq_length,
                self.n_heads,
                self.key_value_proj_dim,
            ),
        )
        value_states = F.reshape(
            value_states,
            (
                batch_size,
                source_seq_length,
                self.n_heads,
                self.key_value_proj_dim,
            ),
        )

        query_states = F.permute(query_states, (0, 2, 1, 3))
        key_states = F.permute(key_states, (0, 2, 1, 3))
        value_states = F.permute(value_states, (0, 2, 1, 3))

        scores = F.matmul(query_states, F.permute(key_states, (0, 1, 3, 2)))

        if self.has_relative_attention_bias:
            position_bias = self.compute_bias(
                seq_length,
                source_seq_length,
                hidden_states.device,
                cache_position=cache_position,
            )
            scores = scores + position_bias

        if attention_mask is not None:
            scores = scores + attention_mask

        attn_weights = F.softmax(F.cast(scores, DType.float32), axis=-1)
        attn_weights = F.cast(attn_weights, self.dtype)
        attn_output = F.matmul(attn_weights, value_states)
        attn_output = F.permute(attn_output, (0, 2, 1, 3))
        attn_output = F.reshape(
            attn_output,
            (batch_size, seq_length, self.inner_dim),
        )
        attn_output = self.o(attn_output)
        return attn_output, attn_weights


class UMT5LayerSelfAttention(
    Module[..., tuple[Tensor] | tuple[Tensor, Tensor]]
):
    def __init__(self, config: UMT5ConfigBase, layer_idx: int | None = None):
        super().__init__()
        self.SelfAttention = UMT5Attention(
            config,
            has_relative_attention_bias=True,
            layer_idx=layer_idx,
        )
        self.layer_norm = UMT5LayerNorm(
            config.d_model,
            eps=config.layer_norm_epsilon,
            dtype=config.dtype,
        )

    def forward(
        self,
        hidden_states: Tensor,
        attention_mask: Tensor | None = None,
        past_key_values: Tensor | None = None,
        cache_position: Tensor | None = None,
        output_attentions: bool = False,
    ) -> tuple[Tensor] | tuple[Tensor, Tensor]:
        normed_hidden_states = self.layer_norm(hidden_states)
        attention_output = self.SelfAttention(
            normed_hidden_states,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            cache_position=cache_position,
            output_attentions=output_attentions,
        )
        hidden_states = hidden_states + attention_output[0]
        if output_attentions:
            return hidden_states, attention_output[1]
        return (hidden_states,)


class UMT5Block(Module[..., tuple[Tensor] | tuple[Tensor, Tensor]]):
    def __init__(self, config: UMT5ConfigBase, layer_idx: int | None = None):
        super().__init__()
        self.is_decoder = config.is_decoder
        self.layer = ModuleList(
            [
                UMT5LayerSelfAttention(config, layer_idx=layer_idx),
                UMT5LayerFF(config),
            ]
        )

    def forward(
        self,
        hidden_states: Tensor,
        attention_mask: Tensor | None = None,
        encoder_hidden_states: Tensor | None = None,
        encoder_attention_mask: Tensor | None = None,
        past_key_values: Tensor | None = None,
        use_cache: bool = False,
        output_attentions: bool = False,
        cache_position: Tensor | None = None,
    ) -> tuple[Tensor] | tuple[Tensor, Tensor]:
        del encoder_attention_mask
        del use_cache

        self_attention_outputs = self.layer[0](
            hidden_states,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            cache_position=cache_position,
            output_attentions=output_attentions,
        )
        hidden_states = self_attention_outputs[0]
        self_attn_weights = (
            self_attention_outputs[1]
            if len(self_attention_outputs) > 1
            else None
        )

        if hidden_states.dtype == DType.float16:
            clamp_value = float(np.finfo(np.float16).max) - 1000
            hidden_states = hidden_states.clip(
                min=-clamp_value, max=clamp_value
            )

        do_cross_attention = (
            self.is_decoder and encoder_hidden_states is not None
        )
        if do_cross_attention:
            raise NotImplementedError(
                "UMT5 decoder cross attention is not implemented in MAX."
            )

        hidden_states = self.layer[-1](hidden_states)
        if hidden_states.dtype == DType.float16:
            clamp_value = float(np.finfo(np.float16).max) - 1000
            hidden_states = hidden_states.clip(
                min=-clamp_value, max=clamp_value
            )

        if output_attentions and self_attn_weights is not None:
            return hidden_states, self_attn_weights
        return (hidden_states,)


class UMT5Stack(Module[..., Tensor]):
    def __init__(
        self,
        config: UMT5ConfigBase,
        embed_tokens: Embedding | None = None,
    ):
        super().__init__()
        self.config = config
        self.embed_tokens = embed_tokens
        self.is_decoder = config.is_decoder
        self.block = ModuleList(
            [UMT5Block(config, layer_idx=i) for i in range(config.num_layers)]
        )
        self.final_layer_norm = UMT5LayerNorm(
            config.d_model,
            eps=config.layer_norm_epsilon,
            dtype=config.dtype,
        )
        self.dtype = config.dtype

    def forward(
        self,
        input_ids: Tensor | None = None,
        attention_mask: Tensor | None = None,
        inputs_embeds: Tensor | None = None,
        encoder_hidden_states: Tensor | None = None,
        encoder_attention_mask: Tensor | None = None,
        past_key_values: Tensor | None = None,
        use_cache: bool = False,
        output_attentions: bool = False,
        cache_position: Tensor | None = None,
    ) -> Tensor:
        del encoder_attention_mask
        del past_key_values
        del cache_position

        use_cache = (
            use_cache if use_cache is not None else self.config.use_cache
        )
        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.config.output_attentions
        )

        if input_ids is not None:
            if input_ids.rank == 1:
                input_ids = F.unsqueeze(input_ids, 0)
            if self.embed_tokens is None:
                raise ValueError(
                    "embed_tokens must be provided when input_ids is used"
                )
            inputs_embeds = self.embed_tokens(input_ids)
        elif inputs_embeds is None:
            raise ValueError(
                "You have to specify either input_ids or inputs_embeds"
            )
        elif inputs_embeds.rank == 2:
            inputs_embeds = F.unsqueeze(inputs_embeds, 0)

        if self.is_decoder or use_cache:
            raise NotImplementedError("UMT5 decoder is not implemented yet.")

        hidden_states = inputs_embeds
        if attention_mask is not None:
            if attention_mask.rank == 1:
                attention_mask = F.unsqueeze(attention_mask, 0)
            try:
                dtype_np = hidden_states.dtype.to_numpy()
            except (AttributeError, ValueError):
                dtype_np = np.float32
            mask_multiplier = F.constant(
                float(np.finfo(dtype_np).min),
                dtype=hidden_states.dtype,
                device=hidden_states.device,
            )
            causal_mask = (
                F.constant(
                    1.0,
                    dtype=hidden_states.dtype,
                    device=hidden_states.device,
                )
                - F.cast(attention_mask, hidden_states.dtype)
            ) * mask_multiplier
            causal_mask = F.unsqueeze(causal_mask, 1)
            causal_mask = F.unsqueeze(causal_mask, 1)
        else:
            causal_mask = None

        all_attentions: tuple[Tensor, ...] = ()
        for layer_module in self.block:
            layer_outputs = layer_module(
                hidden_states,
                attention_mask=causal_mask,
                encoder_hidden_states=encoder_hidden_states,
                output_attentions=output_attentions,
            )
            hidden_states = layer_outputs[0]
            if output_attentions and len(layer_outputs) > 1:
                all_attentions = all_attentions + (layer_outputs[1],)

        hidden_states = self.final_layer_norm(hidden_states)
        return hidden_states


class UMT5EncoderModel(Module[..., Tensor]):
    def __init__(self, config: UMT5ConfigBase):
        super().__init__()
        act_info = config.feed_forward_proj.split("-")
        config.dense_act_fn = act_info[-1]
        config.is_gated_act = act_info[0] == "gated"
        if (len(act_info) > 1 and act_info[0] != "gated") or len(act_info) > 2:
            raise ValueError(
                f"`feed_forward_proj`: {config.feed_forward_proj} is not valid."
            )
        if config.feed_forward_proj == "gated-gelu":
            config.dense_act_fn = "gelu_new"

        self.shared = Embedding(config.vocab_size, dim=config.d_model)

        encoder_config = copy.deepcopy(config)
        encoder_config.is_decoder = False
        encoder_config.use_cache = False
        encoder_config.is_encoder_decoder = False
        self.encoder = UMT5Stack(encoder_config, self.shared)
        self.device = config.device

    def input_types(self) -> tuple[TensorType, ...]:
        return (
            TensorType(
                DType.int64,
                shape=["batch_size", "sequence_length"],
                device=self.device,
            ),
            TensorType(
                DType.int64,
                shape=["batch_size", "sequence_length"],
                device=self.device,
            ),
        )

    def forward(
        self,
        input_ids: Tensor | None = None,
        attention_mask: Tensor | None = None,
    ) -> Tensor:
        return self.encoder(input_ids=input_ids, attention_mask=attention_mask)


__all__ = ["UMT5EncoderModel"]
