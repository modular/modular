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

"""UMT5 encoder implementation aligned with Hugging Face UMT5 behavior.

Uses module_v2 (nn.layer.Module / ops) API exclusively.
"""

from __future__ import annotations

import copy
import math

from max.dtype import DType
from max.graph import DeviceRef, Dim, TensorValue, Weight, ops
from max.nn.embedding import Embedding
from max.nn.layer import LayerList, Module
from max.nn.linear import Linear

from .model_config import UMT5ConfigBase


class UMT5LayerNorm(Module):
    """T5-style RMSNorm (no bias, no mean subtraction)."""

    def __init__(
        self,
        hidden_size: int,
        eps: float = 1e-6,
        *,
        dtype: DType = DType.float32,
        device: DeviceRef = DeviceRef.GPU(),
    ) -> None:
        super().__init__()
        self.weight = Weight("weight", dtype, [hidden_size], device)
        self.variance_epsilon = eps
        self._dtype = dtype

    def __call__(self, hidden_states: TensorValue) -> TensorValue:
        x = ops.cast(hidden_states, DType.float32)
        variance = ops.mean(x * x, axis=-1)
        x = x * ops.rsqrt(variance + self.variance_epsilon)
        if self._dtype in (DType.float16, DType.bfloat16):
            x = ops.cast(x, self._dtype)
        return ops.cast(self.weight, x.dtype) * x


class UMT5DenseActDense(Module):
    def __init__(
        self,
        config: UMT5ConfigBase,
        *,
        dtype: DType,
        device: DeviceRef,
    ) -> None:
        super().__init__()
        self.wi = Linear(
            in_dim=config.d_model,
            out_dim=config.d_ff,
            dtype=dtype,
            device=device,
            has_bias=False,
        )
        self.wo = Linear(
            in_dim=config.d_ff,
            out_dim=config.d_model,
            dtype=dtype,
            device=device,
            has_bias=False,
        )
        act_name = config.dense_act_fn
        if act_name == "relu":
            self._act = "relu"
        elif act_name in ("gelu_new", "gelu"):
            self._act = act_name
        else:
            raise ValueError(f"Unsupported UMT5 dense_act_fn: {act_name}")

    def __call__(self, hidden_states: TensorValue) -> TensorValue:
        hidden_states = self.wi(hidden_states)
        if self._act == "relu":
            hidden_states = ops.relu(hidden_states)
        elif self._act == "gelu_new":
            hidden_states = ops.gelu(hidden_states, approximate="tanh")
        else:
            hidden_states = ops.gelu(hidden_states)
        return self.wo(hidden_states)


class UMT5DenseGatedActDense(Module):
    def __init__(
        self,
        config: UMT5ConfigBase,
        *,
        dtype: DType,
        device: DeviceRef,
    ) -> None:
        super().__init__()
        self.wi_0 = Linear(
            in_dim=config.d_model,
            out_dim=config.d_ff,
            dtype=dtype,
            device=device,
            has_bias=False,
        )
        self.wi_1 = Linear(
            in_dim=config.d_model,
            out_dim=config.d_ff,
            dtype=dtype,
            device=device,
            has_bias=False,
        )
        self.wo = Linear(
            in_dim=config.d_ff,
            out_dim=config.d_model,
            dtype=dtype,
            device=device,
            has_bias=False,
        )
        act_name = config.dense_act_fn
        if act_name == "relu":
            self._act = "relu"
        elif act_name in ("gelu_new", "gelu"):
            self._act = act_name
        else:
            raise ValueError(f"Unsupported UMT5 dense_act_fn: {act_name}")

    def __call__(self, hidden_states: TensorValue) -> TensorValue:
        gated = self.wi_0(hidden_states)
        if self._act == "relu":
            gated = ops.relu(gated)
        elif self._act == "gelu_new":
            gated = ops.gelu(gated, approximate="tanh")
        else:
            gated = ops.gelu(gated)
        linear = self.wi_1(hidden_states)
        return self.wo(gated * linear)


class UMT5LayerFF(Module):
    def __init__(
        self,
        config: UMT5ConfigBase,
        *,
        dtype: DType,
        device: DeviceRef,
    ) -> None:
        super().__init__()
        if config.is_gated_act:
            self.DenseReluDense: UMT5DenseGatedActDense | UMT5DenseActDense = (
                UMT5DenseGatedActDense(config, dtype=dtype, device=device)
            )
        else:
            self.DenseReluDense = UMT5DenseActDense(
                config, dtype=dtype, device=device
            )
        self.layer_norm = UMT5LayerNorm(
            config.d_model,
            eps=config.layer_norm_epsilon,
            dtype=config.dtype,
            device=device,
        )

    def __call__(self, hidden_states: TensorValue) -> TensorValue:
        forwarded = self.layer_norm(hidden_states)
        forwarded = self.DenseReluDense(forwarded)
        return hidden_states + forwarded


class UMT5Attention(Module):
    def __init__(
        self,
        config: UMT5ConfigBase,
        has_relative_attention_bias: bool = False,
        *,
        dtype: DType,
        device: DeviceRef,
    ) -> None:
        super().__init__()
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
        self.inner_dim = self.n_heads * self.key_value_proj_dim
        self._dtype = config.dtype
        self._device = device

        self.q = Linear(
            in_dim=self.d_model,
            out_dim=self.inner_dim,
            dtype=dtype,
            device=device,
            has_bias=False,
        )
        self.k = Linear(
            in_dim=self.d_model,
            out_dim=self.inner_dim,
            dtype=dtype,
            device=device,
            has_bias=False,
        )
        self.v = Linear(
            in_dim=self.d_model,
            out_dim=self.inner_dim,
            dtype=dtype,
            device=device,
            has_bias=False,
        )
        self.o = Linear(
            in_dim=self.inner_dim,
            out_dim=self.d_model,
            dtype=dtype,
            device=device,
            has_bias=False,
        )

        if self.has_relative_attention_bias:
            self.relative_attention_bias = Embedding(
                vocab_size=self.relative_attention_num_buckets,
                hidden_dim=self.n_heads,
                dtype=dtype,
                device=device,
            )

    def _relative_position_bucket(
        self,
        relative_position: TensorValue,
    ) -> TensorValue:
        num_buckets = self.relative_attention_num_buckets
        max_distance = self.relative_attention_max_distance

        dev = self._device
        relative_buckets = ops.constant(0, dtype=DType.int32, device=dev)
        relative_buckets = ops.broadcast_to(
            relative_buckets, relative_position.shape
        )

        if not self.is_decoder:
            num_buckets = num_buckets // 2
            is_positive = ops.greater(relative_position, 0)
            relative_buckets = relative_buckets + (
                ops.cast(is_positive, DType.int32) * num_buckets
            )
            relative_position = ops.abs(relative_position)
        else:
            relative_position = -ops.min(relative_position, 0)

        max_exact = num_buckets // 2
        is_small = ops.greater(
            ops.constant(max_exact, dtype=DType.int32, device=dev).broadcast_to(
                relative_position.shape
            ),
            relative_position,
        )

        scale = (num_buckets - max_exact) / math.log(max_distance / max_exact)
        rel_pos_float = ops.cast(relative_position, DType.float32)
        val_log = ops.log(rel_pos_float / float(max_exact))
        relative_position_if_large = (
            ops.cast(val_log * scale, DType.int32) + max_exact
        )
        max_val = ops.constant(
            num_buckets - 1, dtype=DType.int32, device=dev
        ).broadcast_to(relative_position_if_large.shape)
        relative_position_if_large = ops.where(
            ops.greater(relative_position_if_large, max_val),
            max_val,
            relative_position_if_large,
        )

        return relative_buckets + ops.where(
            is_small, relative_position, relative_position_if_large
        )

    def _compute_bias(
        self,
        query_length: int | Dim,
        key_length: int | Dim,
    ) -> TensorValue:
        context_position = ops.range(
            0,
            query_length,
            1,
            dtype=DType.int32,
            device=self._device,
        )
        context_position = ops.unsqueeze(context_position, 1)

        memory_position = ops.range(
            0,
            key_length,
            1,
            dtype=DType.int32,
            device=self._device,
        )
        memory_position = ops.unsqueeze(memory_position, 0)
        relative_position = memory_position - context_position

        relative_position_bucket = self._relative_position_bucket(
            relative_position
        )
        values = self.relative_attention_bias(relative_position_bucket)
        # values: [query_length, key_length, n_heads]
        values = ops.permute(values, [2, 0, 1])
        values = ops.unsqueeze(values, 0)
        # values: [1, n_heads, query_length, key_length]
        return values

    def __call__(
        self,
        hidden_states: TensorValue,
        attention_mask: TensorValue | None = None,
    ) -> TensorValue:
        batch_size = hidden_states.shape[0]
        seq_length = hidden_states.shape[1]

        query_states = self.q(hidden_states)
        key_states = self.k(hidden_states)
        value_states = self.v(hidden_states)

        query_states = ops.reshape(
            query_states,
            [batch_size, seq_length, self.n_heads, self.key_value_proj_dim],
        )
        key_states = ops.reshape(
            key_states,
            [batch_size, seq_length, self.n_heads, self.key_value_proj_dim],
        )
        value_states = ops.reshape(
            value_states,
            [batch_size, seq_length, self.n_heads, self.key_value_proj_dim],
        )

        # [B, S, H, D] -> [B, H, S, D]
        query_states = ops.permute(query_states, [0, 2, 1, 3])
        key_states = ops.permute(key_states, [0, 2, 1, 3])
        value_states = ops.permute(value_states, [0, 2, 1, 3])

        # scores: [B, H, S, S]
        scores = query_states @ ops.permute(key_states, [0, 1, 3, 2])

        if self.has_relative_attention_bias:
            position_bias = self._compute_bias(seq_length, seq_length)
            scores = scores + position_bias

        if attention_mask is not None:
            scores = scores + attention_mask

        attn_weights = ops.softmax(ops.cast(scores, DType.float32))
        attn_weights = ops.cast(attn_weights, self._dtype)
        attn_output = attn_weights @ value_states

        # [B, H, S, D] -> [B, S, H, D] -> [B, S, inner_dim]
        attn_output = ops.permute(attn_output, [0, 2, 1, 3])
        attn_output = ops.reshape(
            attn_output, [batch_size, seq_length, self.inner_dim]
        )
        return self.o(attn_output)


class UMT5LayerSelfAttention(Module):
    def __init__(
        self,
        config: UMT5ConfigBase,
        layer_idx: int | None = None,
        *,
        dtype: DType,
        device: DeviceRef,
    ) -> None:
        super().__init__()
        del layer_idx
        self.SelfAttention = UMT5Attention(
            config,
            has_relative_attention_bias=True,
            dtype=dtype,
            device=device,
        )
        self.layer_norm = UMT5LayerNorm(
            config.d_model,
            eps=config.layer_norm_epsilon,
            dtype=config.dtype,
            device=device,
        )

    def __call__(
        self,
        hidden_states: TensorValue,
        attention_mask: TensorValue | None = None,
    ) -> TensorValue:
        normed = self.layer_norm(hidden_states)
        attn_output = self.SelfAttention(normed, attention_mask)
        return hidden_states + attn_output


class UMT5Block(Module):
    def __init__(
        self,
        config: UMT5ConfigBase,
        layer_idx: int | None = None,
        *,
        dtype: DType,
        device: DeviceRef,
    ) -> None:
        super().__init__()
        # Use LayerList with index 0 = self-attn, index 1 = FF
        # to match HF weight key paths: block.{i}.layer.0 / block.{i}.layer.1
        self.layer = LayerList(
            [
                UMT5LayerSelfAttention(
                    config, layer_idx=layer_idx, dtype=dtype, device=device
                ),
                UMT5LayerFF(config, dtype=dtype, device=device),
            ]
        )

    def __call__(
        self,
        hidden_states: TensorValue,
        attention_mask: TensorValue | None = None,
    ) -> TensorValue:
        hidden_states = self.layer[0](hidden_states, attention_mask)
        hidden_states = self.layer[1](hidden_states)
        return hidden_states


class UMT5Stack(Module):
    def __init__(
        self,
        config: UMT5ConfigBase,
        embed_tokens: Embedding,
        *,
        dtype: DType,
        device: DeviceRef,
    ) -> None:
        super().__init__()
        self.embed_tokens = embed_tokens
        self.block = LayerList(
            [
                UMT5Block(config, layer_idx=i, dtype=dtype, device=device)
                for i in range(config.num_layers)
            ]
        )
        self.final_layer_norm = UMT5LayerNorm(
            config.d_model,
            eps=config.layer_norm_epsilon,
            dtype=config.dtype,
            device=device,
        )
        self._dtype = config.dtype

    def __call__(
        self,
        input_ids: TensorValue,
        attention_mask: TensorValue | None = None,
    ) -> TensorValue:
        hidden_states = self.embed_tokens(input_ids)

        # Build causal mask from attention_mask [B, S]
        causal_mask: TensorValue | None = None
        if attention_mask is not None:
            # (1 - mask) * dtype_min → masked positions get large negative
            _DTYPE_MIN: dict[DType, float] = {
                DType.float16: -65504.0,
                DType.bfloat16: -3.3895314e38,
                DType.float32: -3.4028235e38,
            }
            mask_min = _DTYPE_MIN.get(self._dtype, -3.4028235e38)
            mask_float = ops.cast(attention_mask, hidden_states.dtype)
            causal_mask = (1.0 - mask_float) * mask_min
            # [B, S] -> [B, 1, 1, S] for broadcasting with [B, H, S, S]
            causal_mask = ops.unsqueeze(ops.unsqueeze(causal_mask, 1), 1)

        for block in self.block:
            hidden_states = block(hidden_states, causal_mask)

        hidden_states = self.final_layer_norm(hidden_states)
        return hidden_states


class UMT5EncoderModel(Module):
    """UMT5 encoder using module_v2 (ops-based) API."""

    def __init__(
        self,
        config: UMT5ConfigBase,
        *,
        dtype: DType,
        device: DeviceRef,
    ) -> None:
        super().__init__()
        # Parse feed_forward config
        act_info = config.feed_forward_proj.split("-")
        config.dense_act_fn = act_info[-1]
        config.is_gated_act = act_info[0] == "gated"
        if (len(act_info) > 1 and act_info[0] != "gated") or len(act_info) > 2:
            raise ValueError(
                f"`feed_forward_proj`: {config.feed_forward_proj} is not valid."
            )
        if config.feed_forward_proj == "gated-gelu":
            config.dense_act_fn = "gelu_new"

        self.shared = Embedding(
            vocab_size=config.vocab_size,
            hidden_dim=config.d_model,
            dtype=dtype,
            device=device,
        )

        encoder_config = copy.deepcopy(config)
        encoder_config.is_decoder = False
        encoder_config.use_cache = False
        encoder_config.is_encoder_decoder = False
        self.encoder = UMT5Stack(
            encoder_config, self.shared, dtype=dtype, device=device
        )

    def __call__(
        self,
        input_ids: TensorValue,
        attention_mask: TensorValue | None = None,
    ) -> TensorValue:
        return self.encoder(input_ids, attention_mask)


__all__ = ["UMT5EncoderModel"]
