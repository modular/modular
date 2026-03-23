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
from __future__ import annotations

import math

import numpy as np
from max.dtype import DType
from max.experimental import functional as F
from max.experimental.nn import Embedding, LayerNorm, Linear, Module, ModuleList
from max.experimental.nn.common_layers.activation import (
    activation_function_from_name,
)
from max.experimental.tensor import Tensor
from max.graph import DeviceRef

from .model_config import MPNetConfig


class MPNetEmbeddings(Module[[Tensor], Tensor]):
    """Combines token embeddings and position embeddings."""

    def __init__(self, config: MPNetConfig) -> None:
        super().__init__()
        hf_config = self.config = config.huggingface_config
        self.pad_token_id = hf_config.pad_token_id
        self.word_embeddings = Embedding(
            hf_config.vocab_size,
            dim=hf_config.hidden_size,
        )
        self.position_embeddings = Embedding(
            hf_config.max_position_embeddings,
            dim=hf_config.hidden_size,
        )
        self.layer_norm = LayerNorm(
            dim=hf_config.hidden_size,
            eps=hf_config.layer_norm_eps,
            keep_dtype=False,
        )

    def forward(self, input_ids: Tensor) -> Tensor:
        position_ids = _create_position_ids_from_input_ids(
            input_ids, self.pad_token_id
        )
        inputs_embeds = self.word_embeddings(input_ids)
        position_embeddings = self.position_embeddings(position_ids)
        embeddings = inputs_embeds + position_embeddings
        return self.layer_norm(embeddings)


def _create_position_ids_from_input_ids(
    input_ids: Tensor, padding_idx: int
) -> Tensor:
    mask = F.cast(input_ids != padding_idx, DType.int64)
    incremental_indices = F.cumsum(mask, axis=1) * mask
    return incremental_indices + padding_idx


class MPNetSelfAttention(Module[[Tensor, Tensor, Tensor], Tensor]):
    """Self-attention layer with position compensation."""

    def __init__(self, config: MPNetConfig) -> None:
        super().__init__()
        hf_config = config.huggingface_config
        self.num_attention_heads = hf_config.num_attention_heads
        self.attention_head_size = int(
            hf_config.hidden_size / hf_config.num_attention_heads
        )
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.q = Linear(
            in_dim=hf_config.hidden_size,
            out_dim=self.all_head_size,
            bias=True,
        )
        self.k = Linear(
            in_dim=hf_config.hidden_size,
            out_dim=self.all_head_size,
            bias=True,
        )
        self.v = Linear(
            in_dim=hf_config.hidden_size,
            out_dim=self.all_head_size,
            bias=True,
        )
        self.o = Linear(
            in_dim=hf_config.hidden_size,
            out_dim=hf_config.hidden_size,
            bias=True,
        )

    def transpose_for_scores(self, x: Tensor) -> Tensor:
        new_x_shape = x.shape[:-1] + [
            self.num_attention_heads,
            self.attention_head_size,
        ]
        x = F.reshape(x, new_x_shape)
        return F.permute(x, [0, 2, 1, 3])

    def forward(
        self,
        hidden_states: Tensor,
        attention_mask: Tensor,
        position_bias: Tensor,
    ) -> Tensor:
        q = self.q(hidden_states)
        k = self.k(hidden_states)
        v = self.v(hidden_states)

        q = self.transpose_for_scores(q)
        k = self.transpose_for_scores(k)
        v = self.transpose_for_scores(v)

        # Take the dot product between "query" and "key" to get the raw
        # attention scores.
        attention_scores = q @ F.transpose(k, -1, -2)
        attention_scores = attention_scores / math.sqrt(
            self.attention_head_size
        )

        # Apply relative position embedding (precomputed in MPNetEncoder).
        attention_scores = attention_scores + position_bias

        attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = F.softmax(attention_scores)

        c = attention_probs @ v

        c = F.permute(c, [0, 2, 1, 3])
        new_c_shape = c.shape[:-2] + [self.all_head_size]
        c = F.reshape(c, new_c_shape)

        return self.o(c)


class MPNetAttention(Module[[Tensor, Tensor, Tensor], Tensor]):
    """Container for the attention and attention output layer norm layers."""

    def __init__(self, config: MPNetConfig) -> None:
        super().__init__()
        self.attn = MPNetSelfAttention(config)
        self.layer_norm = LayerNorm(
            dim=config.huggingface_config.hidden_size,
            eps=config.huggingface_config.layer_norm_eps,
            keep_dtype=False,
        )

    def forward(
        self,
        hidden_states: Tensor,
        attention_mask: Tensor,
        position_bias: Tensor,
    ) -> Tensor:
        attn_output = self.attn(hidden_states, attention_mask, position_bias)
        return self.layer_norm(attn_output + hidden_states)


class MPNetIntermediate(Module[[Tensor], Tensor]):
    """Fully connected layer with an activation function."""

    def __init__(self, config: MPNetConfig) -> None:
        super().__init__()
        hf_config = config.huggingface_config
        self.dense = Linear(
            in_dim=hf_config.hidden_size,
            out_dim=hf_config.intermediate_size,
            bias=True,
        )
        self.intermediate_act_fn = activation_function_from_name(
            hf_config.hidden_act
        )

    def forward(self, hidden_states: Tensor) -> Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states


class MPNetOutput(Module[[Tensor, Tensor], Tensor]):
    """Combines the outputs of the intermediate and attention layers."""

    def __init__(self, config: MPNetConfig) -> None:
        super().__init__()
        hf_config = config.huggingface_config
        self.dense = Linear(
            in_dim=hf_config.intermediate_size,
            out_dim=hf_config.hidden_size,
            bias=True,
        )
        self.layer_norm = LayerNorm(
            dim=hf_config.hidden_size,
            eps=hf_config.layer_norm_eps,
            keep_dtype=False,
        )

    def forward(self, hidden_states: Tensor, input_tensor: Tensor) -> Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.layer_norm(hidden_states + input_tensor)
        return hidden_states


class MPNetLayer(Module[[Tensor, Tensor, Tensor], Tensor]):
    """An Encoder layer block."""

    def __init__(self, config: MPNetConfig) -> None:
        super().__init__()
        self.attention = MPNetAttention(config)
        self.intermediate = MPNetIntermediate(config)
        self.output = MPNetOutput(config)

    def forward(
        self,
        hidden_states: Tensor,
        attention_mask: Tensor,
        position_bias: Tensor,
    ) -> Tensor:
        attention_output = self.attention(
            hidden_states, attention_mask, position_bias
        )
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        return layer_output


class MPNetEncoder(Module[[Tensor, Tensor], Tensor]):
    """Encoder that contains stacks of MPNetLayers."""

    def __init__(self, config: MPNetConfig) -> None:
        super().__init__()
        hf_config = self.config = config.huggingface_config
        self.n_heads = hf_config.num_attention_heads
        num_hidden_layers = hf_config.num_hidden_layers
        self.layer = ModuleList(
            [MPNetLayer(config) for _ in range(num_hidden_layers)]
        )
        self.relative_attention_bias = Embedding(
            hf_config.relative_attention_num_buckets,
            dim=hf_config.num_attention_heads,
        )
        self.num_attention_heads = hf_config.num_attention_heads

    def forward(self, hidden_states: Tensor, attention_mask: Tensor) -> Tensor:
        position_bias = self.compute_position_bias(hidden_states)
        for layer in self.layer:
            hidden_states = layer(hidden_states, attention_mask, position_bias)
        return hidden_states

    def compute_position_bias(self, hidden_states: Tensor) -> Tensor:
        shape = hidden_states.shape
        bsz, qlen, klen = shape[0], shape[1], shape[1]
        context_position = F.unsqueeze(
            F.arange(0, qlen, dtype=DType.int64, device=DeviceRef.CPU()), 1
        )
        memory_position = F.unsqueeze(
            F.arange(0, klen, dtype=DType.int64, device=DeviceRef.CPU()), 0
        )
        relative_position = memory_position - context_position
        rp_bucket = self.relative_position_bucket(
            relative_position,
            num_buckets=self.config.relative_attention_num_buckets,
        )
        rp_bucket = F.transfer_to(
            rp_bucket, self.relative_attention_bias.weight.device
        )
        values = self.relative_attention_bias(rp_bucket)
        values = F.unsqueeze(F.permute(values, [2, 0, 1]), 0)
        values = F.broadcast_to(
            values, [bsz, self.num_attention_heads, qlen, klen]
        )
        return values

    @staticmethod
    def relative_position_bucket(
        relative_position: Tensor,
        num_buckets: int = 32,
        max_distance: int = 128,
    ) -> Tensor:
        n = -relative_position

        num_buckets //= 2
        ret = F.cast(n < 0, DType.int64) * num_buckets
        n = F.abs(n)

        max_exact = num_buckets // 2
        is_small = n < max_exact

        val_if_large = max_exact + F.cast(
            F.log(F.cast(n, DType.float32) / max_exact)
            / math.log(max_distance / max_exact)
            * (num_buckets - max_exact),
            DType.int64,
        )

        # Roundabout implementation of full_like(val_if_large, num_buckets - 1).
        max_bucket = F.broadcast_to(
            F.constant(num_buckets - 1, DType.int64, device=DeviceRef.CPU()),
            val_if_large.shape,
        )

        val_if_large = F.min(val_if_large, max_bucket)
        ret = ret + F.where(is_small, n, val_if_large)
        return ret


class MPNetModel(Module[[Tensor, Tensor], tuple[Tensor, ...]]):
    """The MPNet encoder model.

    Based on the MPNetModel transformers implementation."""

    def __init__(self, config: MPNetConfig) -> None:
        super().__init__()
        self.embeddings = MPNetEmbeddings(config)
        self.encoder = MPNetEncoder(config)
        self.pool_outputs = config.pool_embeddings

    def forward(
        self, input_ids: Tensor, attention_mask: Tensor
    ) -> tuple[Tensor, ...]:
        embedding_output = self.embeddings(input_ids)
        extended_attention_mask = F.reshape(
            attention_mask, ("batch_size", 1, 1, "seq_len")
        )
        extended_attention_mask = (1 - extended_attention_mask) * F.constant(
            np.finfo(np.float32).min,
            DType.float32,
            device=attention_mask.device,
        )
        encoded_results = self.encoder(
            embedding_output, extended_attention_mask
        )
        if self.pool_outputs:
            # Pool the embeddings.
            encoded_results = F.transpose(encoded_results, 1, 2)
            input_mask_expanded = F.broadcast_to(
                F.unsqueeze(attention_mask, 1),
                ("batch_size", encoded_results.shape[1], "seq_len"),
            )
            input_lengths = F.max(
                F.sum(input_mask_expanded),
                F.constant(
                    1e-9, DType.float32, device=input_mask_expanded.device
                ),
            )
            pooled_output = (
                F.sum(encoded_results * input_mask_expanded) / input_lengths
            )
            return (F.squeeze(pooled_output, 2),)
        else:
            return (encoded_results,)
