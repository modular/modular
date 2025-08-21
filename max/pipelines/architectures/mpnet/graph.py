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

import math
from collections.abc import Mapping

import numpy as np
from max.driver import DLPackArray
from max.dtype import DType
from max.graph import DeviceRef, Graph, TensorType, TensorValue, ops
from max.graph.quantization import QuantizationEncoding
from max.graph.weights import WeightData
from max.nn import Embedding, LayerNorm, Linear, Sequential
from max.nn.layer import Module
from max.pipelines.lib import PipelineConfig
from transformers import AutoConfig


def _quantization_encoding(
    pipeline_config: PipelineConfig,
) -> QuantizationEncoding | None:
    if supported_encoding := pipeline_config.model_config.quantization_encoding:
        return supported_encoding.quantization_encoding
    return None


class MPNetEmbeddings(Module):
    """An embeddings layer that combines the tokens embeddings and positions
    embeddings."""

    def __init__(
        self,
        pipeline_config: PipelineConfig,
        huggingface_config: AutoConfig,
        dtype: DType,
        device: DeviceRef,
    ) -> None:
        config = self.config = huggingface_config
        self.word_embeddings = Embedding(
            config.vocab_size,
            config.hidden_size,
            dtype,
            device,
            _quantization_encoding(pipeline_config),
        )
        self.position_embeddings = Embedding(
            config.max_position_embeddings, config.hidden_size, dtype, device
        )
        self.layer_norm = LayerNorm(
            config.hidden_size,
            device,
            DType.float32,
            eps=config.layer_norm_eps,
            use_bias=True,
        )

    def __call__(self, input_ids: TensorValue) -> TensorValue:
        position_ids = _create_position_ids_from_input_ids(
            input_ids, self.config.pad_token_id
        )
        inputs_embeds = self.word_embeddings(input_ids)
        position_embeddings = self.position_embeddings(position_ids)
        embeddings = inputs_embeds + position_embeddings
        return self.layer_norm(embeddings)


def _create_position_ids_from_input_ids(
    input_ids: TensorValue, padding_idx: int
) -> TensorValue:
    mask = (input_ids != padding_idx).cast(DType.int64)
    incremental_indices = ops.cumsum(mask, axis=1) * mask
    return incremental_indices + padding_idx


class MPNetSelfAttention(Module):
    """Self-attention layer with position compensation."""

    def __init__(
        self,
        pipeline_config: PipelineConfig,
        huggingface_config: AutoConfig,
        dtype: DType,
        device: DeviceRef,
    ) -> None:
        config = huggingface_config
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(
            config.hidden_size / config.num_attention_heads
        )
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.q = Linear(
            config.hidden_size,
            self.all_head_size,
            dtype,
            device,
            has_bias=True,
            quantization_encoding=_quantization_encoding(pipeline_config),
        )
        self.k = Linear(
            config.hidden_size,
            self.all_head_size,
            dtype,
            device,
            has_bias=True,
            quantization_encoding=_quantization_encoding(pipeline_config),
        )
        self.v = Linear(
            config.hidden_size,
            self.all_head_size,
            dtype,
            device,
            has_bias=True,
            quantization_encoding=_quantization_encoding(pipeline_config),
        )
        self.o = Linear(
            config.hidden_size,
            config.hidden_size,
            dtype,
            device,
            has_bias=True,
            quantization_encoding=_quantization_encoding(pipeline_config),
        )

    def transpose_for_scores(self, x: TensorValue) -> TensorValue:
        new_x_shape = x.shape[:-1] + [
            self.num_attention_heads,
            self.attention_head_size,
        ]
        x = ops.reshape(x, new_x_shape)
        return ops.permute(x, [0, 2, 1, 3])

    def __call__(
        self,
        hidden_states: TensorValue,
        attention_mask: TensorValue,
        position_bias: TensorValue,
    ) -> TensorValue:
        q = self.q(hidden_states)
        k = self.k(hidden_states)
        v = self.v(hidden_states)

        q = self.transpose_for_scores(q)
        k = self.transpose_for_scores(k)
        v = self.transpose_for_scores(v)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = q @ k.transpose(-1, -2)
        attention_scores = attention_scores / math.sqrt(
            self.attention_head_size
        )

        # Apply relative position embedding (precomputed in MPNetEncoder).
        attention_scores += position_bias

        attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = ops.softmax(attention_scores)

        c = attention_probs @ v

        c = ops.permute(c, [0, 2, 1, 3])
        new_c_shape = c.shape[:-2] + [self.all_head_size]
        c = ops.reshape(c, new_c_shape)

        return self.o(c)


class MPNetAttention(Module):
    """Container for the attention and attention output layer norm layers."""

    def __init__(
        self,
        pipeline_config: PipelineConfig,
        huggingface_config: AutoConfig,
        dtype: DType,
        device: DeviceRef,
    ) -> None:
        config = huggingface_config
        self.attn = MPNetSelfAttention(
            pipeline_config, huggingface_config, dtype, device
        )
        self.layer_norm = LayerNorm(
            config.hidden_size,
            device,
            DType.float32,
            eps=config.layer_norm_eps,
            use_bias=True,
        )

    def __call__(
        self,
        hidden_states: TensorValue,
        attention_mask: TensorValue,
        position_bias: TensorValue,
    ) -> TensorValue:
        attn_output = self.attn(hidden_states, attention_mask, position_bias)
        return self.layer_norm(attn_output + hidden_states)


_ACTIVATIONS = {
    "gelu": ops.gelu,
    "relu": ops.relu,
    "silu": ops.silu,
    "sigmoid": ops.sigmoid,
    "tanh": ops.tanh,
}


class MPNetIntermediate(Module):
    """Fully connected layer with an activation function."""

    def __init__(
        self,
        pipeline_config: PipelineConfig,
        huggingface_config: AutoConfig,
        dtype: DType,
        device: DeviceRef,
    ) -> None:
        config = huggingface_config
        self.dense = Linear(
            config.hidden_size,
            config.intermediate_size,
            dtype,
            device,
            has_bias=True,
            quantization_encoding=_quantization_encoding(pipeline_config),
        )
        self.intermediate_act_fn = _ACTIVATIONS[config.hidden_act]

    def __call__(self, hidden_states: TensorValue) -> TensorValue:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states


class MPNetOutput(Module):
    """Layer that combines the outputs of the intermediate and attention layers."""

    def __init__(
        self,
        pipeline_config: PipelineConfig,
        huggingface_config: AutoConfig,
        dtype: DType,
        device: DeviceRef,
    ) -> None:
        config = huggingface_config
        self.dense = Linear(
            config.intermediate_size,
            config.hidden_size,
            dtype,
            device,
            has_bias=True,
            quantization_encoding=_quantization_encoding(pipeline_config),
        )
        self.layer_norm = LayerNorm(
            config.hidden_size,
            device,
            DType.float32,
            eps=config.layer_norm_eps,
        )

    def __call__(
        self, hidden_states: TensorValue, input_tensor: TensorValue
    ) -> TensorValue:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.layer_norm(hidden_states + input_tensor)
        return hidden_states


class MPNetLayer(Module):
    """An Encoder layer block."""

    def __init__(
        self,
        pipeline_config: PipelineConfig,
        huggingface_config: AutoConfig,
        dtype: DType,
        device: DeviceRef,
    ) -> None:
        self.attention = MPNetAttention(
            pipeline_config, huggingface_config, dtype, device
        )
        self.intermediate = MPNetIntermediate(
            pipeline_config, huggingface_config, dtype, device
        )
        self.output = MPNetOutput(
            pipeline_config, huggingface_config, dtype, device
        )

    def __call__(
        self,
        hidden_states: TensorValue,
        attention_mask: TensorValue,
        position_bias: TensorValue,
    ) -> TensorValue:
        attention_output = self.attention(
            hidden_states, attention_mask, position_bias=position_bias
        )
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        return layer_output


class MPNetEncoder(Module):
    """Encoder that contains stacks of MPNetLayers."""

    def __init__(
        self,
        pipeline_config: PipelineConfig,
        huggingface_config: AutoConfig,
        dtype: DType,
        device: DeviceRef,
    ) -> None:
        config = self.config = huggingface_config
        self.n_heads = config.num_attention_heads
        num_hidden_layers = config.num_hidden_layers
        self.layer = Sequential(
            [
                MPNetLayer(pipeline_config, huggingface_config, dtype, device)
                for _ in range(num_hidden_layers)
            ]
        )
        self.relative_attention_bias = Embedding(
            config.relative_attention_num_buckets,
            config.num_attention_heads,
            dtype,
            device,
        )
        self.num_attention_heads = config.num_attention_heads

    def __call__(
        self, hidden_states: TensorValue, attention_mask: TensorValue
    ) -> TensorValue:
        position_bias = self.compute_position_bias(hidden_states)
        for layer in self.layer.layers:
            hidden_states = layer(hidden_states, attention_mask, position_bias)
        return hidden_states

    def compute_position_bias(self, hidden_states: TensorValue) -> TensorValue:
        shape = hidden_states.shape
        bsz, qlen, klen = shape[0], shape[1], shape[1]
        context_position = ops.range(
            0, qlen, dtype=DType.int64, device=DeviceRef.CPU()
        )[:, None]
        memory_position = ops.range(
            0, klen, dtype=DType.int64, device=DeviceRef.CPU()
        )[None, :]
        relative_position = memory_position - context_position
        rp_bucket = self.relative_position_bucket(
            relative_position,
            num_buckets=self.config.relative_attention_num_buckets,
        ).to(self.relative_attention_bias.device)
        values = self.relative_attention_bias(rp_bucket)
        values = ops.unsqueeze(ops.permute(values, [2, 0, 1]), 0)
        values = ops.broadcast_to(
            values, [bsz, self.num_attention_heads, qlen, klen]
        )
        return values

    @staticmethod
    def relative_position_bucket(
        relative_position: TensorValue,
        num_buckets: int = 32,
        max_distance: int = 128,
    ) -> TensorValue:
        n = -relative_position

        num_buckets //= 2
        ret = (n < 0).cast(DType.int64) * num_buckets
        n = ops.abs(n)

        max_exact = num_buckets // 2
        is_small = n < max_exact

        val_if_large = max_exact + ops.cast(
            ops.log(ops.cast(n, DType.float32) / max_exact)
            / math.log(max_distance / max_exact)
            * (num_buckets - max_exact),
            DType.int64,
        )

        # Roundabout implementation of full_like(val_if_large, num_buckets - 1).
        max_bucket = ops.broadcast_to(
            ops.constant(num_buckets - 1, DType.int64, device=DeviceRef.CPU()),
            val_if_large.shape,
        )

        val_if_large = ops.min(val_if_large, max_bucket)
        ret += ops.where(is_small, n, val_if_large)
        return ret


class MPNetModel(Module):
    """The MPNet encoder model.

    Based on the MPNetModel transformers implementation."""

    def __init__(
        self,
        pipeline_config: PipelineConfig,
        huggingface_config: AutoConfig,
        dtype: DType,
        device: DeviceRef,
    ) -> None:
        self.embeddings = MPNetEmbeddings(
            pipeline_config,
            huggingface_config=huggingface_config,
            dtype=dtype,
            device=device,
        )
        self.encoder = MPNetEncoder(
            pipeline_config,
            huggingface_config=huggingface_config,
            dtype=dtype,
            device=device,
        )
        self.pool_outputs = pipeline_config.pool_embeddings

    def __call__(
        self, input_ids: TensorValue, attention_mask: TensorValue
    ) -> TensorValue:
        embedding_output = self.embeddings(input_ids=input_ids)
        extended_attention_mask = ops.reshape(
            attention_mask, ("batch_size", 1, 1, "seq_len")
        )
        extended_attention_mask = (1 - extended_attention_mask) * ops.constant(
            np.finfo(np.float32).min,
            DType.float32,
            device=attention_mask.device,
        )
        encoded_results = self.encoder(
            embedding_output, attention_mask=extended_attention_mask
        )
        if self.pool_outputs:
            # Pool the embeddings.
            # TODO(KERN-1550): Since GPU can only apply reductions along the
            # inner-most dimension, transpose the mask so the seq_len is last.
            encoded_results = encoded_results.transpose(1, 2)
            input_mask_expanded = ops.broadcast_to(
                ops.unsqueeze(attention_mask, 1),
                ("batch_size", encoded_results.shape[1], "seq_len"),
            )
            input_lengths = ops.max(
                ops.sum(input_mask_expanded),
                ops.constant(
                    1e-9, DType.float32, device=input_mask_expanded.device
                ),
            )
            pooled_output = (
                ops.sum(encoded_results * input_mask_expanded) / input_lengths
            )
            return ops.squeeze(pooled_output, 2)
        else:
            return encoded_results


def build_graph(
    pipeline_config: PipelineConfig,
    state_dict: Mapping[str, DLPackArray | WeightData],
    huggingface_config: AutoConfig,
    dtype: DType,
    input_device: DeviceRef,
) -> Graph:
    # Graph input types.
    input_ids_type = TensorType(
        DType.int64, shape=["batch_size", "seq_len"], device=input_device
    )
    attention_mask_type = TensorType(
        DType.float32, shape=["batch_size", "seq_len"], device=input_device
    )

    with Graph(
        "mpnet", input_types=[input_ids_type, attention_mask_type]
    ) as graph:
        mpnet = MPNetModel(
            pipeline_config,
            huggingface_config,
            dtype,
            device=input_device,
        )
        mpnet.load_state_dict(state_dict)
        input_ids = graph.inputs[0].tensor
        attention_mask = graph.inputs[1].tensor
        graph.output(mpnet(input_ids, attention_mask))

    return graph
