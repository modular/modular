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
"""Graph construction for Bert-based sentence transformer.

This module builds the computation graph for Bert-based models,
which are BERT models for sentence embeddings. Unlike MPNet,
Bert uses:
- Absolute position embeddings (not relative)
- Token type embeddings (typically all zeros for sentence transformers)
- Standard scaled dot-product attention without position bias
"""

from __future__ import annotations

import math
from collections.abc import Mapping

import numpy as np
from max.driver import DLPackArray
from max.dtype import DType
from max.graph import DeviceRef, Graph, TensorType, TensorValue, ops
from max.graph.weights import WeightData
from max.nn.legacy.activation import activation_function_from_name
from max.nn.legacy.embedding import Embedding
from max.nn.legacy.layer import Module
from max.nn.legacy.linear import Linear
from max.nn.legacy.norm import LayerNorm
from max.nn.legacy.sequential import Sequential

from .model_config import BertModelConfig


class BertEmbeddings(Module):
    def __init__(self, config: BertModelConfig) -> None:
        hf_config = config.huggingface_config
        dtype = config.dtype
        device = config.device
        self.word_embeddings = Embedding(
            hf_config.vocab_size,
            hf_config.hidden_size,
            dtype,
            device,
        )
        self.position_embeddings = Embedding(
            hf_config.max_position_embeddings,
            hf_config.hidden_size,
            dtype,
            device,
        )
        self.token_type_embeddings = Embedding(
            hf_config.type_vocab_size,
            hf_config.hidden_size,
            dtype,
            device,
        )
        self.layer_norm = LayerNorm(
            hf_config.hidden_size,
            devices=[device],
            dtype=DType.float32,
            eps=hf_config.layer_norm_eps,
            use_bias=True,
        )

    def __call__(
        self,
        input_ids: TensorValue,
        token_type_ids: TensorValue | None = None,
    ) -> TensorValue:
        seq_length = input_ids.shape[1]

        position_ids = ops.range(
            0, seq_length, dtype=DType.int64, device=DeviceRef.CPU()
        )
        position_ids = ops.unsqueeze(position_ids, 0)  # [1, seq_len]
        position_ids = position_ids.to(input_ids.device)

        inputs_embeds = self.word_embeddings(input_ids)
        position_embeddings = self.position_embeddings(position_ids)

        if token_type_ids is None:
            token_type_ids = ops.broadcast_to(
                ops.constant(0, DType.int64, device=input_ids.device),
                input_ids.shape,
            )
        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        embeddings = inputs_embeds + position_embeddings + token_type_embeddings
        return self.layer_norm(embeddings)


class BertSelfAttention(Module):
    def __init__(self, config: BertModelConfig) -> None:
        hf_config = config.huggingface_config
        dtype = config.dtype
        device = config.device
        self.num_attention_heads = hf_config.num_attention_heads
        self.attention_head_size = int(
            hf_config.hidden_size / hf_config.num_attention_heads
        )
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = Linear(
            hf_config.hidden_size,
            self.all_head_size,
            dtype,
            device,
            has_bias=True,
        )
        self.key = Linear(
            hf_config.hidden_size,
            self.all_head_size,
            dtype,
            device,
            has_bias=True,
        )
        self.value = Linear(
            hf_config.hidden_size,
            self.all_head_size,
            dtype,
            device,
            has_bias=True,
        )

    def transpose_for_scores(self, x: TensorValue) -> TensorValue:
        new_x_shape = x.shape[:-1] + [
            self.num_attention_heads,
            self.attention_head_size,
        ]
        x = ops.reshape(x, new_x_shape)
        return ops.permute(x, [0, 2, 1, 3])

    def __call__(
        self, hidden_states: TensorValue, attention_mask: TensorValue
    ) -> TensorValue:
        query_layer = self.transpose_for_scores(self.query(hidden_states))
        key_layer = self.transpose_for_scores(self.key(hidden_states))
        value_layer = self.transpose_for_scores(self.value(hidden_states))

        attention_scores = query_layer @ key_layer.transpose(-1, -2)
        attention_scores = attention_scores / math.sqrt(
            self.attention_head_size
        )

        attention_scores = attention_scores + attention_mask

        attention_probs = ops.softmax(attention_scores)

        context_layer = attention_probs @ value_layer

        context_layer = ops.permute(context_layer, [0, 2, 1, 3])
        new_context_shape = context_layer.shape[:-2] + [self.all_head_size]
        context_layer = ops.reshape(context_layer, new_context_shape)

        return context_layer


class BertSelfOutput(Module):
    def __init__(self, config: BertModelConfig) -> None:
        hf_config = config.huggingface_config
        dtype = config.dtype
        device = config.device
        self.dense = Linear(
            hf_config.hidden_size,
            hf_config.hidden_size,
            dtype,
            device,
            has_bias=True,
        )
        self.layer_norm = LayerNorm(
            hf_config.hidden_size,
            devices=[device],
            dtype=DType.float32,
            eps=hf_config.layer_norm_eps,
            use_bias=True,
        )

    def __call__(
        self, hidden_states: TensorValue, input_tensor: TensorValue
    ) -> TensorValue:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.layer_norm(hidden_states + input_tensor)
        return hidden_states


class BertAttention(Module):
    def __init__(self, config: BertModelConfig) -> None:
        self.self = BertSelfAttention(config)
        self.output = BertSelfOutput(config)

    def __call__(
        self, hidden_states: TensorValue, attention_mask: TensorValue
    ) -> TensorValue:
        self_output = self.self(hidden_states, attention_mask)
        attention_output = self.output(self_output, hidden_states)
        return attention_output


class BertIntermediate(Module):
    def __init__(self, config: BertModelConfig) -> None:
        hf_config = config.huggingface_config
        dtype = config.dtype
        device = config.device
        self.dense = Linear(
            hf_config.hidden_size,
            hf_config.intermediate_size,
            dtype,
            device,
            has_bias=True,
        )
        self.intermediate_act_fn = activation_function_from_name(
            hf_config.hidden_act
        )

    def __call__(self, hidden_states: TensorValue) -> TensorValue:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states


class BertOutput(Module):
    def __init__(self, config: BertModelConfig) -> None:
        hf_config = config.huggingface_config
        dtype = config.dtype
        device = config.device
        self.dense = Linear(
            hf_config.intermediate_size,
            hf_config.hidden_size,
            dtype,
            device,
            has_bias=True,
        )
        self.layer_norm = LayerNorm(
            hf_config.hidden_size,
            devices=[device],
            dtype=DType.float32,
            eps=hf_config.layer_norm_eps,
            use_bias=True,
        )

    def __call__(
        self, hidden_states: TensorValue, input_tensor: TensorValue
    ) -> TensorValue:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.layer_norm(hidden_states + input_tensor)
        return hidden_states


class BertLayer(Module):
    def __init__(
        self,
        config: BertModelConfig,
    ) -> None:
        self.attention = BertAttention(config)
        self.intermediate = BertIntermediate(config)
        self.output = BertOutput(config)

    def __call__(
        self,
        hidden_states: TensorValue,
        attention_mask: TensorValue,
    ) -> TensorValue:
        attention_output = self.attention(hidden_states, attention_mask)
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        return layer_output


class BertEncoder(Module):
    def __init__(
        self,
        config: BertModelConfig,
    ) -> None:
        hf_config = config.huggingface_config
        num_hidden_layers = hf_config.num_hidden_layers
        self.layer = Sequential(
            [BertLayer(config) for _ in range(num_hidden_layers)]
        )

    def __call__(
        self, hidden_states: TensorValue, attention_mask: TensorValue
    ) -> TensorValue:
        for layer in self.layer.layers:
            hidden_states = layer(hidden_states, attention_mask)
        return hidden_states


class BertModel(Module):
    def __init__(
        self,
        config: BertModelConfig,
    ) -> None:
        self.embeddings = BertEmbeddings(config)
        self.encoder = BertEncoder(config)
        self.pool_outputs = config.pool_embeddings

    def __call__(
        self,
        input_ids: TensorValue,
        attention_mask: TensorValue,
        token_type_ids: TensorValue | None = None,
    ) -> TensorValue:
        embedding_output = self.embeddings(
            input_ids=input_ids, token_type_ids=token_type_ids
        )

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
    config: BertModelConfig,
    state_dict: Mapping[str, DLPackArray | WeightData],
) -> Graph:
    input_ids_type = TensorType(
        DType.int64, shape=["batch_size", "seq_len"], device=config.device
    )
    attention_mask_type = TensorType(
        DType.float32, shape=["batch_size", "seq_len"], device=config.device
    )

    with Graph(
        "bert", input_types=[input_ids_type, attention_mask_type]
    ) as graph:
        model = BertModel(config)
        model.load_state_dict(state_dict)
        input_ids = graph.inputs[0].tensor
        attention_mask = graph.inputs[1].tensor
        graph.output(model(input_ids, attention_mask))

    return graph
