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
"""Embedding layers for transformer models."""

from max.dtype import DType
from max.graph import DeviceRef, TensorValue, ops
from max.graph.weights import Weights
from max.nn import EmbeddingV1, LayerNormV1, LinearV1, Module
from max.pipelines.lib import PipelineConfig
from transformers import AutoConfig

from .utils import _quantization_encoding


class EmbeddingLayer(Module):
    """Combined embedding layer for word, position, and token type embeddings."""

    def __init__(
        self,
        pipeline_config: PipelineConfig,
        weights: Weights,
        huggingface_config: AutoConfig,
        dtype: DType,
        device: DeviceRef,
    ) -> None:
        super().__init__()
        config = huggingface_config

        self.word_embeddings = EmbeddingV1(
            weights.word_embeddings.weight.allocate(
                DType.float32,
                [config.vocab_size, config.hidden_size],
                _quantization_encoding(pipeline_config),
            ).cast(dtype),
            device,
        )
        self.position_embeddings = EmbeddingV1(
            weights.position_embeddings.weight.allocate(
                DType.float32,
                [config.max_position_embeddings, config.hidden_size],
            ).cast(dtype),
            device,
        )
        self.token_type_embeddings = EmbeddingV1(
            weights.token_type_embeddings.weight.allocate(
                DType.float32,
                [config.type_vocab_size, config.hidden_size],
            ).cast(dtype),
            device,
        )
        self.layer_norm = LayerNormV1(
            weight=weights.LayerNorm.weight.allocate(
                DType.float32, [config.hidden_size]
            ),
            bias=weights.LayerNorm.bias.allocate(
                DType.float32, [config.hidden_size]
            ),
            eps=config.layer_norm_eps,
        )

    def __call__(
        self,
        input_ids: TensorValue,
        token_type_ids: TensorValue | None = None,
    ) -> TensorValue:
        """Forward pass through embedding layer."""
        seq_length = input_ids.shape[1]

        # Create position IDs
        start = ops.constant(0, DType.int64, device=DeviceRef.CPU())
        step = ops.constant(1, DType.int64, device=DeviceRef.CPU())
        position_ids = ops.range(
            start, seq_length, step, seq_length, device=DeviceRef.CPU()
        ).cast(DType.int64)
        position_ids = ops.broadcast_to(
            ops.unsqueeze(position_ids, 0), input_ids.shape
        )

        # Get embeddings
        inputs_embeds = self.word_embeddings(input_ids)
        position_embeddings = self.position_embeddings(position_ids)

        if token_type_ids is not None:
            token_type_embeddings = self.token_type_embeddings(token_type_ids)
        else:
            # Create zero token type embeddings
            token_type_embeddings = ops.zeros_like(inputs_embeds)

        # Combine embeddings
        embeddings = inputs_embeds + position_embeddings + token_type_embeddings
        return self.layer_norm(embeddings)


class PoolingLayer(Module):
    """Pooling layer for sentence-level representations."""

    def __init__(
        self,
        pipeline_config: PipelineConfig,
        weights: Weights,
        huggingface_config: AutoConfig,
        dtype: DType,
    ) -> None:
        super().__init__()
        config = huggingface_config
        self.dense = LinearV1(
            weights.dense.weight.allocate(
                DType.float32,
                [config.hidden_size, config.hidden_size],
                _quantization_encoding(pipeline_config),
            ).cast(dtype),
            bias=weights.dense.bias.allocate(
                DType.float32,
                [config.hidden_size],
                _quantization_encoding(pipeline_config),
            ).cast(dtype),
        )

    def __call__(self, hidden_states: TensorValue) -> TensorValue:
        """Pool by taking the first token and applying dense + tanh."""
        # Pool the model by taking the hidden state corresponding to the first token
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = ops.tanh(pooled_output)
        return pooled_output
