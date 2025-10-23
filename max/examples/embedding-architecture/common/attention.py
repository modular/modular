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
"""Multi-head self-attention layer for transformer models."""

import math

from max.dtype import DType
from max.graph import TensorValue, ops
from max.graph.weights import Weights
from max.nn import LayerNormV1, LinearV1, Module
from max.pipelines.lib import PipelineConfig
from transformers import AutoConfig

from .utils import _quantization_encoding


class MultiHeadSelfAttention(Module):
    """Multi-head self-attention layer shared across transformer models."""

    def __init__(
        self,
        pipeline_config: PipelineConfig,
        weights: Weights,
        huggingface_config: AutoConfig,
        dtype: DType,
    ) -> None:
        super().__init__()
        config = huggingface_config
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(
            config.hidden_size / config.num_attention_heads
        )
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = LinearV1(
            weights.query.weight.allocate(
                DType.float32,
                [self.all_head_size, config.hidden_size],
                _quantization_encoding(pipeline_config),
            ).cast(dtype),
            bias=weights.query.bias.allocate(
                DType.float32,
                [self.all_head_size],
                _quantization_encoding(pipeline_config),
            ).cast(dtype),
        )
        self.key = LinearV1(
            weights.key.weight.allocate(
                DType.float32,
                [self.all_head_size, config.hidden_size],
                _quantization_encoding(pipeline_config),
            ).cast(dtype),
            bias=weights.key.bias.allocate(
                DType.float32,
                [self.all_head_size],
                _quantization_encoding(pipeline_config),
            ).cast(dtype),
        )
        self.value = LinearV1(
            weights.value.weight.allocate(
                DType.float32,
                [self.all_head_size, config.hidden_size],
                _quantization_encoding(pipeline_config),
            ).cast(dtype),
            bias=weights.value.bias.allocate(
                DType.float32,
                [self.all_head_size],
                _quantization_encoding(pipeline_config),
            ).cast(dtype),
        )

    def transpose_for_scores(self, x: TensorValue) -> TensorValue:
        """Reshape tensor for multi-head attention computation."""
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
    ) -> TensorValue:
        """Forward pass of multi-head self-attention."""
        query_layer = self.query(hidden_states)
        key_layer = self.key(hidden_states)
        value_layer = self.value(hidden_states)

        query_layer = self.transpose_for_scores(query_layer)
        key_layer = self.transpose_for_scores(key_layer)
        value_layer = self.transpose_for_scores(value_layer)

        # Compute attention scores
        attention_scores = query_layer @ key_layer.transpose(-1, -2)
        attention_scores = attention_scores / math.sqrt(
            self.attention_head_size
        )

        # Apply attention mask
        attention_scores = attention_scores + attention_mask

        # Normalize to probabilities
        attention_probs = ops.softmax(attention_scores)

        # Apply attention to values
        context_layer = attention_probs @ value_layer

        # Reshape back to original format
        context_layer = ops.permute(context_layer, [0, 2, 1, 3])
        new_context_layer_shape = context_layer.shape[:-2] + [
            self.all_head_size
        ]
        context_layer = ops.reshape(context_layer, new_context_layer_shape)

        return context_layer


class AttentionOutput(Module):
    """Attention output layer with residual connection and layer norm."""

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
        self, hidden_states: TensorValue, input_tensor: TensorValue
    ) -> TensorValue:
        """Apply dense projection, residual connection, and layer norm."""
        hidden_states = self.dense(hidden_states)
        hidden_states = self.layer_norm(hidden_states + input_tensor)
        return hidden_states


class AttentionLayer(Module):
    """Complete attention layer combining self-attention and output projection."""

    def __init__(
        self,
        pipeline_config: PipelineConfig,
        weights: Weights,
        huggingface_config: AutoConfig,
        dtype: DType,
    ) -> None:
        super().__init__()
        self.self_attention = MultiHeadSelfAttention(
            pipeline_config, weights.self, huggingface_config, dtype
        )
        self.output = AttentionOutput(
            pipeline_config, weights.output, huggingface_config, dtype
        )

    def __call__(
        self,
        hidden_states: TensorValue,
        attention_mask: TensorValue,
    ) -> TensorValue:
        """Forward pass through attention layer."""
        attention_output = self.self_attention(hidden_states, attention_mask)
        layer_output = self.output(attention_output, hidden_states)
        return layer_output
