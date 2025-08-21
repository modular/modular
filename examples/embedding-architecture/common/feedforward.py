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
"""Feed-forward network layers for transformer models."""

from max.dtype import DType
from max.graph import TensorValue
from max.graph.weights import Weights
from max.nn import LayerNormV1, LinearV1, Module
from max.pipelines.lib import PipelineConfig
from transformers import AutoConfig

from .utils import ACTIVATIONS, _quantization_encoding


class FeedForwardIntermediate(Module):
    """Intermediate (feed-forward expansion) layer."""

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
                [config.intermediate_size, config.hidden_size],
                _quantization_encoding(pipeline_config),
            ).cast(dtype),
            bias=weights.dense.bias.allocate(
                DType.float32,
                [config.intermediate_size],
                _quantization_encoding(pipeline_config),
            ).cast(dtype),
        )
        self.intermediate_act_fn = ACTIVATIONS[config.hidden_act]

    def __call__(self, hidden_states: TensorValue) -> TensorValue:
        """Apply linear transformation and activation function."""
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states


class FeedForwardOutput(Module):
    """Feed-forward output layer with residual connection and layer norm."""

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
                [config.hidden_size, config.intermediate_size],
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
        """Apply linear transformation, residual connection, and layer norm."""
        hidden_states = self.dense(hidden_states)
        hidden_states = self.layer_norm(hidden_states + input_tensor)
        return hidden_states


class FeedForwardBlock(Module):
    """Complete feed-forward block combining intermediate and output layers."""

    def __init__(
        self,
        pipeline_config: PipelineConfig,
        weights: Weights,
        huggingface_config: AutoConfig,
        dtype: DType,
    ) -> None:
        super().__init__()
        self.intermediate = FeedForwardIntermediate(
            pipeline_config, weights.intermediate, huggingface_config, dtype
        )
        self.output = FeedForwardOutput(
            pipeline_config, weights.output, huggingface_config, dtype
        )

    def __call__(
        self, hidden_states: TensorValue, input_tensor: TensorValue
    ) -> TensorValue:
        """Forward pass through feed-forward block."""
        intermediate_output = self.intermediate(hidden_states)
        layer_output = self.output(intermediate_output, input_tensor)
        return layer_output
