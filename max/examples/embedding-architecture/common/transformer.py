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
"""Transformer block and encoder layers for transformer models."""

from max.dtype import DType
from max.graph import TensorValue
from max.graph.weights import Weights
from max.nn import Module, Sequential
from max.pipelines.lib import PipelineConfig
from transformers import AutoConfig

from .attention import AttentionLayer
from .feedforward import FeedForwardBlock


class TransformerBlock(Module):
    """A single transformer block combining attention and feed-forward layers."""

    def __init__(
        self,
        pipeline_config: PipelineConfig,
        weights: Weights,
        huggingface_config: AutoConfig,
        dtype: DType,
    ) -> None:
        super().__init__()
        self.attention = AttentionLayer(
            pipeline_config, weights.attention, huggingface_config, dtype
        )
        self.feedforward = FeedForwardBlock(
            pipeline_config, weights, huggingface_config, dtype
        )

    def __call__(
        self,
        hidden_states: TensorValue,
        attention_mask: TensorValue,
    ) -> TensorValue:
        """Forward pass through transformer block."""
        # Self-attention with residual connection and layer norm
        attention_output = self.attention(hidden_states, attention_mask)

        # Feed-forward with residual connection and layer norm
        layer_output = self.feedforward(attention_output, attention_output)

        return layer_output


class TransformerEncoder(Module):
    """Transformer encoder containing multiple transformer blocks."""

    def __init__(
        self,
        pipeline_config: PipelineConfig,
        weights: Weights,
        huggingface_config: AutoConfig,
        dtype: DType,
    ) -> None:
        super().__init__()
        config = huggingface_config
        num_hidden_layers = config.num_hidden_layers
        self.layers = Sequential(
            [
                TransformerBlock(
                    pipeline_config, weights.layer[n], huggingface_config, dtype
                )
                for n in range(num_hidden_layers)
            ]
        )

    def __call__(
        self,
        hidden_states: TensorValue,
        attention_mask: TensorValue,
    ) -> TensorValue:
        """Forward pass through all transformer layers."""
        for layer in self.layers.layers:
            hidden_states = layer(hidden_states, attention_mask)
        return hidden_states
