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

from max.graph import TensorValue
from max.nn import (
    LayerList,
    LayerNorm,
    Module,
)

from ..attention import Gemma3VisionAttention
from ..model_config import Gemma3ForConditionalGenerationConfig
from .projection import Gemma3VisionMLP


# ✅ based on HF and MLX-VLM
class Gemma3VisionEncoderLayer(Module):
    def __init__(
        self, config: Gemma3ForConditionalGenerationConfig, layer_idx: int
    ):
        vision_config = config.vision_config

        self.embed_dim = vision_config.hidden_size

        # Pre-attention layer norm
        self.layer_norm1 = LayerNorm(
            self.embed_dim,
            eps=vision_config.layer_norm_eps,
            device=config.devices[0],
            dtype=config.dtype,
        )

        # Self-attention
        self.self_attn = Gemma3VisionAttention(
            config=config,
            layer_idx=layer_idx,
        )

        # MLP (Feed-Forward Network) - simple GELUTanh/fc1/fc2 style
        self.mlp = Gemma3VisionMLP(config)

        # post-attention layer norm
        self.layer_norm2 = LayerNorm(
            self.embed_dim,
            eps=vision_config.layer_norm_eps,
            device=config.devices[0],
            dtype=config.dtype,
        )

    def __call__(
        self,
        hidden_states: TensorValue,
    ) -> TensorValue:
        residual = hidden_states
        hidden_states = self.layer_norm1(hidden_states)
        hidden_states = self.self_attn(hidden_states)
        hidden_states = residual + hidden_states

        # MLP with residual
        residual = hidden_states
        hidden_states = self.layer_norm2(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states


# ✅ based on HF and MLX-VLM
class Gemma3VisionEncoder(Module):
    """SigLIP vision encoder with 27 transformer layers."""

    def __init__(self, config: Gemma3ForConditionalGenerationConfig):
        super().__init__()
        self.layers = LayerList(
            [
                Gemma3VisionEncoderLayer(config, layer_idx)
                for layer_idx, _ in enumerate(
                    range(config.vision_config.num_hidden_layers)
                )
            ]
        )

    def __call__(
        self,
        hidden_states: TensorValue,
    ) -> TensorValue:
        # Pass through all layers
        for layer in self.layers:
            hidden_states = layer(hidden_states)
        return hidden_states
