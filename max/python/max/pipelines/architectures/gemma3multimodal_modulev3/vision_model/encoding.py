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

from max.experimental.nn import Module
from max.experimental.nn.norm.layer_norm import LayerNorm
from max.experimental.nn.sequential import ModuleList
from max.experimental.tensor import Tensor

from ..model_config import Gemma3ForConditionalGenerationConfig
from .attention import Gemma3VisionAttention
from .projection import Gemma3VisionMLP


class Gemma3VisionEncoderLayer(Module[[Tensor], Tensor]):
    """An individual layer of encoding within a stack of encoding layers."""

    def __init__(
        self,
        config: Gemma3ForConditionalGenerationConfig,
        layer_idx: int,
    ) -> None:
        super().__init__()
        vision_config = config.vision_config

        self.layer_norm1 = LayerNorm(
            dim=vision_config.hidden_size,
            eps=vision_config.layer_norm_eps,
        )

        self.self_attn = Gemma3VisionAttention(
            config=config,
            layer_idx=layer_idx,
        )

        self.mlp = Gemma3VisionMLP(config)

        self.layer_norm2 = LayerNorm(
            dim=vision_config.hidden_size,
            eps=vision_config.layer_norm_eps,
        )

    def forward(self, hidden_states: Tensor) -> Tensor:
        residual = hidden_states
        hidden_states = self.layer_norm1(hidden_states)
        hidden_states = self.self_attn(hidden_states)
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.layer_norm2(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states


class Gemma3VisionEncoder(Module[[Tensor], Tensor]):
    """Stack of vision encoder layers."""

    def __init__(self, config: Gemma3ForConditionalGenerationConfig) -> None:
        super().__init__()

        self.layers = ModuleList(
            [
                Gemma3VisionEncoderLayer(config, layer_idx)
                for layer_idx in range(config.vision_config.num_hidden_layers)
            ]
        )

    def forward(self, hidden_states: Tensor) -> Tensor:
        for layer in self.layers:
            hidden_states = layer(hidden_states)
        return hidden_states
