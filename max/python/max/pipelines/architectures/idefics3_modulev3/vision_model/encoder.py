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
"""Idefics3 vision encoder implementation (ModuleV3)."""

from __future__ import annotations

from max.experimental import functional as F
from max.experimental.nn import Module
from max.experimental.nn.linear import Linear
from max.experimental.nn.norm import LayerNorm
from max.experimental.nn.sequential import ModuleList
from max.experimental.tensor import Tensor

from ..model_config import Idefics3VisionConfig
from .attention import Idefics3VisionAttention


class Idefics3VisionMLP(Module[[Tensor], Tensor]):
    """Vision MLP for Idefics3 encoder layer (ModuleV3).

    Standard transformer feed-forward network with GELU activation.
    """

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
    ) -> None:
        super().__init__()
        self.fc1 = Linear(
            in_dim=hidden_size, out_dim=intermediate_size, bias=True
        )
        self.fc2 = Linear(
            in_dim=intermediate_size, out_dim=hidden_size, bias=True
        )

    def forward(self, x: Tensor) -> Tensor:
        x = self.fc1(x)
        x = F.gelu(x, approximate="tanh")
        x = self.fc2(x)
        return x


class Idefics3VisionEncoderLayer(Module[[Tensor], Tensor]):
    """Single encoder layer for the Idefics3 vision transformer (ModuleV3).

    Pre-norm architecture with self-attention and MLP, each with residual
    connections.
    """

    def __init__(self, vision_config: Idefics3VisionConfig) -> None:
        super().__init__()
        embed_dim = vision_config.hidden_size

        self.self_attn = Idefics3VisionAttention(
            hidden_size=vision_config.hidden_size,
            num_attention_heads=vision_config.num_attention_heads,
        )
        self.layer_norm1 = LayerNorm(
            embed_dim,
            eps=vision_config.layer_norm_eps,
            use_bias=True,
        )
        self.layer_norm2 = LayerNorm(
            embed_dim,
            eps=vision_config.layer_norm_eps,
            use_bias=True,
        )
        self.mlp = Idefics3VisionMLP(
            hidden_size=vision_config.hidden_size,
            intermediate_size=vision_config.intermediate_size,
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


class Idefics3VisionEncoder(Module[[Tensor], Tensor]):
    """Idefics3 vision encoder stack (ModuleV3).

    Consists of multiple Idefics3VisionEncoderLayer instances.
    """

    def __init__(self, vision_config: Idefics3VisionConfig) -> None:
        super().__init__()
        self.layers = ModuleList(
            [
                Idefics3VisionEncoderLayer(vision_config)
                for _ in range(vision_config.num_hidden_layers)
            ]
        )

    def forward(self, hidden_states: Tensor) -> Tensor:
        for layer in self.layers:
            hidden_states = layer(hidden_states)
        return hidden_states
