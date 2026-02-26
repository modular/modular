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

"""Qwen3 text encoder transformer without KV cache dependency.

This is a standalone transformer implementation for text encoding that does not
require KV cache. Suitable for single-pass encoding in diffusion pipelines.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from max.dtype import DType
from max.experimental import functional as F
from max.experimental.tensor import Tensor
from max.graph import TensorType
from max.nn.module_v3 import Embedding, Linear, Module
from max.nn.module_v3.norm import RMSNorm
from max.nn.module_v3.sequential import ModuleList

from .layers import EncoderAttention, RotaryEmbedding

if TYPE_CHECKING:
    from .model_config import Qwen3TextEncoderConfigBase


class Qwen3MLP(Module[[Tensor], Tensor]):
    """Qwen3 MLP with SiLU gate activation."""

    def __init__(self, hidden_size: int, intermediate_size: int) -> None:
        super().__init__()
        self.gate_proj = Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = Linear(intermediate_size, hidden_size, bias=False)

    def forward(self, hidden_states: Tensor) -> Tensor:
        gate = F.silu(self.gate_proj(hidden_states))
        up = self.up_proj(hidden_states)
        return self.down_proj(gate * up)


class EncoderTransformerBlock(Module[..., Tensor]):
    """Transformer block for encoder-only models without KV cache."""

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int,
        intermediate_size: int,
        rms_norm_eps: float,
        scale: float,
    ) -> None:
        super().__init__()
        self.self_attn = EncoderAttention(
            num_attention_heads=num_heads,
            num_key_value_heads=num_kv_heads,
            hidden_size=hidden_size,
            head_dim=head_dim,
            scale=scale,
            rms_norm_eps=rms_norm_eps,
        )
        self.mlp = Qwen3MLP(hidden_size, intermediate_size)
        self.input_layernorm = RMSNorm(hidden_size, eps=rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(hidden_size, eps=rms_norm_eps)

    def forward(self, x: Tensor, rope: RotaryEmbedding) -> Tensor:
        """Forward pass without KV cache.

        Args:
            x: Input hidden states [seq_len, hidden_dim]
            rope: RoPE embedding module

        Returns:
            Output hidden states [seq_len, hidden_dim]
        """
        residual = x
        x = self.input_layernorm(x)
        x = self.self_attn(x, rope)
        x = residual + x

        residual = x
        x = self.post_attention_layernorm(x)
        x = self.mlp(x)
        x = residual + x

        return x


class Qwen3TextEncoderTransformer(Module[..., tuple[Tensor, ...]]):
    """Qwen3 text encoder transformer without KV cache dependency.

    Returns hidden states from all layers for use in diffusion pipelines.
    """

    def __init__(self, config: Qwen3TextEncoderConfigBase) -> None:
        super().__init__()

        self.dim = config.hidden_size
        self.n_heads = config.num_attention_heads
        self.device = config.device

        self.rope = RotaryEmbedding(
            dim=config.hidden_size,
            n_heads=config.num_attention_heads,
            theta=config.rope_theta,
            max_seq_len=config.max_seq_len,
            device=config.device.to_device(),
            head_dim=config.head_dim,
        )

        self.layers = ModuleList(
            [
                EncoderTransformerBlock(
                    hidden_size=config.hidden_size,
                    num_heads=config.num_attention_heads,
                    num_kv_heads=config.num_key_value_heads,
                    head_dim=config.head_dim,
                    intermediate_size=config.intermediate_size,
                    rms_norm_eps=config.rms_norm_eps,
                    scale=config.attention_multiplier,
                )
                for _ in range(config.num_hidden_layers)
            ]
        )

        self.embed_tokens = Embedding(config.vocab_size, dim=config.hidden_size)

    def input_types(self) -> tuple[TensorType, ...]:
        """Define input tensor types for compilation."""
        return (
            TensorType(
                DType.int64,
                shape=["total_seq_len"],
                device=self.device,
            ),
        )

    def forward(self, tokens: Tensor) -> tuple[Tensor, ...]:
        """Forward pass returning hidden states from all layers.

        Args:
            tokens: Input token IDs [total_seq_len]

        Returns:
            Tuple of hidden states from all layers, each with shape [seq_len, hidden_dim]
        """
        h = self.embed_tokens(tokens)

        all_hidden_states: list[Tensor] = []
        for layer in self.layers:
            h = layer(h, self.rope)
            all_hidden_states.append(h)

        return tuple(all_hidden_states)
