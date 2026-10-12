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
from max.experimental.nn import Embedding, Linear, Module
from max.experimental.nn.common_layers.rotary_embedding import RotaryEmbedding
from max.experimental.nn.norm import RMSNorm
from max.experimental.nn.sequential import ModuleList
from max.experimental.tensor import Tensor
from max.graph import TensorType

from .layers import EncoderAttention

if TYPE_CHECKING:
    from .model_config import Qwen3TextEncoderConfig


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

    def forward(
        self,
        x: Tensor,
        rope: RotaryEmbedding,
        attention_bias: Tensor,
    ) -> Tensor:
        """Forward pass without KV cache.

        Args:
            x: Input hidden states [seq_len, hidden_dim]
            rope: RoPE embedding module
        Returns:
            Output hidden states [seq_len, hidden_dim]
        """
        residual = x
        x = self.input_layernorm(x)
        x = self.self_attn(
            x,
            rope,
            attention_bias=attention_bias,
        )
        x = residual + x

        residual = x
        x = self.post_attention_layernorm(x)
        x = self.mlp(x)
        x = residual + x

        return x


class Qwen3TextEncoderTransformer(Module[..., tuple[Tensor, ...]]):
    """Qwen3 text encoder transformer without KV cache dependency.

    Returns fused prompt embeddings by stacking configured hidden states and
    merging the layer/hidden dimensions.
    """

    def __init__(self, config: Qwen3TextEncoderConfig) -> None:
        super().__init__()

        self.dim = config.hidden_size
        self.n_heads = config.num_attention_heads
        self.device = config.device
        if config.hidden_state_layers:
            self._sorted_hidden_state_layers = sorted(
                config.hidden_state_layers
            )
        else:
            self._sorted_hidden_state_layers = list(
                range(config.num_hidden_layers)
            )
        self._hidden_state_layers = set(self._sorted_hidden_state_layers)

        self.rope = RotaryEmbedding(
            dim=config.hidden_size,
            n_heads=config.num_attention_heads,
            theta=config.rope_theta,
            max_seq_len=config.max_seq_len,
            device=config.device.to_device(),
            head_dim=config.head_dim,
            interleaved=False,
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
            TensorType(
                DType.float32,
                shape=[1, 1, "total_seq_len", "total_seq_len"],
                device=self.device,
            ),
        )

    def forward(
        self,
        tokens: Tensor,
        attention_bias: Tensor,
    ) -> tuple[Tensor, ...]:
        """Forward pass returning fused prompt embeddings.

        Args:
            tokens: Input token IDs [total_seq_len]
            attention_bias: Additive causal+padding mask bias with shape
                [1, 1, seq_len, seq_len].

        Returns:
            Tuple containing one tensor shaped [1, seq_len, num_layers * hidden_dim].
        """
        h = self.embed_tokens(tokens)

        # Match Hugging Face `output.hidden_states` indexing:
        #   hidden_states[0] = token embeddings
        #   hidden_states[i + 1] = output after transformer block i
        # Flux2-Klein layer indices are specified against that HF contract.
        selected: dict[int, Tensor] = {}
        if 0 in self._hidden_state_layers:
            selected[0] = h

        max_layer = self._sorted_hidden_state_layers[-1]
        if max_layer > 0:
            for i, layer in enumerate(self.layers):
                h = layer(h, self.rope, attention_bias)
                hf_hidden_state_index = i + 1
                if hf_hidden_state_index in self._hidden_state_layers:
                    selected[hf_hidden_state_index] = h
                if hf_hidden_state_index == max_layer:
                    break

        hidden_states = [selected[i] for i in self._sorted_hidden_state_layers]

        stacked = F.stack(hidden_states, axis=0)  # [L, S, D]
        stacked = F.unsqueeze(stacked, axis=0)  # [1, L, S, D]
        stacked = F.permute(stacked, [0, 2, 1, 3])  # [1, S, L, D]
        seq_len = stacked.shape[1]
        return (
            F.reshape(
                stacked, [1, seq_len, stacked.shape[2] * stacked.shape[3]]
            ),
        )
