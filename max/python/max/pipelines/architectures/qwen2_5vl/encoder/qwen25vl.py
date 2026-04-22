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

"""Qwen2.5-VL text encoder transformer (module v2).

Standalone transformer for text encoding in diffusion pipelines.
Returns the final normed hidden states.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from max.graph import TensorValue, ops
from max.nn.layer import LayerList, Module
from max.nn.linear import Linear
from max.nn.norm.rms_norm import RMSNorm
from max.nn.rotary_embedding import RotaryEmbedding

from .layers import Qwen25VLEncoderAttention

if TYPE_CHECKING:
    from max.dtype import DType
    from max.graph import DeviceRef

    from .model_config import Qwen25VLTextEncoderConfigBase


class Qwen25VLMLP(Module):
    """Qwen2.5-VL MLP with SiLU gate activation (module v2)."""

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        *,
        dtype: DType,
        device: DeviceRef,
    ) -> None:
        super().__init__()
        self.gate_proj = Linear(
            hidden_size,
            intermediate_size,
            dtype=dtype,
            device=device,
            has_bias=False,
        )
        self.up_proj = Linear(
            hidden_size,
            intermediate_size,
            dtype=dtype,
            device=device,
            has_bias=False,
        )
        self.down_proj = Linear(
            intermediate_size,
            hidden_size,
            dtype=dtype,
            device=device,
            has_bias=False,
        )

    def __call__(self, hidden_states: TensorValue) -> TensorValue:
        gate = ops.silu(self.gate_proj(hidden_states))
        up = self.up_proj(hidden_states)
        return self.down_proj(gate * up)


class Qwen25VLEncoderTransformerBlock(Module):
    """Transformer block for Qwen2.5-VL encoder (module v2)."""

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int,
        intermediate_size: int,
        rms_norm_eps: float,
        scale: float,
        attention_bias: bool = True,
        *,
        dtype: DType,
        device: DeviceRef,
    ) -> None:
        super().__init__()
        self.self_attn = Qwen25VLEncoderAttention(
            num_attention_heads=num_heads,
            num_key_value_heads=num_kv_heads,
            hidden_size=hidden_size,
            head_dim=head_dim,
            scale=scale,
            attention_bias=attention_bias,
            dtype=dtype,
            device=device,
        )
        self.mlp = Qwen25VLMLP(
            hidden_size,
            intermediate_size,
            dtype=dtype,
            device=device,
        )
        self.input_layernorm = RMSNorm(
            hidden_size, dtype=dtype, eps=rms_norm_eps
        )
        self.post_attention_layernorm = RMSNorm(
            hidden_size,
            dtype=dtype,
            eps=rms_norm_eps,
        )

    def __call__(
        self,
        x: TensorValue,
        rope: RotaryEmbedding,
    ) -> TensorValue:
        residual = x
        x = self.input_layernorm(x)
        x = self.self_attn(x, rope)
        x = residual + x

        residual = x
        x = self.post_attention_layernorm(x)
        x = self.mlp(x)
        x = residual + x
        return x


class Qwen25VLTextEncoderTransformer(Module):
    """Qwen2.5-VL text encoder (module v2).

    Split into two sub-modules for separate compilation:
    - embed_tokens: token embedding
    - layers + norm: transformer blocks + final norm
    """

    def __init__(self, config: Qwen25VLTextEncoderConfigBase) -> None:
        super().__init__()
        dtype = config.dtype
        device = config.device

        self.rope = RotaryEmbedding(
            dim=config.hidden_size,
            n_heads=config.num_attention_heads,
            theta=config.rope_theta,
            max_seq_len=config.max_seq_len,
            head_dim=config.head_dim,
            interleaved=False,
        )

        self.layers = LayerList(
            [
                Qwen25VLEncoderTransformerBlock(
                    hidden_size=config.hidden_size,
                    num_heads=config.num_attention_heads,
                    num_kv_heads=config.num_key_value_heads,
                    head_dim=config.head_dim,
                    intermediate_size=config.intermediate_size,
                    rms_norm_eps=config.rms_norm_eps,
                    scale=config.attention_multiplier,
                    attention_bias=config.attention_bias,
                    dtype=dtype,
                    device=device,
                )
                for _ in range(config.num_hidden_layers)
            ]
        )
        self.norm = RMSNorm(
            config.hidden_size, dtype=dtype, eps=config.rms_norm_eps
        )

    def __call__(self, hidden_states: TensorValue) -> TensorValue:
        """Run transformer layers + norm on pre-embedded hidden states."""
        h = hidden_states
        for layer in self.layers:
            h = layer(h, self.rope)
        return self.norm(h)
