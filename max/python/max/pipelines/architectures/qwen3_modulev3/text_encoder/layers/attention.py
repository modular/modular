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

"""Encoder-only attention without KV cache."""

from __future__ import annotations

from max import functional as F
from max.nn import Linear, Module
from max.nn._graph_ops import (
    masked_flash_attention_gpu as _masked_flash_attention_gpu,
)
from max.nn.norm import RMSNorm
from max.tensor import Tensor

from .rotary_embedding import RotaryEmbedding

masked_flash_attention_gpu = F.functional(_masked_flash_attention_gpu)


class EncoderAttention(Module[..., Tensor]):
    """Encoder-only attention without KV cache (Qwen3: interleaved RoPE via rope.forward)."""

    def __init__(
        self,
        num_attention_heads: int,
        num_key_value_heads: int,
        hidden_size: int,
        head_dim: int,
        scale: float,
        rms_norm_eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.n_heads = num_attention_heads
        self.n_kv_heads = num_key_value_heads
        self.head_dim = head_dim
        self.hidden_size = hidden_size
        self.scale = scale

        q_dim = head_dim * num_attention_heads
        kv_dim = head_dim * num_key_value_heads

        self.q_proj = Linear(hidden_size, q_dim, bias=False)
        self.k_proj = Linear(hidden_size, kv_dim, bias=False)
        self.v_proj = Linear(hidden_size, kv_dim, bias=False)
        self.o_proj = Linear(q_dim, hidden_size, bias=False)

        # Qwen3: Q/K norm over head_dim before RoPE (eps matches config.rms_norm_eps)
        self.q_norm = RMSNorm(head_dim, eps=rms_norm_eps)
        self.k_norm = RMSNorm(head_dim, eps=rms_norm_eps)

    def forward(
        self,
        x: Tensor,
        rope: RotaryEmbedding,
        attention_bias: Tensor | None = None,
    ) -> Tensor:
        """Forward pass computing causal self-attention.

        Args:
            x: Input tensor with shape [total_seq_len, hidden_dim]
            rope: RotaryEmbedding module
            attention_bias: The additive causal mask, or ``None`` to have
                the kernel derive the causal mask arithmetically. The
                arithmetic path reads no mask, which saves an
                ``[S, S]`` mask upload and its per-layer reads.

        Returns:
            Output tensor with shape [total_seq_len, hidden_dim]
        """
        total_seq_len = x.shape[0]

        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        q = F.reshape(q, (total_seq_len, self.n_heads, self.head_dim))
        k = F.reshape(k, (total_seq_len, self.n_kv_heads, self.head_dim))
        v = F.reshape(v, (total_seq_len, self.n_kv_heads, self.head_dim))

        # Qwen3: norm over head_dim (per-head), then RoPE
        q = self.q_norm(q)
        k = self.k_norm(k)

        # module_v3.common_layers RotaryEmbedding.forward expects 4D (B, S, H, D); add batch dim
        q = F.squeeze(rope(F.unsqueeze(q, 0)), 0)
        k = F.squeeze(rope(F.unsqueeze(k, 0)), 0)

        # GQA is the kernel's: K and V stay at n_kv_heads [B, S, H_kv, D]
        # heads and the kernel groups them over the query heads. Repeating
        # them here would materialize a second copy of K and V per layer.
        q = F.unsqueeze(q, 0)
        k = F.unsqueeze(k, 0)
        v = F.unsqueeze(v, 0)
        mask = (
            None
            if attention_bias is None
            else F.squeeze(attention_bias, axis=1)
        )
        attn_out = masked_flash_attention_gpu(
            q,
            k,
            v,
            mask=mask,
            scale=self.scale,
            causal=attention_bias is None,
        )
        attn_out = F.squeeze(attn_out, 0)
        attn_out = F.reshape(attn_out, (total_seq_len, -1))
        return self.o_proj(attn_out)
