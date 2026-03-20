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
"""Cache-free attention layer for Qwen3 embedding models."""

from __future__ import annotations

import math

from max.driver import CPU
from max.experimental import functional as F
from max.experimental.nn import Linear, Module
from max.experimental.nn.common_layers.rotary_embedding import RotaryEmbedding
from max.experimental.nn.norm import RMSNorm
from max.experimental.tensor import Tensor
from max.graph import Dim
from max.nn.attention.mask_config import MHAMaskVariant
from max.nn.kernels import (
    flash_attention_ragged_gpu as _flash_attention_ragged_gpu,
)

flash_attention_ragged_gpu = F.functional(_flash_attention_ragged_gpu)


class Qwen3AttentionNoCache(
    Module[[Tensor, Tensor], Tensor],
):
    """Qwen3 attention layer without KV caching for embedding models.

    Uses flash_attention_ragged_gpu directly with Q, K, V tensors,
    with Qwen3-specific Q/K normalization and RoPE.
    """

    def __init__(
        self,
        *,
        rope: RotaryEmbedding,
        num_attention_heads: int,
        num_key_value_heads: int,
        hidden_size: int,
        head_dim: int,
        scale: float | None = None,
        qk_norm_eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.rope = rope
        self.n_heads = num_attention_heads
        self.n_kv_heads = num_key_value_heads
        self.head_dim = head_dim
        self.hidden_size = hidden_size
        self.scale = (
            scale if scale is not None else math.sqrt(1.0 / self.head_dim)
        )

        self.q_weight_dim = self.n_heads * self.head_dim
        self.kv_weight_dim = self.n_kv_heads * self.head_dim

        self.q_proj = Linear(
            in_dim=hidden_size, out_dim=self.q_weight_dim, bias=False
        )
        self.k_proj = Linear(
            in_dim=hidden_size, out_dim=self.kv_weight_dim, bias=False
        )
        self.v_proj = Linear(
            in_dim=hidden_size, out_dim=self.kv_weight_dim, bias=False
        )

        # Q and K normalization layers (Qwen3-specific)
        self.q_norm = RMSNorm(self.head_dim, eps=qk_norm_eps)
        self.k_norm = RMSNorm(self.head_dim, eps=qk_norm_eps)

        self.o_proj = Linear(
            in_dim=self.q_weight_dim, out_dim=hidden_size, bias=False
        )

    def forward(
        self,
        x: Tensor,
        input_row_offsets: Tensor,
    ) -> Tensor:
        total_seq_len = x.shape[0]

        # Project Q, K, V separately
        xq = self.q_proj(x)
        xk = self.k_proj(x)
        xv = self.v_proj(x)

        # Reshape for multi-head attention
        xq = xq.reshape((-1, self.n_heads, self.head_dim))
        xk = xk.reshape((-1, self.n_kv_heads, self.head_dim))
        xv = xv.reshape((-1, self.n_kv_heads, self.head_dim))

        # Apply Q and K normalization before RoPE (Qwen3-specific)
        xq = self.q_norm(xq)
        xk = self.k_norm(xk)

        # Apply RoPE after normalization
        # RoPE expects shape (batch, seq_len, n_heads, head_dim)
        xq_reshaped = xq.reshape((1, -1, self.n_heads, self.head_dim))
        xk_reshaped = xk.reshape((1, -1, self.n_kv_heads, self.head_dim))

        xq = self.rope(
            xq_reshaped, start_pos=Dim(0), seq_len=total_seq_len
        ).reshape((-1, self.n_heads, self.head_dim))
        xk = self.rope(
            xk_reshaped, start_pos=Dim(0), seq_len=total_seq_len
        ).reshape((-1, self.n_kv_heads, self.head_dim))

        # Compute max sequence length from row offsets
        seq_lens = input_row_offsets[1:] - input_row_offsets[:-1]
        max_seq_len = F.max(seq_lens, axis=0)

        # For GQA, expand K and V to match Q's number of heads
        if self.n_kv_heads != self.n_heads:
            n_rep = self.n_heads // self.n_kv_heads
            xk = self._repeat_kv(xk, n_rep)
            xv = self._repeat_kv(xv, n_rep)

        # Qwen3-Embedding uses causal masking by design (decoder-only encoder).
        # Unwrap max_seq_len to TensorValue so the v2 kernel's device
        # validation sees DeviceRef (Tensor.device returns Device which
        # fails the DeviceRef equality check in _validate_argument_tensor).
        max_seq_len_cpu = max_seq_len.to(CPU()).__tensorvalue__()

        attn_out = flash_attention_ragged_gpu(
            q=xq,
            k=xk,
            v=xv,
            input_row_offsets=input_row_offsets,
            max_seq_len=max_seq_len_cpu,
            mask_variant=MHAMaskVariant.CAUSAL_MASK,
            scale=self.scale,
        )

        attn_out = F.reshape(attn_out, shape=[total_seq_len, -1])
        return self.o_proj(attn_out)

    def _repeat_kv(self, x: Tensor, n_rep: int) -> Tensor:
        """Repeat K or V tensors to match the number of query heads (for GQA)."""
        if n_rep == 1:
            return x

        x_expanded = x.unsqueeze(2)
        tensors_to_concat = [x_expanded] * n_rep
        x_repeated = F.concat(tensors_to_concat, axis=2)
        return x_repeated.reshape((-1, self.n_kv_heads * n_rep, self.head_dim))
