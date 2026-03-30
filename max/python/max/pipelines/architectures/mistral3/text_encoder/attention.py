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

from max.dtype import DType
from max.graph import DeviceRef, TensorValue, ops
from max.nn.attention.mask_config import MHAMaskVariant
from max.nn.kernels import flash_attention_gpu
from max.nn.layer import Module
from max.nn.linear import Linear
from max.nn.rotary_embedding import RotaryEmbedding


class EncoderAttention(Module):
    """Encoder-only attention without KV cache."""

    def __init__(
        self,
        *,
        num_attention_heads: int,
        num_key_value_heads: int,
        hidden_size: int,
        head_dim: int,
        dtype: DType,
        device: DeviceRef,
        scale: float,
    ) -> None:
        super().__init__()
        self.n_heads = num_attention_heads
        self.n_kv_heads = num_key_value_heads
        self.head_dim = head_dim
        self.scale = scale

        q_dim = head_dim * num_attention_heads
        kv_dim = head_dim * num_key_value_heads

        self.q_proj = Linear(
            hidden_size,
            q_dim,
            dtype=dtype,
            device=device,
            has_bias=False,
        )
        self.k_proj = Linear(
            hidden_size,
            kv_dim,
            dtype=dtype,
            device=device,
            has_bias=False,
        )
        self.v_proj = Linear(
            hidden_size,
            kv_dim,
            dtype=dtype,
            device=device,
            has_bias=False,
        )
        self.o_proj = Linear(
            q_dim,
            hidden_size,
            dtype=dtype,
            device=device,
            has_bias=False,
        )

    def _repeat_kv(self, x: TensorValue, n_rep: int) -> TensorValue:
        """Repeat KV heads for grouped-query attention.

        Args:
            x: Tensor with shape ``[1, seq_len, n_kv_heads, head_dim]``.
            n_rep: Number of repetitions per KV head.

        Returns:
            Tensor with shape ``[1, seq_len, n_heads, head_dim]``.
        """
        if n_rep == 1:
            return x

        batch_size = x.shape[0]
        seq_len = x.shape[1]
        x = ops.unsqueeze(x, 3)
        x = ops.concat([x] * n_rep, axis=3)
        return ops.reshape(
            x,
            [batch_size, seq_len, self.n_kv_heads * n_rep, self.head_dim],
        )

    def __call__(self, x: TensorValue, rope: RotaryEmbedding) -> TensorValue:
        """Forward pass computing causal self-attention.

        Args:
            x: Hidden states with shape ``[seq_len, hidden_size]``.
            rope: Rotary embedding layer applied to Q and K.

        Returns:
            Tensor with shape ``[seq_len, hidden_size]``.
        """
        total_seq_len = x.shape[0]

        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        q = ops.reshape(q, [1, total_seq_len, self.n_heads, self.head_dim])
        k = ops.reshape(k, [1, total_seq_len, self.n_kv_heads, self.head_dim])
        v = ops.reshape(v, [1, total_seq_len, self.n_kv_heads, self.head_dim])

        q = rope(q)
        k = rope(k)

        if self.n_kv_heads != self.n_heads:
            n_rep = self.n_heads // self.n_kv_heads
            k = self._repeat_kv(k, n_rep)
            v = self._repeat_kv(v, n_rep)

        attn_out = flash_attention_gpu(
            q,
            k,
            v,
            mask_variant=MHAMaskVariant.CAUSAL_MASK,
            scale=self.scale,
        )
        attn_out = ops.reshape(attn_out, [total_seq_len, -1])
        return self.o_proj(attn_out)
