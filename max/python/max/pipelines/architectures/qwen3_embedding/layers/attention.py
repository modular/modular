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

from max.dtype import DType
from max.graph import DeviceRef, Dim, TensorValue, ops
from max.nn.attention.mask_config import MHAMaskVariant
from max.nn.kernels import flash_attention_ragged_gpu
from max.nn.layer import Module
from max.nn.linear import Linear
from max.nn.norm import RMSNorm
from max.nn.rotary_embedding import RotaryEmbedding


class Qwen3AttentionNoCache(Module):
    """Qwen3 attention layer without KV caching for embedding models.

    This is a simplified version of Qwen3Attention that:
    - Does not use KVCache (no PagedCacheValues, no fused_qkv_ragged_matmul)
    - Uses flash_attention_ragged_gpu directly with Q, K, V tensors
    - Still applies Qwen3-specific Q/K normalization and RoPE
    - Optimized for single-pass forward (embedding generation)
    """

    def __init__(
        self,
        *,
        rope: RotaryEmbedding,
        num_attention_heads: int,
        num_key_value_heads: int,
        hidden_size: int,
        head_dim: int,
        dtype: DType = DType.float32,
        devices: list[DeviceRef],
        scale: float | None = None,
        qk_norm_eps: float = 1e-6,
    ) -> None:
        """Initializes the cache-free attention layer.

        Args:
            rope: The rope layer for rotary position embeddings.
            num_attention_heads: The number of attention heads.
            num_key_value_heads: Number of key/value heads (for GQA).
            hidden_size: The dimension of the hidden states.
            head_dim: The dimension of each attention head.
            dtype: Data type for weights and computations.
            devices: Device(s) on which to place the weights.
            scale: Value used to scale the attention scores. If None, uses 1/sqrt(head_dim).
            qk_norm_eps: Epsilon value for Q/K normalization. Defaults to 1e-6.
        """
        super().__init__()
        self.rope = rope
        self.n_heads = num_attention_heads
        self.n_kv_heads = num_key_value_heads
        self.head_dim = head_dim
        self.hidden_size = hidden_size
        self.devices = devices
        self.scale = (
            scale if scale is not None else math.sqrt(1.0 / self.head_dim)
        )
        self.qk_norm_eps = qk_norm_eps

        # Weight dimensions
        self.q_weight_dim = self.n_heads * self.head_dim
        self.kv_weight_dim = self.n_kv_heads * self.head_dim

        # Q, K, V projection layers (separate, not fused)
        # Qwen3-Embedding uses bias=False for attention projections
        self.q_proj = Linear(
            in_dim=hidden_size,
            out_dim=self.q_weight_dim,
            dtype=dtype,
            device=devices[0],
            has_bias=False,
        )
        self.k_proj = Linear(
            in_dim=hidden_size,
            out_dim=self.kv_weight_dim,
            dtype=dtype,
            device=devices[0],
            has_bias=False,
        )
        self.v_proj = Linear(
            in_dim=hidden_size,
            out_dim=self.kv_weight_dim,
            dtype=dtype,
            device=devices[0],
            has_bias=False,
        )

        # Q and K normalization layers (Qwen3-specific)
        self.q_norm = RMSNorm(
            self.head_dim,
            dtype=dtype,
            eps=self.qk_norm_eps,
            multiply_before_cast=False,
        )
        self.k_norm = RMSNorm(
            self.head_dim,
            dtype=dtype,
            eps=self.qk_norm_eps,
            multiply_before_cast=False,
        )

        # Output projection (Qwen3-Embedding uses bias=False)
        self.o_proj = Linear(
            in_dim=self.q_weight_dim,
            out_dim=hidden_size,
            dtype=dtype,
            device=devices[0],
            has_bias=False,
        )

    def __call__(
        self,
        x: TensorValue,
        input_row_offsets: TensorValue,
    ) -> TensorValue:
        """Forward pass without KV caching.

        Args:
            x: Input tensor of shape [total_seq_len, hidden_size] (ragged)
            input_row_offsets: Offset tensor of shape [batch_size + 1] indicating
                sequence boundaries in the ragged input

        Returns:
            Output tensor of shape [total_seq_len, hidden_size]
        """
        total_seq_len = x.shape[0]

        # Project Q, K, V separately
        xq = self.q_proj(x)
        xk = self.k_proj(x)
        xv = self.v_proj(x)

        # Reshape for multi-head attention
        # Q: [total_seq_len, n_heads, head_dim]
        # K, V: [total_seq_len, n_kv_heads, head_dim]
        xq = xq.reshape((-1, self.n_heads, self.head_dim))
        xk = xk.reshape((-1, self.n_kv_heads, self.head_dim))
        xv = xv.reshape((-1, self.n_kv_heads, self.head_dim))

        # Apply Q and K normalization before RoPE (Qwen3-specific)
        # PyTorch: query_states = self.q_norm(self.q_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        xq = self.q_norm(xq)
        xk = self.k_norm(xk)

        # Apply RoPE after normalization
        # RoPE expects shape (batch, seq_len, n_heads, head_dim)
        # We reshape to add batch dimension, but must pass seq_len explicitly
        # since the ragged batch looks like one long sequence
        xq_reshaped = xq.reshape((1, -1, self.n_heads, self.head_dim))
        xk_reshaped = xk.reshape((1, -1, self.n_kv_heads, self.head_dim))

        # Pass explicit seq_len to RoPE so it applies positions 0..total_seq_len-1
        xq = self.rope(
            xq_reshaped, start_pos=Dim(0), seq_len=total_seq_len
        ).reshape((-1, self.n_heads, self.head_dim))
        xk = self.rope(
            xk_reshaped, start_pos=Dim(0), seq_len=total_seq_len
        ).reshape((-1, self.n_kv_heads, self.head_dim))

        # Compute max sequence length from row offsets
        # input_row_offsets is [batch_size + 1], so differences give sequence lengths
        seq_lens = input_row_offsets[1:] - input_row_offsets[:-1]
        max_seq_len = ops.max(seq_lens, axis=0)

        # Flash attention without cache
        # Note: flash_attention_ragged_gpu expects all tensors to have same num_heads
        # For GQA, we need to expand K and V to match Q's number of heads
        if self.n_kv_heads != self.n_heads:
            # Expand K and V by repeating along head dimension
            n_rep = self.n_heads // self.n_kv_heads
            xk = self._repeat_kv(xk, n_rep)
            xv = self._repeat_kv(xv, n_rep)

        attn_out = flash_attention_ragged_gpu(
            q=xq,
            k=xk,
            v=xv,
            input_row_offsets=input_row_offsets,
            max_seq_len=max_seq_len.to(DeviceRef.CPU()),
            mask_variant=MHAMaskVariant.CAUSAL_MASK,
            scale=self.scale,
        )

        # Reshape back and apply output projection
        attn_out = ops.reshape(attn_out, shape=[total_seq_len, -1])
        return self.o_proj(attn_out)

    def _repeat_kv(self, x: TensorValue, n_rep: int) -> TensorValue:
        """Repeat K or V tensors to match the number of query heads (for GQA).

        Args:
            x: Input tensor of shape [total_seq_len, n_kv_heads, head_dim]
            n_rep: Number of times to repeat each KV head

        Returns:
            Tensor of shape [total_seq_len, n_kv_heads * n_rep, head_dim]
        """
        if n_rep == 1:
            return x

        # Expand and reshape to repeat KV heads
        # [seq_len, n_kv_heads, head_dim] -> [seq_len, n_kv_heads, n_rep, head_dim]
        x_expanded = ops.unsqueeze(x, 2)
        # Tile along the n_rep dimension
        # TODO: Use ops.tile when available, for now we'll use a workaround
        # by concatenating n_rep times
        tensors_to_concat = [x_expanded] * n_rep
        x_repeated = ops.concat(tensors_to_concat, axis=2)
        # Reshape to [seq_len, n_kv_heads * n_rep, head_dim]
        return x_repeated.reshape((-1, self.n_kv_heads * n_rep, self.head_dim))
