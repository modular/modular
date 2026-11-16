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
"""Pixtral's vision attention."""

import math

from max.dtype import DType
from max.graph import DeviceRef, TensorValue, TensorValueLike, ops
from max.nn._graph_ops import flash_attention
from max.nn.legacy.layer import Module
from max.nn.legacy.linear import Linear

from .attention_utils import rotate_half


class Attention(Module):
    """Pixtral's vision self attention."""

    n_heads: int
    dim: int
    head_dim: int  # hidden_size // self.n_heads

    k_proj: Linear
    v_proj: Linear
    q_proj: Linear
    o_proj: Linear

    def __init__(
        self,
        n_heads: int,
        dim: int,
        head_dim: int,
        dtype: DType,
        device: DeviceRef,
    ) -> None:
        super().__init__()

        self.n_heads = n_heads
        self.dim = dim
        self.head_dim = head_dim
        self.q_proj = Linear(
            dim, dim, dtype=dtype, device=device, has_bias=False
        )
        self.k_proj = Linear(
            dim, dim, dtype=dtype, device=device, has_bias=False
        )
        self.v_proj = Linear(
            dim, dim, dtype=dtype, device=device, has_bias=False
        )
        self.o_proj = Linear(
            dim, dim, dtype=dtype, device=device, has_bias=False
        )

    def apply_rotary_embedding(
        self,
        xq: TensorValue,
        xk: TensorValue,
        cos: TensorValue,
        sin: TensorValue,
    ) -> tuple[TensorValue, TensorValue]:
        """Applies Rotary Position Embedding to the query and key tensors.

        Args:
            xq: The query tensor, of shape
                ``[batch_size, seq_len, n_heads, head_dim]``.
            xk: The key tensor, with the same shape as ``xq``.
            cos: The cosine part of the rotary embedding, of shape
                ``[seq_len, head_dim]``.
            sin: The sine part of the rotary embedding, of shape
                ``[seq_len, head_dim]``.

        Returns:
            The query and key tensors rotated using the Rotary Position
            Embedding, in the layout of ``xq`` and ``xk``.
        """
        cos = ops.unsqueeze(cos, 1)
        sin = ops.unsqueeze(sin, 1)

        q_embed = (xq * cos) + (rotate_half(xq) * sin)
        k_embed = (xk * cos) + (rotate_half(xk) * sin)
        return q_embed, k_embed

    def attention(
        self,
        xq: TensorValue,
        xk: TensorValue,
        xv: TensorValue,
        attn_mask: TensorValueLike,
    ) -> TensorValue:
        """Runs the attention over the query, key and value.

        Args:
            xq: The query tensor, of shape
                ``[batch_size, seq_len, n_heads, head_dim]``.
            xk: The key tensor, with the same shape as ``xq``.
            xv: The value tensor, with the same shape as ``xq``.
            attn_mask: The additive mask, of shape
                ``[batch_size, 1, seq_len, seq_len]``. It keeps each image's
                patches from attending to other images' patches.

        Returns:
            The attention output, with the same shape as ``xq``.
        """
        attn_mask = TensorValue(attn_mask)
        attn_mask = ops.rebind(
            attn_mask, (xq.shape[0], 1, xq.shape[1], xk.shape[1])
        )
        return flash_attention(
            xq,
            xk,
            xv,
            scale=math.sqrt(1.0 / self.head_dim),
            mask=ops.squeeze(attn_mask, 1),
        )

    def __call__(
        self,
        x: TensorValue,
        attention_mask: TensorValueLike,
        position_embeddings: tuple[TensorValue, TensorValue],
    ) -> TensorValue:
        """Computes attention on x.

        Args:
            x: Activations with shape (batch, seq_len, dim).
            attention_mask: a mask to ensure different blocks of patches (images)
            can only attend to patches within their respective block (image).
            position_embeddings: The rotary embedding's cosine and sine
                tables, applied to the query and key.

        Returns:
            The result of multi-headed self attention on the input.
        """
        batch_size, n_patches = x.shape[0], x.shape[1]
        # matmul weights
        xq = self.q_proj(x)
        xk = self.k_proj(x)
        xv = self.v_proj(x)

        xq = ops.reshape(
            xq, [batch_size, n_patches, self.n_heads, self.head_dim]
        )
        xk = ops.reshape(
            xk, [batch_size, n_patches, self.n_heads, self.head_dim]
        )
        xv = ops.reshape(
            xv, [batch_size, n_patches, self.n_heads, self.head_dim]
        )

        cos, sin = position_embeddings
        xq, xk = self.apply_rotary_embedding(xq, xk, cos, sin)

        output = self.attention(xq, xk, xv, attention_mask).reshape(
            [batch_size, n_patches, -1]
        )
        return self.o_proj(output)
