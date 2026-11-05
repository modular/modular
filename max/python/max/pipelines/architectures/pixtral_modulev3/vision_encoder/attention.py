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

from max.graph import TensorValue, TensorValueLike, ops
from max.nn import Module
from max.nn.kernels.attention import flash_attention
from max.nn.linear import Linear
from max.tensor import Tensor

from .attention_utils import rotate_half


class Attention(Module[..., Tensor]):
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
    ) -> None:
        super().__init__()

        self.n_heads = n_heads
        self.dim = dim
        self.head_dim = head_dim
        self.q_proj = Linear(in_dim=dim, out_dim=dim, bias=False)
        self.k_proj = Linear(in_dim=dim, out_dim=dim, bias=False)
        self.v_proj = Linear(in_dim=dim, out_dim=dim, bias=False)
        self.o_proj = Linear(in_dim=dim, out_dim=dim, bias=False)

    def apply_rotary_embedding(
        self,
        xq: TensorValue,
        xk: TensorValue,
        cos: TensorValue,
        sin: TensorValue,
    ) -> tuple[TensorValue, TensorValue]:
        """Applies Rotary Position Embedding to the query and key tensors.

        ``xq`` and ``xk`` are ``[batch_size, seq_len, n_heads, head_dim]``;
        ``cos`` and ``sin`` are ``[seq_len, head_dim]``.
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

        ``xq``, ``xk`` and ``xv`` are ``[batch_size, seq_len, n_heads,
        head_dim]``; ``attn_mask`` is ``[batch_size, 1, seq_len, seq_len]``.
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

    def forward(
        self,
        x: Tensor,
        attention_mask: Tensor,
        position_embeddings: tuple[Tensor, Tensor],
    ) -> Tensor:
        """Computes attention on x.

        Args:
            x: Activations with shape (batch, seq_len, dim).
            attention_mask: a mask to ensure different blocks of patches (images)
            can only attend to patches within their respective block (image).
            position_embeddings: (cos, sin) tuple.

        Returns the result of multi-headed self attention on the input.
        """
        batch_size, n_patches = x.shape[0], x.shape[1]
        # matmul weights
        xq = self.q_proj(x)
        xk = self.k_proj(x)
        xv = self.v_proj(x)

        xq_v = xq.reshape([batch_size, n_patches, self.n_heads, self.head_dim])
        xk_v = xk.reshape([batch_size, n_patches, self.n_heads, self.head_dim])
        xv_v = xv.reshape([batch_size, n_patches, self.n_heads, self.head_dim])

        cos, sin = position_embeddings
        xq_r, xk_r = self.apply_rotary_embedding(
            TensorValue(xq_v),
            TensorValue(xk_v),
            TensorValue(cos),
            TensorValue(sin),
        )

        output = self.attention(
            xq_r, xk_r, TensorValue(xv_v), attention_mask
        ).reshape([batch_size, n_patches, -1])
        return self.o_proj(output)  # type: ignore[arg-type]
