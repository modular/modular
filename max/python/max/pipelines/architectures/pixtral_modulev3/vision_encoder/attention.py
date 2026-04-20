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


import math

from max.experimental import functional as F
from max.experimental.nn import Module
from max.experimental.nn.linear import Linear
from max.experimental.tensor import Tensor
from max.graph import TensorValue, TensorValueLike, ops

from .attention_utils import rotate_half


class Attention(Module[..., Tensor]):
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
        unsqueeze_dim: int = 0,
    ) -> tuple[TensorValue, TensorValue]:
        """Applies Rotary Position Embedding to the query and key tensors."""
        cos = ops.unsqueeze(cos, unsqueeze_dim)
        sin = ops.unsqueeze(sin, unsqueeze_dim)
        xq = xq.transpose(1, 2)
        xk = xk.transpose(1, 2)

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
        xv = xv.transpose(1, 2)

        scale = math.sqrt(1.0 / self.head_dim)
        scores = xq @ ops.transpose(xk, 2, 3)
        attn_mask = TensorValue(attn_mask)
        # Cast mask to match scores dtype to avoid float32 promotion
        attn_mask = attn_mask.cast(scores.dtype)

        attn_mask = ops.rebind(
            attn_mask, (scores.shape[0], 1, scores.shape[2], scores.shape[3])
        )
        scores = ops.softmax(scores * scale + attn_mask)

        return scores @ xv

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

        xq_v = F.reshape(
            xq, [batch_size, n_patches, self.n_heads, self.head_dim]
        )
        xk_v = F.reshape(
            xk, [batch_size, n_patches, self.n_heads, self.head_dim]
        )
        xv_v = F.reshape(
            xv, [batch_size, n_patches, self.n_heads, self.head_dim]
        )

        cos, sin = position_embeddings
        xq_r, xk_r = self.apply_rotary_embedding(
            TensorValue(xq_v),
            TensorValue(xk_v),
            TensorValue(cos),
            TensorValue(sin),
            unsqueeze_dim=0,
        )

        output = (
            self.attention(xq_r, xk_r, xv_v, attention_mask)
            .transpose(1, 2)
            .reshape([batch_size, n_patches, -1])
        )
        return self.o_proj(output)  # type: ignore[arg-type]
