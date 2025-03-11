# ===----------------------------------------------------------------------=== #
# Copyright (c) 2025, Modular Inc. All rights reserved.
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
"""An attention layer used in Llama 3.2 vision pipeline."""

import math
from dataclasses import dataclass

from max.graph import TensorValue, ops
from max.nn import Linear
from max.nn.layer import Layer


@dataclass
class Attention(Layer):
    n_heads: int
    head_dim: int

    wq: Linear
    wk: Linear
    wv: Linear
    wo: Linear

    def attention(
        self,
        xq: TensorValue,
        xk: TensorValue,
        xv: TensorValue,
        attn_mask: TensorValue,
    ) -> TensorValue:
        # Broadcast the attention mask across heads.
        # Do so in the graph so that the broadcast can be fused into downstream
        # ops.
        batch, _, seq_len, post_seq_len = attn_mask.shape
        attn_mask = attn_mask.broadcast_to(
            (
                batch,
                self.n_heads,
                seq_len,
                post_seq_len,
            )
        )

        scale = math.sqrt(1.0 / self.head_dim)
        scores = xq @ ops.transpose(xk, -2, -1)
        # Note, the graph compiler currently requires the order of operands
        # to be `scores * scale` in order to pattern match the fused attention
        # operator.
        scores = ops.softmax(scores * scale + attn_mask)

        return scores @ xv

    def __call__(
        self, x: TensorValue, attention_mask: TensorValue
    ) -> TensorValue:
        """Computes attention on x, reusing the KV cache.

        Args:
            x: Activations with shape (batch, seq_len, dim).
        Returns the result of multi-headed self attention on the input.
        """
        batch, seq_len = x.shape[0], x.shape[1]
        # matmul weights
        xq = self.wq(x)
        xk = self.wk(x)
        xv = self.wv(x)

        xq = ops.reshape(xq, [batch, seq_len, self.n_heads, self.head_dim])

        xk = ops.reshape(
            xk,
            [
                batch,
                seq_len,
                self.n_heads,
                self.head_dim,
            ],
        )
        xv = ops.reshape(
            xv,
            [
                batch,
                seq_len,
                self.n_heads,
                self.head_dim,
            ],
        )

        xq = xq.transpose(1, 2)
        xk = xk.transpose(1, 2)
        xv = xv.transpose(1, 2)

        # (1, 16, 4128, 80) -> (1, 4128, 16, 80)
        attn_out = self.attention(xq, xk, xv, attention_mask).transpose(1, 2)
        output = attn_out.reshape(
            [batch, seq_len, self.n_heads * self.head_dim]
        )  # (1, 4128, 16 * 80)
        return self.wo(output)
