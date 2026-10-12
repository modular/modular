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
"""Idefics3 vision attention layers (ModuleV3)."""

from __future__ import annotations

from max.experimental import functional as F
from max.experimental.nn import Module
from max.experimental.nn.linear import Linear
from max.experimental.tensor import Tensor
from max.nn.attention.mask_config import MHAMaskVariant
from max.nn.kernels import flash_attention_gpu as _flash_attention_gpu

flash_attention_gpu = F.functional(_flash_attention_gpu)


class Idefics3VisionAttention(Module[[Tensor], Tensor]):
    """Idefics3 vision multi-head attention layer (ModuleV3).

    Standard multi-head self-attention for the vision transformer. Uses
    separate Q, K, V projections with bias (matching SigLIP architecture).

    Naming uses ``out_proj`` (not ``o_proj``) to match the PyTorch/HuggingFace
    checkpoint key names for the vision encoder.
    """

    def __init__(
        self,
        hidden_size: int,
        num_attention_heads: int,
    ) -> None:
        super().__init__()

        if hidden_size % num_attention_heads != 0:
            raise ValueError(
                f"hidden_size must be divisible by num_attention_heads "
                f"(got `hidden_size`: {hidden_size} and "
                f"`num_attention_heads`: {num_attention_heads})."
            )

        self.num_heads = num_attention_heads
        self.head_dim = hidden_size // num_attention_heads
        self.scale = self.head_dim ** (-0.5)
        self.embed_dim = hidden_size

        self.q_proj = Linear(in_dim=hidden_size, out_dim=hidden_size, bias=True)
        self.k_proj = Linear(in_dim=hidden_size, out_dim=hidden_size, bias=True)
        self.v_proj = Linear(in_dim=hidden_size, out_dim=hidden_size, bias=True)
        self.out_proj = Linear(
            in_dim=hidden_size, out_dim=hidden_size, bias=True
        )

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass of the vision attention.

        Args:
            x: Input tensor of shape [batch_size, seq_len, hidden_size].

        Returns:
            Output tensor of shape [batch_size, seq_len, hidden_size].
        """
        batch_size = x.shape[0]
        seq_len = x.shape[1]

        # Project Q, K, V
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        # Reshape to [batch, seq_len, num_heads, head_dim] for flash attention.
        q = F.reshape(q, [batch_size, seq_len, self.num_heads, self.head_dim])
        k = F.reshape(k, [batch_size, seq_len, self.num_heads, self.head_dim])
        v = F.reshape(v, [batch_size, seq_len, self.num_heads, self.head_dim])

        # Flash attention: fused, O(N) memory, no materialized NxN matrix.
        # NULL_MASK = full bidirectional attention (no causal masking).
        attn_output = flash_attention_gpu(
            q,
            k,
            v,
            mask_variant=MHAMaskVariant.NULL_MASK,
            scale=self.scale,
        )

        # Reshape back to [batch, seq_len, hidden_size]
        attn_output = F.reshape(
            attn_output, [batch_size, seq_len, self.embed_dim]
        )

        return self.out_proj(attn_output)
