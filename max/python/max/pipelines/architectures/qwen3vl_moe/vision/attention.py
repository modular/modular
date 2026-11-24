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
"""The attention layer of the vision tower.

Each attention segment of ``cu_seqlens`` (an image, or a temporal patch of a
clip) attends to itself in full, without a mask.
"""

from __future__ import annotations

import math

from max import functional as F
from max.dtype import DType
from max.nn import Linear, Module
from max.nn.attention_mask import MHAMaskVariant
from max.nn.kernels import flash_attention_ragged_gpu
from max.tensor import Tensor

__all__ = ["VisionAttention", "apply_rotary_pos_emb"]


def _rotate_half(x: Tensor) -> Tensor:
    """Returns ``[-x2, x1]`` for the halves ``[x1, x2]`` of the last axis."""
    half = int(x.shape[-1]) // 2
    return F.concat([-x[:, :, half:], x[:, :, :half]], axis=-1)


def apply_rotary_pos_emb(
    q: Tensor, k: Tensor, cos: Tensor, sin: Tensor
) -> tuple[Tensor, Tensor]:
    """Rotates the query and key by the patches' rotary table, in float32.

    Args:
        q: ``[seq_len, n_heads, head_dim]``.
        k: ``[seq_len, n_heads, head_dim]``.
        cos: ``[seq_len, head_dim]``.
        sin: ``[seq_len, head_dim]``.

    Returns:
        The rotated query and key, in their own dtypes.
    """
    q_dtype, k_dtype = q.dtype, k.dtype
    q, k = q.cast(DType.float32), k.cast(DType.float32)
    cos = cos.unsqueeze(-2).cast(DType.float32)
    sin = sin.unsqueeze(-2).cast(DType.float32)
    q_embed = (q * cos) + (_rotate_half(q) * sin)
    k_embed = (k * cos) + (_rotate_half(k) * sin)
    return q_embed.cast(q_dtype), k_embed.cast(k_dtype)


class VisionAttention(Module[..., Tensor]):
    """Multi-head attention over the tower's ragged attention segments."""

    def __init__(self, hidden_size: int, n_heads: int) -> None:
        self.hidden_size = hidden_size
        self.head_dim = hidden_size // n_heads
        self.scale = math.sqrt(1.0 / self.head_dim)
        # The checkpoint stacks q, k and v in one projection.
        self.qkv_proj = Linear(hidden_size, 3 * hidden_size, bias=True)
        self.proj = Linear(hidden_size, hidden_size, bias=True)

    def forward(
        self,
        x: Tensor,
        cos: Tensor,
        sin: Tensor,
        cu_seqlens: Tensor,
        max_seqlen: Tensor,
    ) -> Tensor:
        """Attends within each segment and projects back.

        Args:
            x: ``[seq_len, hidden_size]``.
            cos: The rotary table's cosines, ``[seq_len, head_dim]``.
            sin: Its sines.
            cu_seqlens: ``[n_segments + 1]`` uint32 segment offsets.
            max_seqlen: The longest segment, ``[1]`` uint32 on the host.
        """
        seq_len = x.shape[0]
        q, k, v = (
            part.reshape([seq_len, -1, self.head_dim])
            for part in F.split(
                self.qkv_proj(x), [self.hidden_size] * 3, axis=-1
            )
        )
        q, k = apply_rotary_pos_emb(q, k, cos, sin)
        out = flash_attention_ragged_gpu(
            q,
            k,
            v,
            input_row_offsets=cu_seqlens,
            max_seq_len=max_seqlen,
            mask_variant=MHAMaskVariant.NULL_MASK,
            scale=self.scale,
        )
        return self.proj(out.reshape([seq_len, -1]))
