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
"""GLM-5.3-Flash vision attention on ModuleV3.

Differs from the GLM-4.6V attention this family shares in two ways, both read
off the checkpoint: ``qkv`` and ``proj`` carry biases
(``vision_config.attention_bias`` is true, where GLM-4.6V's is false), and q and
k are RMS-normalized over the head slice. The norms run *before* the 2D rotary
embedding; after it they would be a different function, because rope does not
commute with a per-head rescaling.
"""

from __future__ import annotations

from max import functional as F
from max.dtype import DType
from max.nn import Linear, Module, RMSNorm
from max.nn.attention_mask import MHAMaskVariant
from max.nn.kernels import flash_attention_ragged_gpu
from max.tensor import Tensor

__all__ = ["Glm5NextVisionAttention", "apply_rotary_pos_emb_vision"]


def _rotate_half(x: Tensor) -> Tensor:
    half_dim = x.shape[-1] // 2
    return F.concat((-x[..., half_dim:], x[..., :half_dim]), axis=-1)


def apply_rotary_pos_emb_vision(
    q: Tensor, k: Tensor, cos: Tensor, sin: Tensor
) -> tuple[Tensor, Tensor]:
    """Applies 2D vision RoPE to q and k, rotating the full head.

    Args:
        q: Query of shape ``(seq_len, n_heads, head_dim)``.
        k: Key of shape ``(seq_len, n_heads, head_dim)``.
        cos: Cosine table of shape ``(seq_len, head_dim)``.
        sin: Sine table of shape ``(seq_len, head_dim)``.

    Returns:
        The rotated ``(q, k)``, cast back to their input dtypes.
    """
    q_dtype, k_dtype = q.dtype, k.dtype
    q, k = q.cast(DType.float32), k.cast(DType.float32)
    cos = cos.unsqueeze(-2).cast(DType.float32)
    sin = sin.unsqueeze(-2).cast(DType.float32)
    q_embed = q * cos + _rotate_half(q) * sin
    k_embed = k * cos + _rotate_half(k) * sin
    return q_embed.cast(q_dtype), k_embed.cast(k_dtype)


class Glm5NextVisionAttention(Module[..., Tensor]):
    """Full (non-windowed) ragged attention over one frame's patches.

    There is no window attention anywhere in this tower: every block attends
    over the patches of a single frame, delimited by ``input_row_offsets``.
    """

    def __init__(
        self,
        *,
        hidden_size: int,
        num_heads: int,
        rms_norm_eps: float,
        has_bias: bool,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.scaling = self.head_dim**-0.5
        # Named `qkv_proj`, not `qkv`: the weight rules rename the
        # checkpoint's fused `attn.qkv` onto this.
        self.qkv_proj = Linear(hidden_size, 3 * hidden_size, bias=has_bias)
        self.proj = Linear(hidden_size, hidden_size, bias=has_bias)
        self.q_norm = RMSNorm(self.head_dim, eps=rms_norm_eps)
        self.k_norm = RMSNorm(self.head_dim, eps=rms_norm_eps)

    def forward(
        self,
        x: Tensor,
        cos: Tensor,
        sin: Tensor,
        input_row_offsets: Tensor,
        max_seqlen: Tensor,
    ) -> Tensor:
        """Runs attention over ``x``.

        Args:
            x: Input of shape ``(seq_len, hidden_size)``.
            cos: Rotary cosine table, ``(seq_len, head_dim)``.
            sin: Rotary sine table, ``(seq_len, head_dim)``.
            input_row_offsets: ``(num_frames + 1,)`` uint32 prefix sum of
                per-frame patch counts.
            max_seqlen: Longest frame length, as a ``(1,)`` uint32 tensor.

        Returns:
            Attention output of shape ``(seq_len, hidden_size)``.
        """
        seq_len = x.shape[0]
        q, k, v = self.qkv_proj(x).split([self.hidden_size] * 3, axis=-1)
        q = q.reshape([seq_len, self.num_heads, self.head_dim])
        k = k.reshape([seq_len, self.num_heads, self.head_dim])
        v = v.reshape([seq_len, self.num_heads, self.head_dim])

        # Before the rotary embedding, not after. Rope does not commute with a
        # per-head rescaling, so the two orders are different functions.
        q = self.q_norm(q)
        k = self.k_norm(k)
        q, k = apply_rotary_pos_emb_vision(q, k, cos, sin)

        attn_out = flash_attention_ragged_gpu(
            q,
            k,
            v,
            input_row_offsets=input_row_offsets,
            max_seq_len=max_seqlen,
            mask_variant=MHAMaskVariant.NULL_MASK,
            scale=self.scaling,
        )
        return self.proj(attn_out.reshape([seq_len, self.hidden_size]))
