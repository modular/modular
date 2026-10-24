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
"""GLM-5.3-Flash vision attention.

Differs from the GLM-4.6V attention this family shares in two ways, both read
off the checkpoint: ``qkv`` and ``proj`` carry biases
(``vision_config.attention_bias`` is true, where GLM-4.6V's is false), and q and
k are RMS-normalized over the head slice. The norms run *before* the 2D rotary
embedding; after it they would be a different function, because rope does not
commute with a per-head rescaling.
"""

from __future__ import annotations

from max.dtype import DType
from max.graph import DeviceRef, TensorValue, ops
from max.nn.kernels import MHAMaskVariant, flash_attention_ragged_gpu
from max.nn.layer import Module
from max.nn.linear import Linear
from max.nn.norm import RMSNorm

__all__ = ["Glm5NextVisionAttention", "apply_rotary_pos_emb_vision"]


def apply_rotary_pos_emb_vision(
    q: TensorValue, k: TensorValue, cos: TensorValue, sin: TensorValue
) -> tuple[TensorValue, TensorValue]:
    """Applies 2D vision RoPE to q and k, rotating the full head.

    Args:
        q: Query of shape ``(seq_len, n_heads, head_dim)``.
        k: Key of shape ``(seq_len, n_heads, head_dim)``.
        cos: Cosine table of shape ``(seq_len, head_dim)``.
        sin: Sine table of shape ``(seq_len, head_dim)``.

    Returns:
        The rotated ``(q, k)``, cast back to their input dtypes.
    """

    def rotate_half(x: TensorValue) -> TensorValue:
        half_dim = x.shape[-1] // 2
        return ops.concat((-x[..., half_dim:], x[..., :half_dim]), -1)

    q_dtype, k_dtype = q.dtype, k.dtype
    q, k = q.cast(DType.float32), k.cast(DType.float32)
    cos = ops.unsqueeze(cos, -2).cast(DType.float32)
    sin = ops.unsqueeze(sin, -2).cast(DType.float32)
    q_embed = q * cos + rotate_half(q) * sin
    k_embed = k * cos + rotate_half(k) * sin
    return q_embed.cast(q_dtype), k_embed.cast(k_dtype)


class Glm5NextVisionAttention(Module):
    """Full (non-windowed) ragged attention over one frame's patches.

    There is no window attention anywhere in this tower: every block attends
    over the patches of a single frame, delimited by ``input_row_offsets``.
    """

    def __init__(
        self,
        dtype: DType,
        device: DeviceRef,
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

        # Named `qkv_proj`, not `qkv`: the weight adapter renames the
        # checkpoint's fused `attn.qkv` onto this.
        self.qkv_proj = Linear(
            in_dim=hidden_size,
            out_dim=3 * hidden_size,
            dtype=dtype,
            device=device,
            has_bias=has_bias,
        )
        self.proj = Linear(
            in_dim=hidden_size,
            out_dim=hidden_size,
            dtype=dtype,
            device=device,
            has_bias=has_bias,
        )
        self.q_norm = RMSNorm(
            dim=self.head_dim,
            dtype=dtype,
            eps=rms_norm_eps,
            multiply_before_cast=False,
        )
        self.k_norm = RMSNorm(
            dim=self.head_dim,
            dtype=dtype,
            eps=rms_norm_eps,
            multiply_before_cast=False,
        )

    def __call__(
        self,
        x: TensorValue,
        position_embeddings: tuple[TensorValue, TensorValue],
        input_row_offsets: TensorValue,
        max_seqlen: TensorValue,
    ) -> TensorValue:
        """Runs attention over ``x``.

        Args:
            x: Input of shape ``(seq_len, hidden_size)``.
            position_embeddings: ``(cos, sin)``, each ``(seq_len, head_dim)``.
            input_row_offsets: ``(num_frames + 1,)`` uint32 prefix sum of
                per-frame patch counts.
            max_seqlen: Longest frame length, as a ``(1,)`` uint32 tensor.

        Returns:
            Attention output of shape ``(seq_len, hidden_size)``.
        """
        seq_len = x.shape[0]
        qkv = self.qkv_proj(x)
        q, k, v = ops.split(qkv, [self.hidden_size] * 3, axis=-1)
        q = q.reshape((seq_len, self.num_heads, self.head_dim))
        k = k.reshape((seq_len, self.num_heads, self.head_dim))
        v = v.reshape((seq_len, self.num_heads, self.head_dim))

        # Before the rotary embedding, not after. Rope does not commute with a
        # per-head rescaling, so the two orders are different functions.
        q = self.q_norm(q)
        k = self.k_norm(k)

        cos, sin = position_embeddings
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
        return self.proj(attn_out.reshape((seq_len, self.hidden_size)))
