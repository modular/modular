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
"""Laguna attention layer (ModuleV3).

GQA attention, causal or sliding-window per the layer's type, with per-head
Q/K RMSNorm, a partial-rotary lane permutation ahead of the fused RoPE store,
and a softplus output gate. See the graph architecture's
:mod:`max.pipelines.architectures.laguna.layers.attention` for the derivation
of each.
"""

from __future__ import annotations

from max.dtype import DType
from max.experimental import functional as F
from max.experimental.nn import Linear, Module
from max.experimental.nn.common_layers.functional_kernels import (
    flash_attention_ragged,
    rope_split_store_ragged,
)
from max.experimental.nn.common_layers.kv_cache import PagedCacheValues
from max.experimental.tensor import Tensor, default_dtype
from max.nn.attention import MHAMaskVariant
from max.nn.kv_cache import KVCacheParams

from .norm import LagunaRMSNorm

# The lane permutation halves both the rotated and pass-through blocks,
# and each half must hold whole (cos, sin) pairs.
_ROTARY_DIM_MULTIPLE = 4


GATING_MODES = ("per-head", "per-element")


def softplus(x: Tensor) -> Tensor:
    """Computes ``log(1 + exp(x))`` without overflowing on large inputs.

    Folding out ``max(x, 0)`` leaves a non-positive exponent argument, so the
    identity is finite where the direct form saturates to ``inf``.
    """
    return F.max(x, 0.0) + F.log1p(F.exp(-F.abs(x)))


class LagunaAttention(Module[..., Tensor]):
    """Per-layer attention with per-head QK-norm and a softplus output gate.

    Every value that changes between layers of one subgraph group -- the
    layer's index into its cache leaf, and the RoPE table -- arrives as a
    ``forward`` argument, never as a constant captured at construction. The
    first layer of a group is what gets traced, so a captured constant would be
    that layer's value for every layer in the group.

    Args:
        num_attention_heads: Query heads in this layer.
        num_key_value_heads: KV heads, uniform across layers.
        hidden_size: Model hidden dimension.
        kv_params: The cache leaf this layer addresses.
        is_sliding: Whether the layer masks to ``local_window_size`` keys.
        local_window_size: The sliding window, ignored on a full layer.
        gating_mode: ``"per-head"`` or ``"per-element"`` output gate.
        rotary_dim: Lanes of each head the RoPE table rotates.
        scale: Attention softmax scale.
        dtype: Storage dtype of the projections.
        norm_dtype: Storage dtype of the Q/K norm gammas.
        qk_norm_eps: Epsilon of the Q/K norms.
    """

    def __init__(
        self,
        *,
        num_attention_heads: int,
        num_key_value_heads: int,
        hidden_size: int,
        kv_params: KVCacheParams,
        is_sliding: bool,
        local_window_size: int,
        gating_mode: str,
        rotary_dim: int,
        scale: float,
        dtype: DType,
        norm_dtype: DType,
        qk_norm_eps: float = 1e-6,
    ) -> None:
        super().__init__()
        head_dim = kv_params.head_dim
        if is_sliding and local_window_size <= 0:
            raise ValueError(
                "A sliding-attention Laguna layer needs a positive window, "
                f"got local_window_size={local_window_size}."
            )
        if gating_mode not in GATING_MODES:
            raise ValueError(
                "Laguna attention gating must be 'per-head' or 'per-element', "
                f"got {gating_mode!r}."
            )
        if not 0 < rotary_dim <= head_dim:
            raise ValueError(
                f"Laguna rotary_dim must be in (0, head_dim={head_dim}], got "
                f"{rotary_dim}."
            )
        if rotary_dim % _ROTARY_DIM_MULTIPLE:
            raise ValueError(
                f"Laguna rotary_dim must be a multiple of {_ROTARY_DIM_MULTIPLE}, got {rotary_dim}."
            )

        self.n_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.kv_params = kv_params
        self.is_sliding = is_sliding
        self.local_window_size = local_window_size if is_sliding else -1
        self.gating_mode = gating_mode
        self.rotary_dim = rotary_dim
        self.scale = scale
        self.dtype = dtype

        q_dim = head_dim * num_attention_heads
        kv_dim = head_dim * num_key_value_heads
        with default_dtype(dtype):
            self.q_proj = Linear(hidden_size, q_dim, bias=False)
            self.k_proj = Linear(hidden_size, kv_dim, bias=False)
            self.v_proj = Linear(hidden_size, kv_dim, bias=False)
            self.o_proj = Linear(q_dim, hidden_size, bias=False)
            self.g_proj = Linear(
                hidden_size,
                num_attention_heads if gating_mode == "per-head" else q_dim,
                bias=False,
            )
        self.q_norm = LagunaRMSNorm(head_dim, eps=qk_norm_eps, dtype=norm_dtype)
        self.k_norm = LagunaRMSNorm(head_dim, eps=qk_norm_eps, dtype=norm_dtype)

    def _per_head_norm(
        self, x: Tensor, norm: LagunaRMSNorm, num_heads: int
    ) -> Tensor:
        """Applies a ``[head_dim]`` norm to each head of a flat projection."""
        head_dim = self.kv_params.head_dim
        normed = norm(x.reshape((-1, num_heads, head_dim)))
        return normed.reshape((-1, num_heads * head_dim))

    def _permute_rotary_lanes(self, x: Tensor, num_heads: int) -> Tensor:
        """Reorders head lanes so the kernel's RoPE pairs match the reference's.

        Interleaves the rotated and pass-through blocks at their midpoints, so
        every kernel pair ``(i, i + head_dim // 2)`` is either one reference
        pair ``(j, j + rotary_dim // 2)`` or two pass-through lanes, which the
        table's identity angles leave alone. ``q . k`` is invariant under a
        permutation shared by q and k, and V is never permuted, so the
        attention output stays in checkpoint lane order.
        """
        head_dim = self.kv_params.head_dim
        half_rotated = self.rotary_dim // 2
        half_passed = (head_dim - self.rotary_dim) // 2
        heads = x.reshape((-1, num_heads, head_dim))
        permuted = F.concat(
            (
                heads[:, :, :half_rotated],
                heads[:, :, self.rotary_dim : self.rotary_dim + half_passed],
                heads[:, :, half_rotated : self.rotary_dim],
                heads[:, :, self.rotary_dim + half_passed :],
            ),
            axis=-1,
        )
        return permuted.reshape((-1, num_heads * head_dim))

    def forward(
        self,
        x: Tensor,
        layer_idx: Tensor,
        kv_collection: PagedCacheValues,
        freqs_cis: Tensor,
        input_row_offsets: Tensor,
    ) -> Tensor:
        """Runs Laguna attention.

        Args:
            x: Normalized hidden states, ``[total_seq_len, hidden_size]``.
            layer_idx: This layer's index within its own cache leaf, a CPU
                ``uint32`` scalar.
            kv_collection: The cache leaf this layer addresses.
            freqs_cis: This layer's RoPE table, ``[positions, head_dim]``.
            input_row_offsets: Ragged row offsets.

        Returns:
            The attention output, ``[total_seq_len, hidden_size]``.
        """
        total_seq_len = x.shape[0]
        head_dim = self.kv_params.head_dim

        q = self._per_head_norm(self.q_proj(x), self.q_norm, self.n_heads)
        k = self._per_head_norm(
            self.k_proj(x), self.k_norm, self.num_key_value_heads
        )
        # After the norms, so the [head_dim] gammas stay in checkpoint order.
        if self.rotary_dim < head_dim:
            q = self._permute_rotary_lanes(q, self.n_heads)
            k = self._permute_rotary_lanes(k, self.num_key_value_heads)
        qkv = F.concat((q, k, self.v_proj(x)), axis=-1)

        # An FP8 cache needs the roped Q in the cache dtype too, so the QK^T
        # dot sees matching operands.
        q_out_dtype = (
            self.kv_params.dtype if self.kv_params.is_fp8_kv_dtype else None
        )
        xq = rope_split_store_ragged(
            kv_params=self.kv_params,
            qkv=qkv,
            input_row_offsets=input_row_offsets,
            freqs_cis=freqs_cis.cast(qkv.dtype).to(qkv.device),
            kv_collection=kv_collection,
            layer_idx=layer_idx,
            n_heads=self.n_heads,
            interleaved=False,
            q_out_dtype=q_out_dtype,
        )
        xq = xq.reshape((-1, self.n_heads, head_dim))

        attn_out = flash_attention_ragged(
            self.kv_params,
            input=xq,
            kv_collection=kv_collection,
            layer_idx=layer_idx,
            input_row_offsets=input_row_offsets,
            mask_variant=(
                MHAMaskVariant.SLIDING_WINDOW_CAUSAL_MASK
                if self.is_sliding
                else MHAMaskVariant.CAUSAL_MASK
            ),
            local_window_size=self.local_window_size,
            scale=self.scale,
            output_dtype=self.dtype,
        )

        # The reference accumulates the gate in float32 and casts back.
        gate = softplus(self.g_proj(x).cast(DType.float32)).cast(attn_out.dtype)
        if self.gating_mode == "per-head":
            gated = attn_out.reshape(
                (total_seq_len, self.n_heads, head_dim)
            ) * F.unsqueeze(gate, -1)
            attn_out = gated.reshape((total_seq_len, -1))
        else:
            attn_out = attn_out.reshape((total_seq_len, -1)) * gate
        return self.o_proj(attn_out)
