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

"""Qwen3.5 full attention layer.

Differences from Qwen3 attention:
- q_proj outputs 2x width: [hidden_size -> num_heads * head_dim * 2] where the
  extra half is a sigmoid gate applied to the attention output.
- Partial RoPE: only partial_rotary_factor * head_dim dimensions get rotation.
- RMSNorm on Q/K uses (1 + weight) offset (weight_offset=1.0).
"""

from __future__ import annotations

import math
from collections.abc import Callable

from max.dtype import DType
from max.graph import DeviceRef, TensorValue, ops
from max.nn.attention import MHAMaskVariant
from max.nn.kernels import (
    flash_attention_ragged,
    fused_qk_ragged_rope,
    store_k_cache_ragged,
    store_v_cache_ragged,
)
from max.nn.kv_cache import KVCacheParams, PagedCacheValues
from max.nn.layer import Module
from max.nn.linear import Linear
from max.nn.norm import RMSNorm
from max.nn.rotary_embedding import RotaryEmbedding


class Qwen3_5Attention(Module):
    """Full attention layer for Qwen3.5 with gated output and partial RoPE.

    This attention layer differs from standard GQA in several ways:
    1. q_proj produces 2x output width - the second half is a sigmoid gate
       applied to the attention output before the output projection.
    2. Only partial_rotary_factor (25%) of head_dim gets rotary embedding.
    3. QK RMSNorm uses (1 + weight) scaling (weight_offset=1.0).
    """

    def __init__(
        self,
        *,
        rope: RotaryEmbedding,
        num_attention_heads: int,
        num_key_value_heads: int,
        hidden_size: int,
        head_dim: int,
        kv_params: KVCacheParams,
        layer_idx: int,
        dtype: DType = DType.float32,
        devices: list[DeviceRef],
        linear_cls: Callable[..., Linear] = Linear,
        scale: float | None = None,
        partial_rotary_factor: float = 0.25,
        has_bias: bool = False,
        norm_dtype: DType | None = None,
        norm_eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.rope = rope
        self.n_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.head_dim = head_dim
        self.layer_idx = layer_idx
        self.kv_params = kv_params
        self.has_bias = has_bias
        self.devices = devices
        self.hidden_size = hidden_size
        self.dtype = dtype
        self.partial_rotary_factor = partial_rotary_factor
        self.rotary_dim = int(head_dim * partial_rotary_factor)
        self.norm_dtype = norm_dtype if norm_dtype is not None else dtype
        self.scale = (
            scale if scale is not None else math.sqrt(1.0 / self.head_dim)
        )

        # QK norm with (1 + weight) offset
        self.q_norm = RMSNorm(
            self.head_dim,
            dtype=self.norm_dtype,
            eps=norm_eps,
            weight_offset=1.0,
            multiply_before_cast=False,
        )
        self.k_norm = RMSNorm(
            self.head_dim,
            dtype=self.norm_dtype,
            eps=norm_eps,
            weight_offset=1.0,
            multiply_before_cast=False,
        )

        q_weight_dim = head_dim * num_attention_heads
        kv_weight_dim = head_dim * num_key_value_heads

        # q_proj outputs 2x width when gated (query + gate)
        self.q_proj = linear_cls(
            in_dim=hidden_size,
            out_dim=q_weight_dim * 2,  # 2x for query + gate
            dtype=dtype,
            device=devices[0],
            has_bias=has_bias,
        )
        self.k_proj = linear_cls(
            in_dim=hidden_size,
            out_dim=kv_weight_dim,
            dtype=dtype,
            device=devices[0],
            has_bias=has_bias,
        )
        self.v_proj = linear_cls(
            in_dim=hidden_size,
            out_dim=kv_weight_dim,
            dtype=dtype,
            device=devices[0],
            has_bias=has_bias,
        )
        self.o_proj = linear_cls(
            in_dim=q_weight_dim,
            out_dim=hidden_size,
            dtype=dtype,
            device=devices[0],
            has_bias=has_bias,
        )

    def __call__(
        self,
        layer_idx: TensorValue,
        x: TensorValue,
        kv_collection: PagedCacheValues,
        freqs_cis: TensorValue,
        input_row_offsets: TensorValue,
    ) -> TensorValue:
        """Forward pass through the gated full attention layer.

        Args:
            layer_idx: Layer index for KV cache.
            x: Input hidden states [total_seq_len, hidden_size].
            kv_collection: KV cache handle.
            freqs_cis: RoPE frequency table.
            input_row_offsets: Ragged offsets for batched sequences.

        Returns:
            Output hidden states [total_seq_len, hidden_size].
        """
        total_seq_len = x.shape[0]

        # Q projection: produces [total_seq_len, n_heads * head_dim * 2]
        q_out = self.q_proj(x)

        # Split into query and gate.
        # The weight layout is interleaved per head:
        # [head0_query, head0_gate, head1_query, head1_gate, ...]
        # Reshape to [total_seq_len, n_heads, head_dim * 2], then split on last axis.
        q_out_reshaped = ops.reshape(
            q_out, shape=[-1, self.n_heads, self.head_dim * 2]
        )
        # query: [total_seq_len, n_heads, head_dim]
        query = ops.slice_tensor(
            q_out_reshaped,
            [slice(None), slice(None), slice(0, self.head_dim)],
        )
        # gate: [total_seq_len, n_heads, head_dim] -> flatten to [total_seq_len, n_heads * head_dim]
        gate = ops.reshape(
            ops.slice_tensor(
                q_out_reshaped,
                [
                    slice(None),
                    slice(None),
                    slice(self.head_dim, self.head_dim * 2),
                ],
            ),
            shape=[-1, self.n_heads * self.head_dim],
        )

        # K, V projections
        key = self.k_proj(x)
        value = self.v_proj(x)

        # query is already [total_seq_len, n_heads, head_dim] from the reshape above

        # Apply Q/K norms in original dim ordering (weight elements align
        # with the HF convention where RoPE dims come first).
        query = self.q_norm(query)

        # Reshape K for per-head norm: [total_seq_len, n_kv_heads, head_dim]
        key = ops.reshape(
            key, shape=[-1, self.num_key_value_heads, self.head_dim]
        )
        key = self.k_norm(key)

        # Reshape V: [total_seq_len, n_kv_heads, head_dim]
        value = ops.reshape(
            value, shape=[-1, self.num_key_value_heads, self.head_dim]
        )

        # Rearrange Q and K head dims for the fused RoPE kernel.
        #
        # Two transformations needed:
        # 1. NoPE/RoPE swap: HF puts RoPE dims first [RoPE_64 | NoPE_192],
        #    but the kernel rotates the LAST dims: [NoPE_192 | RoPE_64].
        # 2. Interleave RoPE dims: HF uses rotate_half (non-interleaved)
        #    which pairs (dim_i, dim_{i+D/2}), but the kernel with
        #    interleaved=True pairs consecutive dims (dim_{2i}, dim_{2i+1}).
        #    Rearrange [x0,..,x31,x32,..,x63] → [x0,x32,x1,x33,..,x31,x63]
        #    so the kernel's consecutive-pair rotation matches HF's halves.
        #
        # The interleaving is equivalent to: reshape [rd] → [2, half_rd],
        # transpose to [half_rd, 2], flatten back to [rd]. This uses fewer
        # graph ops than the slice+concat approach (5 vs 9 per Q/K).
        rd = self.rotary_dim  # 64
        half_rd = rd // 2  # 32
        q_rope = ops.slice_tensor(
            query, [slice(None), slice(None), slice(0, rd)]
        )
        q_pass = ops.slice_tensor(
            query, [slice(None), slice(None), slice(rd, self.head_dim)]
        )
        q_rope_interleaved = ops.reshape(
            ops.transpose(
                ops.reshape(q_rope, [-1, self.n_heads, 2, half_rd]),
                -1,
                -2,
            ),
            [-1, self.n_heads, rd],
        )
        query = ops.concat([q_pass, q_rope_interleaved], axis=-1)

        k_rope = ops.slice_tensor(key, [slice(None), slice(None), slice(0, rd)])
        k_pass = ops.slice_tensor(
            key, [slice(None), slice(None), slice(rd, self.head_dim)]
        )
        k_rope_interleaved = ops.reshape(
            ops.transpose(
                ops.reshape(k_rope, [-1, self.num_key_value_heads, 2, half_rd]),
                -1,
                -2,
            ),
            [-1, self.num_key_value_heads, rd],
        )
        key = ops.concat([k_pass, k_rope_interleaved], axis=-1)

        # Write rearranged, normed K and V to cache.
        store_k_cache_ragged(kv_collection, key, input_row_offsets, layer_idx)
        store_v_cache_ragged(kv_collection, value, input_row_offsets, layer_idx)

        # Apply RoPE (kernel rotates last rotary_dim dims of Q and K in cache)
        freqs_cis = ops.cast(freqs_cis, query.dtype).to(query.device)
        query = fused_qk_ragged_rope(
            self.kv_params,
            query,
            input_row_offsets,
            kv_collection,
            freqs_cis,
            layer_idx,
            interleaved=self.rope.interleaved,
        )

        # Flash attention
        attn_out = flash_attention_ragged(
            self.kv_params,
            input=query,
            kv_collection=kv_collection,
            layer_idx=layer_idx,
            input_row_offsets=input_row_offsets,
            mask_variant=MHAMaskVariant.CAUSAL_MASK,
            scale=self.scale,
        )

        # Reshape attention output: [total_seq_len, n_heads * head_dim]
        attn_out = ops.reshape(attn_out, shape=[total_seq_len, -1])

        # Apply sigmoid gate
        gate_sigmoid = ops.sigmoid(gate)
        attn_out = attn_out * gate_sigmoid

        # Output projection
        return self.o_proj(attn_out)
