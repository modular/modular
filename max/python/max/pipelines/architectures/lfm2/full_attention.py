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
"""LFM2 full-attention block: Qwen3.5-style GQA (optional output gate + partial RoPE).

Vendored here so LFM2 does not depend on the full ``qwen3_5_moe`` architecture package.
Logic matches ``qwen3_5_moe.layers.attention`` for the same checkpoint layout.

Key differences from plain Qwen3 attention:
  - attn_output_gate: q_proj output dim is doubled; the second half is a gate
    applied (after silu) to the attention output before o_proj.
  - Unfused KV path: fused_qkv_ragged_matmul cannot handle a doubled Q dim,
    so Q is computed via q_proj(x) and K/V are written to the paged cache via
    matmul_kv_cache_ragged(concat(k_proj.weight, v_proj.weight)).
  - partial RoPE: freqs_cis.shape[-1] == head_dim * partial_rotary_factor,
    and fused_qk_ragged_rope applies rotation to only those leading dims.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Iterable

from max.dtype import DType
from max.graph import DeviceRef, ShardingStrategy, TensorValue, ops
from max.nn.attention import MHAMaskVariant
from max.nn.attention.attention_with_rope import _compute_shard_range
from max.nn.kernels import (
    flash_attention_ragged,
    fused_qk_ragged_rope,
    matmul_kv_cache_ragged,
    rms_norm_key_cache,
)
from max.nn.kv_cache import KVCacheParams, PagedCacheValues
from max.nn.layer import Module, Shardable
from max.nn.linear import Linear
from max.nn.norm import RMSNorm
from max.nn.rotary_embedding import RotaryEmbedding


class LFM2FullAttention(Module, Shardable):
    """Full-attention (GQA) block for LFM2 decoder layers marked ``full_attention``.

    Used by :class:`~max.pipelines.architectures.lfm2.lfm2.LFM2DecoderLayer`
    for layers where ``layer_types[i] == "full_attention"``.

    Implementation follows the Qwen3.5 MoE full-attention path (per-head Q/K norm,
    optional output gate, unfused KV when gating). It supports tensor parallelism
    and optional ``attn_output_gate``.

    The unfused KV path is used when attn_output_gate=True because
    fused_qkv_ragged_matmul computes KV offsets from n_heads * head_dim and
    would mis-split a doubled Q projection.  K and V are written to the paged
    cache via :func:`~max.nn.kernels.matmul_kv_cache_ragged` instead.
    """

    def __init__(
        self,
        *,
        rope: RotaryEmbedding,
        num_attention_heads: int,
        num_key_value_heads: int,
        hidden_size: int,
        kv_params: KVCacheParams,
        layer_idx: int,
        dtype: DType = DType.float32,
        devices: list[DeviceRef],
        linear_cls: Callable[..., Linear] = Linear,
        scale: float | None = None,
        has_bias: bool = False,
        qk_norm_eps: float = 1e-6,
        norm_dtype: DType | None = None,
        attn_output_gate: bool = False,
        mask_variant: MHAMaskVariant = MHAMaskVariant.CAUSAL_MASK,
        local_window_size: int = 512,
    ) -> None:
        super().__init__()
        self.rope = rope
        self.n_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.layer_idx = layer_idx
        self.kv_params = kv_params
        self.has_bias = has_bias
        self.devices = devices
        self.hidden_size = hidden_size
        self.dtype = dtype
        self.norm_dtype = norm_dtype if norm_dtype is not None else dtype
        self.linear_cls = linear_cls
        self.attn_output_gate = attn_output_gate
        self.mask_variant = mask_variant
        self.local_window_size = local_window_size
        self.scale = (
            scale
            if scale is not None
            else math.sqrt(1.0 / self.kv_params.head_dim)
        )
        self.qk_norm_eps = qk_norm_eps
        self._sharding_strategy: ShardingStrategy | None = None

        # Q/K per-head RMSNorm (Qwen3-style)
        self.q_norm = RMSNorm(
            self.kv_params.head_dim,
            dtype=self.norm_dtype,
            eps=self.qk_norm_eps,
            multiply_before_cast=False,
        )
        self.k_norm = RMSNorm(
            self.kv_params.head_dim,
            dtype=self.norm_dtype,
            eps=self.qk_norm_eps,
            multiply_before_cast=False,
        )

        q_weight_dim = self.kv_params.head_dim * num_attention_heads
        kv_weight_dim = self.kv_params.head_dim * num_key_value_heads

        # When attn_output_gate=True q_proj emits [Q | gate] concatenated.
        q_proj_out_dim = q_weight_dim * 2 if attn_output_gate else q_weight_dim

        self.q_proj = linear_cls(
            in_dim=hidden_size,
            out_dim=q_proj_out_dim,
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

    @property
    def wkv(self) -> TensorValue:
        """Concatenated K and V weights for matmul_kv_cache_ragged."""
        wk: TensorValue = self.k_proj.weight
        wv: TensorValue = self.v_proj.weight
        return ops.concat((wk, wv)).to(self.devices[0])

    @property
    def sharding_strategy(self) -> ShardingStrategy | None:
        return self._sharding_strategy

    @sharding_strategy.setter
    def sharding_strategy(self, strategy: ShardingStrategy) -> None:
        num_devices = strategy.num_devices

        if strategy.is_replicate:
            for proj in (self.q_proj, self.k_proj, self.v_proj, self.o_proj):
                proj.sharding_strategy = strategy
            self.q_norm.sharding_strategy = strategy
            self.k_norm.sharding_strategy = strategy

        elif strategy.is_tensor_parallel:
            # Q projection: when gate is present use gate_up sharding so each
            # shard receives [q_shard | gate_shard] rather than a mixed slice.
            if self.attn_output_gate:
                self.q_proj.sharding_strategy = ShardingStrategy.gate_up(
                    num_devices, axis=0
                )
            else:
                self.q_proj.sharding_strategy = ShardingStrategy.rowwise(
                    num_devices
                )
            self.k_proj.sharding_strategy = ShardingStrategy.rowwise(
                num_devices
            )
            self.v_proj.sharding_strategy = ShardingStrategy.rowwise(
                num_devices
            )
            self.o_proj.sharding_strategy = (
                ShardingStrategy.head_aware_columnwise(
                    num_devices, self.n_heads, self.kv_params.head_dim
                )
            )
            self.q_norm.sharding_strategy = ShardingStrategy.replicate(
                num_devices
            )
            self.k_norm.sharding_strategy = ShardingStrategy.replicate(
                num_devices
            )
        else:
            raise ValueError(
                "LFM2FullAttention supports tensor_parallel and "
                "replicate sharding only."
            )

        self._sharding_strategy = strategy

    def shard(self, devices: Iterable[DeviceRef]) -> list[LFM2FullAttention]:
        """Create sharded copies of this attention layer."""
        if not self._sharding_strategy:
            raise ValueError(
                "LFM2FullAttention cannot be sharded without a "
                "sharding strategy."
            )

        devices_list = list(devices)
        num_devices = len(devices_list)

        q_proj_shards = self.q_proj.shard(devices_list)
        k_proj_shards = self.k_proj.shard(devices_list)
        v_proj_shards = self.v_proj.shard(devices_list)
        o_proj_shards = self.o_proj.shard(devices_list)
        q_norm_shards = self.q_norm.shard(devices_list)
        k_norm_shards = self.k_norm.shard(devices_list)

        shards: list[LFM2FullAttention] = []

        for shard_idx, device in enumerate(devices_list):
            head_start, head_end = _compute_shard_range(
                self.n_heads, shard_idx, num_devices
            )
            kv_head_start, kv_head_end = _compute_shard_range(
                self.num_key_value_heads, shard_idx, num_devices
            )

            sharded = LFM2FullAttention(
                rope=self.rope,
                num_attention_heads=head_end - head_start,
                num_key_value_heads=kv_head_end - kv_head_start,
                hidden_size=self.hidden_size,
                kv_params=self.kv_params,
                layer_idx=self.layer_idx,
                dtype=self.dtype,
                devices=[device],
                linear_cls=self.linear_cls,
                scale=self.scale,
                has_bias=self.has_bias,
                qk_norm_eps=self.qk_norm_eps,
                norm_dtype=self.norm_dtype,
                attn_output_gate=self.attn_output_gate,
                mask_variant=self.mask_variant,
                local_window_size=self.local_window_size,
            )

            sharded.q_proj = q_proj_shards[shard_idx]
            sharded.k_proj = k_proj_shards[shard_idx]
            sharded.v_proj = v_proj_shards[shard_idx]
            sharded.o_proj = o_proj_shards[shard_idx]
            sharded.q_norm = q_norm_shards[shard_idx]
            sharded.k_norm = k_norm_shards[shard_idx]
            shards.append(sharded)

        return shards

    def __call__(
        self,
        layer_idx: TensorValue,
        x: TensorValue,
        kv_collection: PagedCacheValues,
        freqs_cis: TensorValue,
        input_row_offsets: TensorValue,
    ) -> TensorValue:
        """Forward pass through the full-attention block.

        Args:
            layer_idx: KV-cache layer index for this block (0-based over
                full-attention layers only).
            x: Input hidden states ``[total_seq_len, hidden_size]``.
            kv_collection: Paged KV cache handle.
            freqs_cis: RoPE frequency table ``[max_seq_len*2, rope_head_dim]``.
            input_row_offsets: Ragged sequence offsets ``[batch+1]``.

        Returns:
            Output hidden states ``[total_seq_len, hidden_size]``.
        """
        total_seq_len = x.shape[0]

        # Q projection (may include gate in second half).
        # The checkpoint layout is PER-HEAD interleaved:
        #   [Q_h0(head_dim), Gate_h0(head_dim), Q_h1, Gate_h1, ...]
        # We must reshape to [seq_len, n_heads, 2*head_dim] and then slice,
        # NOT take a flat [:q_dim] / [q_dim:] split.
        xq_full = self.q_proj(x)

        gate_3d: TensorValue | None = None
        if self.attn_output_gate:
            # [seq_len, n_heads, 2*head_dim] → split along last dim
            xq_interleaved = xq_full.reshape(
                [-1, self.n_heads, 2 * self.kv_params.head_dim]
            )
            xq = xq_interleaved[:, :, : self.kv_params.head_dim]
            gate_3d = xq_interleaved[:, :, self.kv_params.head_dim :]
        else:
            xq = xq_full

        # Reshape for per-head operations (no-op when gate split already 3D)
        xq = xq.reshape([-1, self.n_heads, self.kv_params.head_dim])

        # Per-head Q norm (Qwen3-style)
        xq = self.q_norm(xq)

        # K/V projection + write to paged KV cache
        matmul_kv_cache_ragged(
            self.kv_params,
            x.cast(self.dtype).to(self.devices[0]),
            input_row_offsets,
            self.wkv,
            kv_collection,
            layer_idx,
        )

        # Per-head K norm in the cache (in-place)
        rms_norm_key_cache(
            self.kv_params,
            kv_collection=kv_collection,
            gamma=self.k_norm.weight.cast(self.kv_params.dtype).to(
                self.devices[0]
            ),
            epsilon=self.qk_norm_eps,
            layer_idx=layer_idx,
            total_seq_len=total_seq_len,
            input_row_offsets=input_row_offsets,
            weight_offset=0.0,
            multiply_before_cast=False,
            per_head_norm=True,
        )

        # Apply rotary embedding (partial RoPE when freqs_cis dim < head_dim)
        freqs_cis = ops.cast(freqs_cis, xq.dtype).to(xq.device)
        xq = fused_qk_ragged_rope(
            self.kv_params,
            xq,
            input_row_offsets,
            kv_collection,
            freqs_cis,
            layer_idx,
            interleaved=self.rope.interleaved,
        )

        # Causal flash attention
        attn_out = flash_attention_ragged(
            self.kv_params,
            input=xq,
            kv_collection=kv_collection,
            layer_idx=layer_idx,
            input_row_offsets=input_row_offsets,
            mask_variant=self.mask_variant,
            scale=self.scale,
            local_window_size=self.local_window_size,
        )

        # Apply output gate before projection (attn_output_gate=True).
        # Reference: attn_output * sigmoid(gate)  (not silu).
        if gate_3d is not None:
            attn_out = attn_out * ops.sigmoid(gate_3d)

        # Output projection
        attn_out = ops.reshape(attn_out, shape=[total_seq_len, -1])
        return self.o_proj(attn_out)
