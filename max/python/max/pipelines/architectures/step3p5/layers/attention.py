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

"""Step-3.5 Attention with per-layer RoPE, QK norm, sliding window, and head-wise gating."""

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
    fused_qkv_ragged_matmul,
    rms_norm_key_cache,
)
from max.nn.kv_cache import KVCacheParams, PagedCacheValues
from max.nn.layer import Module, Shardable
from max.nn.linear import Linear
from max.nn.norm import RMSNorm
from max.nn.rotary_embedding import RotaryEmbedding


class Step3p5Attention(Module, Shardable):
    """Step-3.5 attention layer with QK norm, head-wise gating, and sliding window support.

    Features vs standard attention:
    - Per-head zero-centered RMSNorm on Q and K (weight_offset=1.0)
    - Head-wise sigmoid attention gate (g_proj)
    - Full or sliding window attention (per-layer)
    - Per-layer RoPE embedding
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
        is_sliding: bool,
        sliding_window: int,
        use_head_wise_attn_gate: bool,
        dtype: DType = DType.float32,
        devices: list[DeviceRef],
        linear_cls: Callable[..., Linear] = Linear,
        scale: float | None = None,
        qk_norm_eps: float = 1e-5,
        norm_dtype: DType | None = None,
    ) -> None:
        super().__init__()
        self.rope = rope
        self.n_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.layer_idx = layer_idx
        self.kv_params = kv_params
        self.devices = devices
        self.hidden_size = hidden_size
        self.dtype = dtype
        self.norm_dtype = norm_dtype if norm_dtype is not None else dtype
        self.linear_cls = linear_cls
        self.is_sliding = is_sliding
        self.sliding_window = sliding_window
        self.use_head_wise_attn_gate = use_head_wise_attn_gate
        self.scale = (
            scale
            if scale is not None
            else math.sqrt(1.0 / self.kv_params.head_dim)
        )
        self.qk_norm_eps = qk_norm_eps
        self._sharding_strategy: ShardingStrategy | None = None

        q_weight_dim = self.kv_params.head_dim * num_attention_heads
        kv_weight_dim = self.kv_params.head_dim * num_key_value_heads

        # QKV projections
        self.q_proj = linear_cls(
            in_dim=hidden_size,
            out_dim=q_weight_dim,
            dtype=dtype,
            device=devices[0],
        )
        self.k_proj = linear_cls(
            in_dim=hidden_size,
            out_dim=kv_weight_dim,
            dtype=dtype,
            device=devices[0],
        )
        self.v_proj = linear_cls(
            in_dim=hidden_size,
            out_dim=kv_weight_dim,
            dtype=dtype,
            device=devices[0],
        )
        self.o_proj = linear_cls(
            in_dim=q_weight_dim,
            out_dim=hidden_size,
            dtype=dtype,
            device=devices[0],
        )

        # Per-head QK norm (zero-centered: weight_offset=1.0)
        self.q_norm = RMSNorm(
            self.kv_params.head_dim,
            dtype=self.norm_dtype,
            eps=self.qk_norm_eps,
            weight_offset=1.0,
            multiply_before_cast=False,
        )
        self.k_norm = RMSNorm(
            self.kv_params.head_dim,
            dtype=self.norm_dtype,
            eps=self.qk_norm_eps,
            weight_offset=1.0,
            multiply_before_cast=False,
        )

        # Head-wise attention gate
        if self.use_head_wise_attn_gate:
            self.g_proj = linear_cls(
                in_dim=hidden_size,
                out_dim=num_attention_heads,
                dtype=dtype,
                device=devices[0],
            )

    @property
    def wqkv(self) -> TensorValue:
        """Concatenated Q, K, V weight matrices."""
        wq: TensorValue = self.q_proj.weight
        wk: TensorValue = self.k_proj.weight
        wv: TensorValue = self.v_proj.weight
        return ops.concat((wq, wk, wv)).to(self.devices[0])

    @property
    def sharding_strategy(self) -> ShardingStrategy | None:
        return self._sharding_strategy

    @sharding_strategy.setter
    def sharding_strategy(self, strategy: ShardingStrategy) -> None:
        num_devices = strategy.num_devices

        if strategy.is_replicate:
            self.q_proj.sharding_strategy = strategy
            self.k_proj.sharding_strategy = strategy
            self.v_proj.sharding_strategy = strategy
            self.o_proj.sharding_strategy = strategy
            self.q_norm.sharding_strategy = strategy
            self.k_norm.sharding_strategy = strategy
            if self.use_head_wise_attn_gate:
                self.g_proj.sharding_strategy = strategy
        elif strategy.is_tensor_parallel:
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
            if self.use_head_wise_attn_gate:
                self.g_proj.sharding_strategy = ShardingStrategy.rowwise(
                    num_devices
                )
        else:
            raise ValueError(
                "Step3p5Attention only supports tensor parallel and replicate "
                "sharding strategies."
            )

        self._sharding_strategy = strategy

    def shard(self, devices: Iterable[DeviceRef]) -> list[Step3p5Attention]:
        if not self._sharding_strategy:
            raise ValueError(
                "Step3p5Attention layer cannot be sharded because no sharding "
                "strategy was provided."
            )

        devices_list = list(devices)
        num_devices = len(devices_list)

        q_proj_shards = self.q_proj.shard(devices_list)
        k_proj_shards = self.k_proj.shard(devices_list)
        v_proj_shards = self.v_proj.shard(devices_list)
        o_proj_shards = self.o_proj.shard(devices_list)
        q_norm_shards = self.q_norm.shard(devices_list)
        k_norm_shards = self.k_norm.shard(devices_list)
        g_proj_shards = (
            self.g_proj.shard(devices_list)
            if self.use_head_wise_attn_gate
            else None
        )

        is_replicate = self._sharding_strategy.is_replicate
        shards: list[Step3p5Attention] = []
        for shard_idx, device in enumerate(devices_list):
            if is_replicate:
                # Replicated: every shard owns the full set of heads.
                sharded_num_heads = self.n_heads
                sharded_num_kv_heads = self.num_key_value_heads
            else:
                head_start, head_end = _compute_shard_range(
                    self.n_heads, shard_idx, num_devices
                )
                sharded_num_heads = head_end - head_start

                kv_head_start, kv_head_end = _compute_shard_range(
                    self.num_key_value_heads, shard_idx, num_devices
                )
                sharded_num_kv_heads = kv_head_end - kv_head_start

            sharded = Step3p5Attention(
                rope=self.rope,
                num_attention_heads=sharded_num_heads,
                num_key_value_heads=sharded_num_kv_heads,
                hidden_size=self.hidden_size,
                kv_params=self.kv_params,
                layer_idx=self.layer_idx,
                is_sliding=self.is_sliding,
                sliding_window=self.sliding_window,
                use_head_wise_attn_gate=self.use_head_wise_attn_gate,
                dtype=self.dtype,
                devices=[device],
                linear_cls=self.linear_cls,
                scale=self.scale,
                qk_norm_eps=self.qk_norm_eps,
                norm_dtype=self.norm_dtype,
            )

            sharded.q_proj = q_proj_shards[shard_idx]
            sharded.k_proj = k_proj_shards[shard_idx]
            sharded.v_proj = v_proj_shards[shard_idx]
            sharded.o_proj = o_proj_shards[shard_idx]
            sharded.q_norm = q_norm_shards[shard_idx]
            sharded.k_norm = k_norm_shards[shard_idx]
            if g_proj_shards is not None:
                sharded.g_proj = g_proj_shards[shard_idx]

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
        """Forward pass.

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
        wqkv = self.wqkv

        # Fused QKV matmul + KV cache write
        xq = fused_qkv_ragged_matmul(
            self.kv_params,
            input=x,
            wqkv=wqkv,
            bias=None,
            input_row_offsets=input_row_offsets,
            kv_collection=kv_collection,
            layer_idx=layer_idx,
            n_heads=self.n_heads,
        )

        # Apply QK norm (zero-centered, weight_offset=1.0)
        xq = xq.reshape((-1, self.n_heads, self.kv_params.head_dim))
        xq = self.q_norm(xq)

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
            weight_offset=1.0,
            multiply_before_cast=False,
            per_head_norm=True,
        )

        # Apply rotary embedding
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

        # Flash Attention with appropriate mask
        mask_variant = (
            MHAMaskVariant.SLIDING_WINDOW_CAUSAL_MASK
            if self.is_sliding
            else MHAMaskVariant.CAUSAL_MASK
        )
        attn_out = flash_attention_ragged(
            self.kv_params,
            input=xq,
            kv_collection=kv_collection,
            layer_idx=layer_idx,
            input_row_offsets=input_row_offsets,
            mask_variant=mask_variant,
            scale=self.scale,
            local_window_size=self.sliding_window if self.is_sliding else 0,
        )

        # Reshape: [total_seq_len, n_heads, head_dim] -> [total_seq_len, n_heads * head_dim]
        attn_out = ops.reshape(attn_out, shape=[total_seq_len, -1])

        # Head-wise attention gate: output *= sigmoid(g_proj(x))
        if self.use_head_wise_attn_gate:
            gate = ops.sigmoid(self.g_proj(x))  # [total_seq_len, n_heads]
            # Expand gate to match attn_out: [total_seq_len, n_heads] -> [total_seq_len, n_heads, 1]
            gate = ops.unsqueeze(gate, -1)
            attn_out_reshaped = attn_out.reshape(
                (total_seq_len, self.n_heads, self.kv_params.head_dim)
            )
            attn_out = (attn_out_reshaped * gate).reshape((total_seq_len, -1))

        return self.o_proj(attn_out)
