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

"""Qwen3 Attention Layer with support for tensor parallelism."""

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
from max.nn.kv_cache import (
    KVCacheParams,
    PagedCacheValues,
    uses_opaque,
)
from max.nn.layer import Module, Shardable
from max.nn.linear import Linear
from max.nn.norm import RMSNorm
from max.nn.rotary_embedding import RotaryEmbedding


class Qwen3Attention(Module, Shardable):
    """Implementation of the attention layer for the Qwen3 text model.

    Specifically: applies RMSNorm to the query and key states before applying
    rotary embedding, removes window attention, and has a norm weight offset of 0.0
    (as opposed to 1.0 in Gemma3 implementation).

    This implementation supports tensor parallelism for multi-GPU inference.
    When sharded, the attention heads are distributed across devices.
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
    ) -> None:
        """Initializes the attention layer.

        Args:
            rope: The rope layer to borrow the freq_cis value from.
            num_attention_heads: The number of attention heads.
            num_key_value_heads: Number of key/value heads.
            hidden_size: The dimension of the hidden states.
            kv_params: KV Cache Params, including the number of kv heads, the head dim, and data type.
            layer_idx: The layer number associated with this Attention block.
            dtype: DType of the attention inputs and weights.
            devices: Device to place the weights and run the computation.
            linear_cls: Linear class to use for the projection layers.
            scale: Value used to scale the results of the attention output.
            has_bias: Whether to use an attention bias. Defaults to False.
            qk_norm_eps: Value to use for numerical stability. Defaults to 1e-6.
        """

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
        self.linear_cls = linear_cls
        self.scale = (
            scale
            if scale is not None
            else math.sqrt(1.0 / self.kv_params.head_dim)
        )
        self.qk_norm_eps = qk_norm_eps
        self._sharding_strategy: ShardingStrategy | None = None

        if not uses_opaque(self.kv_params.cache_strategy):
            raise ValueError(
                f"{self.kv_params.cache_strategy} cache strategy, not supported"
                " in Attention layer."
            )

        # Per-head RMSNorm for Q and K (Qwen3-specific)
        self.q_norm = RMSNorm(
            self.kv_params.head_dim,
            dtype=dtype,
            eps=self.qk_norm_eps,
            multiply_before_cast=False,
        )
        self.k_norm = RMSNorm(
            self.kv_params.head_dim,
            dtype=dtype,
            eps=self.qk_norm_eps,
            multiply_before_cast=False,
        )

        q_weight_dim = self.kv_params.head_dim * num_attention_heads
        kv_weight_dim = self.kv_params.head_dim * num_key_value_heads

        # Use Linear layers instead of raw Weights for easier sharding
        self.q_proj = linear_cls(
            in_dim=hidden_size,
            out_dim=q_weight_dim,
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
    def wqkv(self) -> TensorValue:
        """The concatenation of q, k, and v weight vectors."""
        wq: TensorValue = self.q_proj.weight
        wk: TensorValue = self.k_proj.weight
        wv: TensorValue = self.v_proj.weight
        return ops.concat((wq, wk, wv)).to(self.devices[0])

    @property
    def wqkv_bias(self) -> TensorValue | None:
        """The concatenation of q, k, and v bias weight vectors."""
        if not self.has_bias:
            return None
        assert self.q_proj.bias is not None
        assert self.k_proj.bias is not None
        assert self.v_proj.bias is not None
        return ops.concat(
            (self.q_proj.bias, self.k_proj.bias, self.v_proj.bias)
        ).to(self.devices[0])

    @property
    def sharding_strategy(self) -> ShardingStrategy | None:
        """Get the sharding strategy for this module."""
        return self._sharding_strategy

    @sharding_strategy.setter
    def sharding_strategy(self, strategy: ShardingStrategy) -> None:
        """Set the sharding strategy for this module.

        For tensor parallelism:
        - Q, K, V projections are sharded rowwise (split output heads)
        - O projection is sharded columnwise (split input, requires allreduce)
        - Q/K norms are replicated (shared across heads)

        Args:
            strategy: The sharding strategy to apply.
        """
        num_devices = strategy.num_devices

        if strategy.is_replicate:
            # Replicate all weights
            self.q_proj.sharding_strategy = strategy
            self.k_proj.sharding_strategy = strategy
            self.v_proj.sharding_strategy = strategy
            self.o_proj.sharding_strategy = strategy
            self.q_norm.sharding_strategy = strategy
            self.k_norm.sharding_strategy = strategy

        elif strategy.is_tensor_parallel:
            # Q, K, V: rowwise sharding (split heads across devices)
            self.q_proj.sharding_strategy = ShardingStrategy.rowwise(
                num_devices
            )
            self.k_proj.sharding_strategy = ShardingStrategy.rowwise(
                num_devices
            )
            self.v_proj.sharding_strategy = ShardingStrategy.rowwise(
                num_devices
            )

            # O: columnwise with head-aware sharding
            # This ensures proper splitting when heads are distributed
            self.o_proj.sharding_strategy = (
                ShardingStrategy.head_aware_columnwise(
                    num_devices, self.n_heads, self.kv_params.head_dim
                )
            )

            # Norms are replicated (they operate per-head, same weights everywhere)
            self.q_norm.sharding_strategy = ShardingStrategy.replicate(
                num_devices
            )
            self.k_norm.sharding_strategy = ShardingStrategy.replicate(
                num_devices
            )
        else:
            raise ValueError(
                "Qwen3Attention only supports tensor parallel and replicate "
                "sharding strategies."
            )

        self._sharding_strategy = strategy

    def shard(self, devices: Iterable[DeviceRef]) -> list[Qwen3Attention]:
        """Create sharded views of this attention layer across multiple devices.

        Each shard contains a subset of attention heads and operates independently.
        After forward pass, outputs must be combined via allreduce.

        Args:
            devices: Iterable of devices to place the shards on.

        Returns:
            List of sharded Qwen3Attention instances, one per device.

        Raises:
            ValueError: If no sharding strategy was set.
        """
        if not self._sharding_strategy:
            raise ValueError(
                "Qwen3Attention layer cannot be sharded because no sharding "
                "strategy was provided."
            )

        devices_list = list(devices)
        num_devices = len(devices_list)

        # Shard all projection layers
        q_proj_shards = self.q_proj.shard(devices_list)
        k_proj_shards = self.k_proj.shard(devices_list)
        v_proj_shards = self.v_proj.shard(devices_list)
        o_proj_shards = self.o_proj.shard(devices_list)

        # Shard norm layers (replicated)
        q_norm_shards = self.q_norm.shard(devices_list)
        k_norm_shards = self.k_norm.shard(devices_list)

        shards: list[Qwen3Attention] = []

        for shard_idx, device in enumerate(devices_list):
            # Compute the number of attention heads for this shard
            head_start, head_end = _compute_shard_range(
                self.n_heads, shard_idx, num_devices
            )
            sharded_num_heads = head_end - head_start

            # Compute the number of KV heads for this shard
            kv_head_start, kv_head_end = _compute_shard_range(
                self.num_key_value_heads, shard_idx, num_devices
            )
            sharded_num_kv_heads = kv_head_end - kv_head_start

            # Create a new attention layer for this shard
            sharded = Qwen3Attention(
                rope=self.rope,
                num_attention_heads=sharded_num_heads,
                num_key_value_heads=sharded_num_kv_heads,
                hidden_size=self.hidden_size,
                kv_params=self.kv_params,
                layer_idx=self.layer_idx,
                dtype=self.dtype,
                devices=[device],
                linear_cls=self.linear_cls,
                scale=self.scale,
                has_bias=self.has_bias,
                qk_norm_eps=self.qk_norm_eps,
            )

            # Replace the projection layers with sharded versions
            sharded.q_proj = q_proj_shards[shard_idx]
            sharded.k_proj = k_proj_shards[shard_idx]
            sharded.v_proj = v_proj_shards[shard_idx]
            sharded.o_proj = o_proj_shards[shard_idx]

            # Replace norm layers with sharded versions
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
        """Forward pass through the attention layer.

        Args:
            layer_idx: Layer index for KV cache.
            x: Input hidden states of shape [total_seq_len, hidden_size].
            kv_collection: KV cache handle.
            freqs_cis: RoPE frequency table.
            input_row_offsets: Ragged offsets for batched sequences.

        Returns:
            Output hidden states of shape [total_seq_len, hidden_size].
        """
        total_seq_len = x.shape[0]

        # Call into fused qkv ragged matmul
        wqkv = self.wqkv
        xq = fused_qkv_ragged_matmul(
            self.kv_params,
            input=x,
            wqkv=wqkv,
            bias=self.wqkv_bias,
            input_row_offsets=input_row_offsets,
            kv_collection=kv_collection,
            layer_idx=layer_idx,
            n_heads=self.n_heads,
        )

        # Apply QK norm to query and key states before RoPE (Qwen3-specific)
        xq = xq.reshape((-1, self.n_heads, self.kv_params.head_dim))
        xq = self.q_norm(xq)

        # Apply K norm in-place on the KV cache
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

        # Calculate Flash Attention
        # NOTE: Qwen3 never uses sliding window pattern
        attn_out = flash_attention_ragged(
            self.kv_params,
            input=xq,
            kv_collection=kv_collection,
            layer_idx=layer_idx,
            input_row_offsets=input_row_offsets,
            mask_variant=MHAMaskVariant.CAUSAL_MASK,
            scale=self.scale,
        )

        # Output projection
        attn_out = ops.reshape(attn_out, shape=[total_seq_len, -1])
        return self.o_proj(attn_out)
