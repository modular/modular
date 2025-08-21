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

"""Gemma3 Attention Layer."""

from __future__ import annotations

import math
from collections.abc import Iterable
from typing import Callable

from max.dtype import DType
from max.graph import (
    DeviceRef,
    ShardingStrategy,
    TensorValue,
    _OpaqueValue,
    ops,
)
from max.nn.attention import MHAMaskVariant
from max.nn.float8_config import Float8Config
from max.nn.kernels import (
    flash_attention_ragged,
    fused_qk_ragged_rope,
    fused_qkv_ragged_matmul,
    fused_qkv_ragged_matmul_scaled_float8,
    quantize_dynamic_scaled_float8,
    quantize_static_scaled_float8,
    rms_norm_key_cache,
)
from max.nn.kv_cache import KVCacheParams, PagedKVCacheCollection
from max.nn.layer import Module, Shardable
from max.nn.linear import Linear
from max.nn.rotary_embedding import Llama3RotaryEmbedding
from max.pipelines.architectures.gemma3.layers.rms_norm import Gemma3RMSNorm


def compute_heads_per_device(
    *, total_heads: int, device_idx: int, num_devices: int
) -> int:
    """Computes the number of attention heads per device for sharding.

    This function calculates the number of heads for a given device, enforcing
    that the total number of heads is evenly divisible by the number of devices.
    Uneven distribution is disallowed to prevent workload imbalance.

    Args:
        total_heads: The total number of attention heads.
        device_idx: The index of the current device (0-indexed).
        num_devices: The total number of devices for sharding.

    Returns:
        The number of heads assigned to the specified device.

    Raises:
        ValueError: If `total_heads` is not evenly divisible by `num_devices`.
    """
    base_heads, remainder = divmod(total_heads, num_devices)
    if device_idx < remainder:
        raise ValueError(
            "An uneven distribution of heads is not supported as it will cause a workload imbalance."
        )
    else:
        return base_heads


class Gemma3Attention(Module, Shardable):
    """Implementation of the attention layer for the Gemma3 text model."""

    def __init__(
        self,
        *,
        rope_global: Llama3RotaryEmbedding,
        rope_local: Llama3RotaryEmbedding,
        num_attention_heads: int,
        num_key_value_heads: int,
        hidden_size: int,
        kv_params: KVCacheParams,
        layer_idx: int,
        sliding_window_pattern: int = 6,
        dtype: DType = DType.float32,
        devices: list[DeviceRef],
        linear_cls: Callable[..., Linear] = Linear,
        scale: float | None = None,
        has_bias: bool = False,
        qk_norm_eps: float = 1e-6,
        local_window_size: int = 1024,
        float8_config: Float8Config | None = None,
    ) -> None:
        """Initializes the attention layer.

        Args:
            rope_global: Rotary embedding used for global (non-sliding window)
                attention layers.
            rope_local: Rotary embedding used for sliding window attention
                layers.
            num_attention_heads: The number of attention heads.
            num_key_value_heads: The number of key/value heads.
            hidden_size: The dimension of the hidden states.
            kv_params: KV Cache Params, including the number of kv heads, the
                head dim, and data type.
            layer_idx: The layer number associated with this Attention block.
            dtype: DType of the attention inputs and weights.
            devices: Device to place the weights and run the computation. If
                multiple are provided, the first device is used. Use
                `DistributedAttentionWithRope` to use all devices during
                attention computation.
            linear_cls: Linear class to use for the outputs dense layer.
            scale: Value used to scale the results of the attention output.
            has_bias: Whether to use an attention bias. Defaults to False.
            qk_norm_eps: Value to use for numerical stability. Defaults to 1e-6.
            float8_config: Float8 quantization configuration. Defaults to None.
        """

        super().__init__()
        self.rope_global = rope_global
        self.rope_local = rope_local
        self.n_heads = num_attention_heads
        self.layer_idx = layer_idx
        self.kv_params = kv_params
        self.has_bias = has_bias
        self.devices = devices
        self._sharding_strategy: ShardingStrategy | None = None
        self.scale = (
            scale
            if scale is not None
            else math.sqrt(1.0 / self.kv_params.head_dim)
        )
        self.local_window_size = local_window_size
        self.sliding_window_pattern = sliding_window_pattern
        self.qk_norm_eps = qk_norm_eps
        self.float8_config = float8_config

        if not self.kv_params.cache_strategy.uses_opaque():
            raise ValueError(
                f"{self.kv_params.cache_strategy} cache strategy, not supported"
                " in Attention layer."
            )

        self.q_norm = Gemma3RMSNorm(
            self.kv_params.head_dim, DType.bfloat16, self.qk_norm_eps
        )
        self.k_norm = Gemma3RMSNorm(
            self.kv_params.head_dim, DType.bfloat16, self.qk_norm_eps
        )
        self.q_weight_dim = self.kv_params.head_dim * num_attention_heads
        self.kv_weight_dim = self.kv_params.head_dim * num_key_value_heads

        self.q_proj = linear_cls(
            in_dim=hidden_size,
            out_dim=self.q_weight_dim,
            dtype=dtype,
            device=devices[0],
            has_bias=has_bias,
            float8_config=float8_config,
        )
        self.k_proj = linear_cls(
            in_dim=hidden_size,
            out_dim=self.kv_weight_dim,
            dtype=dtype,
            device=devices[0],
            has_bias=has_bias,
            float8_config=float8_config,
        )
        self.v_proj = linear_cls(
            in_dim=hidden_size,
            out_dim=self.kv_weight_dim,
            dtype=dtype,
            device=devices[0],
            has_bias=has_bias,
            float8_config=float8_config,
        )

        self.o_proj = linear_cls(
            in_dim=self.q_weight_dim,
            out_dim=hidden_size,
            dtype=dtype,
            device=devices[0],
            float8_config=float8_config,
        )

    @property
    def wqkv(self) -> TensorValue:
        """The concatenation of q, k, and v weight vectors."""
        wq: TensorValue = self.q_proj.weight
        wk: TensorValue = self.k_proj.weight
        wv: TensorValue = self.v_proj.weight
        wqkv = ops.concat((wq, wk, wv))
        if self.float8_config and self.float8_config.is_static:
            # Float8 always has a weight scale.
            assert self.qkv_weight_scale is not None
            wqkv = quantize_static_scaled_float8(
                wqkv, self.qkv_weight_scale.to(DeviceRef.CPU())
            )
        return wqkv.to(self.devices[0])

    @property
    def wqkv_bias(self) -> TensorValue | None:
        """The concatenation of q, k, and v bias weight vectors."""
        if not self.has_bias:
            return None

        # Access bias from Linear layers
        assert self.q_proj.bias is not None
        assert self.k_proj.bias is not None
        assert self.v_proj.bias is not None
        return ops.concat(
            (self.q_proj.bias, self.k_proj.bias, self.v_proj.bias)
        )

    @property
    def qkv_input_scale(self) -> TensorValue | None:
        """The max of q, k, and v scale input vectors."""
        if not self.float8_config or self.float8_config.is_dynamic:
            return None

        assert self.q_proj.input_scale is not None
        assert self.k_proj.input_scale is not None
        assert self.v_proj.input_scale is not None

        return ops.max(
            ops.concat(
                (
                    self.q_proj.input_scale.reshape((1,)),
                    self.k_proj.input_scale.reshape((1,)),
                    self.v_proj.input_scale.reshape((1,)),
                )
            )
        ).reshape(())

    @property
    def qkv_weight_scale(self) -> TensorValue:
        """The max of q, k, and v scale weight vectors."""
        assert self.float8_config

        assert self.q_proj.weight_scale is not None
        assert self.k_proj.weight_scale is not None
        assert self.v_proj.weight_scale is not None

        q_scale: TensorValue = self.q_proj.weight_scale
        k_scale: TensorValue = self.k_proj.weight_scale
        v_scale: TensorValue = self.v_proj.weight_scale
        if len(q_scale.shape) == 0:
            q_scale = q_scale.reshape((1,))
        if len(k_scale.shape) == 0:
            k_scale = k_scale.reshape((1,))
        if len(v_scale.shape) == 0:
            v_scale = v_scale.reshape((1,))

        weight_scale = ops.concat((q_scale, k_scale, v_scale))

        if self.float8_config.is_dynamic:
            # In the dynamic scaling case, return the weight scales directly.
            return weight_scale

        assert self.float8_config.is_static
        # Otherwise, return a scalar max QKV weight scale in the static case.
        return ops.max(weight_scale).reshape([])

    def __call__(
        self,
        x: TensorValue,
        kv_collection: PagedKVCacheCollection,
        **kwargs,
    ) -> TensorValue:
        # Get attributes from input.
        total_seq_len = x.shape[0]

        layer_idx = ops.constant(
            self.layer_idx, DType.uint32, device=DeviceRef.CPU()
        )

        if self.float8_config:
            assert isinstance(kv_collection, PagedKVCacheCollection) or (
                isinstance(kv_collection, _OpaqueValue)
                and kv_collection.type.name == "PagedKVCacheCollection"
            )

            x_scales: TensorValue
            weight_scale = self.qkv_weight_scale
            if self.float8_config.is_static:
                assert self.qkv_input_scale is not None
                x = quantize_static_scaled_float8(
                    x, self.qkv_input_scale.to(DeviceRef.CPU())
                )
                x_scales = self.qkv_input_scale
            else:
                x, x_scales = quantize_dynamic_scaled_float8(
                    x,
                    self.float8_config.input_scale,
                    self.float8_config.weight_scale,
                    scales_type=weight_scale.dtype,
                )

            xq = fused_qkv_ragged_matmul_scaled_float8(
                self.kv_params,
                input=x,
                wqkv=self.wqkv,
                bias=self.wqkv_bias,
                input_row_offsets=kwargs["input_row_offsets"],
                kv_collection=kv_collection,
                layer_idx=layer_idx,
                n_heads=self.n_heads,
                input_scale=x_scales.to(x.device),
                weight_scale=weight_scale.to(x.device),
            )
        else:
            # Call into fused qkv ragged matmul.
            xq = fused_qkv_ragged_matmul(
                self.kv_params,
                input=x,
                wqkv=self.wqkv,
                bias=self.wqkv_bias,
                input_row_offsets=kwargs["input_row_offsets"],
                kv_collection=kv_collection,
                layer_idx=layer_idx,
                n_heads=self.n_heads,
            )
        # Apply rope.
        xq = xq.reshape((-1, self.n_heads, self.kv_params.head_dim))

        # Apply QK norm to query and key states.
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
            input_row_offsets=kwargs["input_row_offsets"],
            weight_offset=1.0,
        )

        # Apply rotary embedding.
        use_local = bool((self.layer_idx + 1) % self.sliding_window_pattern)
        rope = self.rope_local if use_local else self.rope_global

        if xq.device is not None:
            freqs_cis = ops.cast(rope.freqs_cis, xq.dtype).to(xq.device)
        else:
            freqs_cis = ops.cast(rope.freqs_cis, xq.dtype)
        xq = fused_qk_ragged_rope(
            self.kv_params,
            xq,
            kwargs["input_row_offsets"],
            kv_collection,
            freqs_cis,
            layer_idx,
            interleaved=rope.interleaved,
        )

        # Calculate Flash Attention.
        mask_variant = (
            MHAMaskVariant.SLIDING_WINDOW_CAUSAL_MASK
            if bool((self.layer_idx + 1) % self.sliding_window_pattern)
            else MHAMaskVariant.CAUSAL_MASK
        )
        attn_out = flash_attention_ragged(
            self.kv_params,
            input=xq,
            kv_collection=kv_collection,
            layer_idx=layer_idx,
            input_row_offsets=kwargs["input_row_offsets"],
            mask_variant=mask_variant,
            scale=self.scale,
            local_window_size=self.local_window_size,
        )
        attn_out = ops.reshape(attn_out, shape=[total_seq_len, -1])
        ret = self.o_proj(attn_out)
        return ret

    @property
    def sharding_strategy(self) -> ShardingStrategy | None:
        return self._sharding_strategy

    @sharding_strategy.setter
    def sharding_strategy(self, sharding_strategy: ShardingStrategy) -> None:
        num_devices = sharding_strategy.num_devices

        if sharding_strategy.is_replicate:
            self.q_norm.sharding_strategy = sharding_strategy
            self.k_norm.sharding_strategy = sharding_strategy
            self.q_proj.sharding_strategy = sharding_strategy
            self.k_proj.sharding_strategy = sharding_strategy
            self.v_proj.sharding_strategy = sharding_strategy
            self.o_proj.sharding_strategy = sharding_strategy

        elif sharding_strategy.is_tensor_parallel:
            self.q_norm.sharding_strategy = ShardingStrategy.replicate(
                num_devices
            )
            self.k_norm.sharding_strategy = ShardingStrategy.replicate(
                num_devices
            )

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

        else:
            raise ValueError(
                "Gemma3Attention only supports tensor parallel and replicate sharding strategy"
            )

        self._sharding_strategy = sharding_strategy

    def shard(self, devices: Iterable[DeviceRef]) -> list[Gemma3Attention]:
        """Creates sharded views of this attention layer across multiple devices.

        Overrides the parent method to handle QK normalization layers.

        Args:
            devices: Iterable of devices to place the shards on.

        Returns:
            List of sharded Gemma3Attention instances, one for each device.
        """
        if not self.sharding_strategy:
            raise ValueError(
                "Gemma3Attention layer cannot be sharded because no sharding strategy was provided."
            )

        # Get sharded weights
        q_proj_shards = self.q_proj.shard(devices)
        k_proj_shards = self.k_proj.shard(devices)
        v_proj_shards = self.v_proj.shard(devices)
        o_proj_shards = self.o_proj.shard(devices)

        # Shard QK normalization weights
        q_norm_weight_shards = self.q_norm.weight.shard(devices)
        k_norm_weight_shards = self.k_norm.weight.shard(devices)

        shards = []
        for shard_idx, device in enumerate(devices):
            # Calculate sharded dimensions - handle uneven head distribution
            sharded_num_heads = compute_heads_per_device(
                total_heads=self.n_heads,
                device_idx=shard_idx,
                num_devices=self.sharding_strategy.num_devices,
            )
            sharded_num_kv_heads = compute_heads_per_device(
                total_heads=self.kv_params.n_kv_heads,
                device_idx=shard_idx,
                num_devices=self.sharding_strategy.num_devices,
            )

            # Create new attention instance with sharded configuration
            sharded = Gemma3Attention(
                rope_global=self.rope_global,
                rope_local=self.rope_local,
                num_attention_heads=sharded_num_heads,
                num_key_value_heads=sharded_num_kv_heads,
                hidden_size=self.q_weight_dim + self.kv_weight_dim * 2,
                kv_params=self.kv_params,
                layer_idx=self.layer_idx,
                sliding_window_pattern=self.sliding_window_pattern,
                dtype=self.q_proj.weight.dtype,
                devices=[device],
                linear_cls=self.o_proj.__class__,
                scale=self.scale,
                has_bias=self.has_bias,
                qk_norm_eps=self.qk_norm_eps,
                local_window_size=self.local_window_size,
                float8_config=self.float8_config,
            )

            # Assign sharded weights
            sharded.q_proj = q_proj_shards[shard_idx]
            sharded.k_proj = k_proj_shards[shard_idx]
            sharded.v_proj = v_proj_shards[shard_idx]
            sharded.o_proj = o_proj_shards[shard_idx]

            # Assign QK normalization weights
            sharded.q_norm.weight = q_norm_weight_shards[shard_idx]
            sharded.k_norm.weight = k_norm_weight_shards[shard_idx]

            shards.append(sharded)

        return shards
