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

"""GptOss Attention Layer."""

from __future__ import annotations

import math
from collections.abc import Iterable

from max.dtype import DType
from max.graph import DeviceRef, ShardingStrategy, TensorValue, Weight, ops
from max.nn.attention import MHAMaskVariant
from max.nn.kernels import (
    flash_attention_ragged,
    fused_qk_ragged_rope,
    fused_qkv_ragged_matmul,
)
from max.nn.kv_cache import (
    KVCacheParams,
    PagedKVCacheCollection,
)
from max.nn.layer import Module, Shardable
from max.nn.linear import Linear
from max.nn.rotary_embedding import YarnRotaryEmbedding


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


class GptOssAttention(Module, Shardable):
    """Implementation of the distributed attention layer for the GptOss text model.

    Depending on the layer type, the attention layer can be either a full attention
    layer or a sliding window attention layer. This layer generates the attention mask
    based on the layer type.

    This layer also supports sink attention, which is a technique to improve the
    attention mechanism by adding an extra logit column that acts as an attention
    sink.
    """

    def __init__(
        self,
        *,
        rope: YarnRotaryEmbedding,
        num_attention_heads: int,
        num_key_value_heads: int,
        hidden_size: int,
        kv_params: KVCacheParams,
        layer_idx: int,
        layer_type: str = "full_attention",
        dtype: DType = DType.float32,
        devices: list[DeviceRef],
        scale: float | None = None,
        has_bias: bool = False,
        local_window_size: int = 1024,
    ) -> None:
        """Initializes the attention layer.

        Args:
            rope: Rotary embedding used for all attention layers (full + sliding window).
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
        """

        super().__init__()
        self.rope = rope
        self.n_heads = num_attention_heads
        self.hidden_size = hidden_size
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
        self.layer_type = layer_type

        # Initialize sinks parameter for each attention head
        self.sinks = Weight(
            name="sinks",
            dtype=dtype,
            shape=[num_attention_heads],
            device=devices[0],
        )

        if not self.kv_params.cache_strategy.uses_opaque():
            raise ValueError(
                f"{self.kv_params.cache_strategy} cache strategy, not supported"
                " in Attention layer."
            )

        self.q_weight_dim = self.kv_params.head_dim * num_attention_heads
        self.kv_weight_dim = self.kv_params.head_dim * num_key_value_heads

        self.q_proj = Linear(
            in_dim=hidden_size,
            out_dim=self.q_weight_dim,
            dtype=dtype,
            device=devices[0],
            has_bias=self.has_bias,
        )
        self.k_proj = Linear(
            in_dim=hidden_size,
            out_dim=self.kv_weight_dim,
            dtype=dtype,
            device=devices[0],
            has_bias=self.has_bias,
        )
        self.v_proj = Linear(
            in_dim=hidden_size,
            out_dim=self.kv_weight_dim,
            dtype=dtype,
            device=devices[0],
            has_bias=self.has_bias,
        )

        self.o_proj = Linear(
            in_dim=self.q_weight_dim,
            out_dim=hidden_size,
            dtype=dtype,
            device=devices[0],
            has_bias=self.has_bias,
        )

    @property
    def wqkv(self) -> TensorValue:
        """The concatenation of q, k, and v weight vectors."""
        wq: TensorValue = self.q_proj.weight
        wk: TensorValue = self.k_proj.weight
        wv: TensorValue = self.v_proj.weight
        return ops.concat([wq, wk, wv], axis=0)

    @property
    def wqkv_bias(self) -> TensorValue | None:
        """The concatenation of q, k, and v bias weight vectors."""
        if not self.has_bias:
            return None

        if (
            self.q_proj.bias is None
            or self.k_proj.bias is None
            or self.v_proj.bias is None
        ):
            raise ValueError(
                "Projection bias is None, but has_bias=True was specified."
            )

        return ops.concat(
            [self.q_proj.bias, self.k_proj.bias, self.v_proj.bias], axis=0
        )

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
        # Call into fused qkv ragged matmul.
        wqkv = self.wqkv
        xq = fused_qkv_ragged_matmul(
            self.kv_params,
            input=x,
            wqkv=wqkv,
            bias=self.wqkv_bias,
            input_row_offsets=kwargs["input_row_offsets"],
            kv_collection=kv_collection,
            layer_idx=layer_idx,
            n_heads=self.n_heads,
        )
        # Apply rope.
        xq = xq.reshape((-1, self.n_heads, self.kv_params.head_dim))

        # Apply rotary embedding based on layer type
        rope = self.rope

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

        # Calculate Flash Attention with sinks.
        mask_variant = (
            MHAMaskVariant.SLIDING_WINDOW_CAUSAL_MASK
            if self.layer_type == "sliding_attention"
            else MHAMaskVariant.CAUSAL_MASK
        )
        # The sinks parameter modifies the attention computation by adding an extra
        # logit column that acts as an attention sink.
        attn_out = flash_attention_ragged(
            self.kv_params,
            input=xq,
            kv_collection=kv_collection,
            layer_idx=layer_idx,
            input_row_offsets=kwargs["input_row_offsets"],
            mask_variant=mask_variant,
            scale=self.scale,
            local_window_size=self.local_window_size,
            sink_weights=self.sinks,
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
            self.q_proj.sharding_strategy = sharding_strategy
            self.k_proj.sharding_strategy = sharding_strategy
            self.v_proj.sharding_strategy = sharding_strategy
            self.o_proj.sharding_strategy = sharding_strategy

        elif sharding_strategy.is_tensor_parallel:
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
            self.sinks.sharding_strategy = ShardingStrategy.rowwise(num_devices)
        else:
            raise ValueError(
                "GptOssAttention only supports tensor parallel and replicate sharding strategy"
            )

        self._sharding_strategy = sharding_strategy

    def shard(self, devices: Iterable[DeviceRef]) -> list[GptOssAttention]:
        """Creates sharded views of this attention layer across multiple devices.

        Overrides the parent method to handle QK normalization layers.

        Args:
            devices: Iterable of devices to place the shards on.

        Returns:
            List of sharded GptOssAttention instances, one for each device.
        """
        if not self.sharding_strategy:
            raise ValueError(
                "GptOssAttention layer cannot be sharded because no sharding strategy was provided."
            )

        # Get sharded weights
        q_proj_shards = self.q_proj.shard(devices)
        k_proj_shards = self.k_proj.shard(devices)
        v_proj_shards = self.v_proj.shard(devices)
        o_proj_shards = self.o_proj.shard(devices)

        # Shard sinks parameter
        sinks_shards = self.sinks.shard(devices)

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
            sharded = GptOssAttention(
                rope=self.rope,
                num_attention_heads=sharded_num_heads,
                num_key_value_heads=sharded_num_kv_heads,
                hidden_size=self.hidden_size,
                kv_params=self.kv_params,
                layer_idx=self.layer_idx,
                layer_type=self.layer_type,
                dtype=self.q_proj.weight.dtype,
                devices=[device],
                scale=self.scale,
                has_bias=self.has_bias,
                local_window_size=self.local_window_size,
            )

            # Assign sharded weights
            sharded.q_proj = q_proj_shards[shard_idx]
            sharded.k_proj = k_proj_shards[shard_idx]
            sharded.v_proj = v_proj_shards[shard_idx]
            sharded.o_proj = o_proj_shards[shard_idx]

            # Assign sinks parameter
            sharded.sinks = sinks_shards[shard_idx]

            shards.append(sharded)

        return shards
