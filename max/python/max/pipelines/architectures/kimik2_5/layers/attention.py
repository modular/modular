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

"""Attention layer for Kimi K2.5 vision tower."""

from __future__ import annotations

import math
from collections.abc import Iterable

from max.dtype import DType
from max.graph import (
    DeviceRef,
    ShardingStrategy,
    TensorValue,
    TensorValueLike,
    ops,
)
from max.nn.attention.mask_config import MHAMaskVariant
from max.nn.kernels import flash_attention_ragged_gpu
from max.nn.layer import Module, Shardable
from max.nn.linear import Linear


def _compute_heads_per_device(
    *, total_heads: int, device_idx: int, num_devices: int
) -> int:
    """Computes the number of heads assigned to a specific device.

    Distributes heads across devices, handling cases where the total is not
    evenly divisible by the number of devices.

    Args:
        total_heads: Total number of attention heads.
        device_idx: The index of the current device (0-based).
        num_devices: Total number of devices.

    Returns:
        Number of heads assigned to the specified device.
    """
    base_heads, remainder = divmod(total_heads, num_devices)
    if device_idx < remainder:
        return base_heads + 1
    return base_heads


class Attention(Module, Shardable):
    """QKV-packed self-attention for the Kimi K2.5 vision encoder.

    Args:
        num_heads: Number of attention heads.
        hidden_dim: Hidden dimension.
        dtype: Data type for all layer weights.
        device: Device on which to allocate weights.
        has_bias: Whether linear projections include bias terms.
        head_dim: Dimension per attention head. Defaults to
            ``hidden_dim // num_heads``.
    """

    def __init__(
        self,
        num_heads: int,
        hidden_dim: int,
        dtype: DType,
        device: DeviceRef,
        has_bias: bool = False,
        head_dim: int | None = None,
        _is_sharding: bool = False,
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.hidden_dim = hidden_dim
        self.head_dim = (
            head_dim if head_dim is not None else hidden_dim // num_heads
        )
        self.scale = 1.0 / math.sqrt(self.head_dim)
        self.dtype = dtype
        self.has_bias = has_bias
        self._sharding_strategy: ShardingStrategy | None = None

        # During sharding, sub-layers are assigned directly from the
        # parent (see shard()), so skip creation here.
        if not _is_sharding:
            self.wqkv = Linear(
                in_dim=hidden_dim,
                out_dim=hidden_dim * 3,
                dtype=dtype,
                device=device,
                has_bias=has_bias,
            )
            self.wo = Linear(
                in_dim=hidden_dim,
                out_dim=hidden_dim,
                dtype=dtype,
                device=device,
                has_bias=has_bias,
            )

    @staticmethod
    def _apply_rope(x: TensorValue, freqs_cis: TensorValue) -> TensorValue:
        """Applies 2-D rotary position embedding to a query or key tensor.

        Args:
            x: Tensor of shape (tot_seqlens, num_heads, head_dim).
            freqs_cis: Precomputed [cos, sin] pairs of shape
                (tot_seqlens, head_dim // 2, 2).

        Returns:
            Tensor with the same shape as *x* after RoPE rotation.
        """
        orig_dtype = x.dtype
        x = ops.cast(x, DType.float32)

        complex_x = ops.as_interleaved_complex(x)
        x_re = complex_x[..., 0]
        x_im = complex_x[..., 1]

        freqs_cis = ops.cast(ops.unsqueeze(freqs_cis, 1), DType.float32)
        freqs_re = freqs_cis[..., 0]
        freqs_im = freqs_cis[..., 1]

        rope_re = x_re * freqs_re - x_im * freqs_im
        rope_im = x_re * freqs_im + x_im * freqs_re

        result = ops.stack([rope_re, rope_im], axis=-1)
        result = ops.reshape(result, x.shape)
        return ops.cast(result, orig_dtype)

    def __call__(
        self,
        x: TensorValueLike,
        input_row_offsets: TensorValue,
        max_seq_len: TensorValue,
        rope_freqs_cis: TensorValue,
    ) -> TensorValue:
        """Compute self-attention with packed QKV projection.

        Args:
            x: Input tensor of shape (n_patches, hidden_dim).
            input_row_offsets: Cumulative sequence lengths of shape
                (batch_size + 1,), dtype uint32.
            max_seq_len: Maximum sequence length, shape (1,), dtype uint32.
            rope_freqs_cis: Precomputed [cos, sin] pairs of shape
                (n_patches, head_dim // 2, 2).

        Returns:
            Output tensor of shape (n_patches, hidden_dim).
        """
        x = TensorValue(x)
        n_patches = x.shape[0]

        xqkv = self.wqkv(x)

        # Reshape to [n_patches, 3, num_heads, head_dim] and split Q, K, V
        xqkv = ops.reshape(xqkv, (n_patches, 3, self.num_heads, self.head_dim))
        xq, xk, xv = ops.split(xqkv, [1, 1, 1], axis=1)
        xq = ops.squeeze(xq, 1)
        xk = ops.squeeze(xk, 1)
        xv = ops.squeeze(xv, 1)

        xq = self._apply_rope(xq, rope_freqs_cis)
        xk = self._apply_rope(xk, rope_freqs_cis)

        attn_out = flash_attention_ragged_gpu(
            xq,
            xk,
            xv,
            input_row_offsets=input_row_offsets,
            max_seq_len=max_seq_len,
            mask_variant=MHAMaskVariant.NULL_MASK,
            scale=self.scale,
        )

        attn_out = ops.reshape(attn_out, (n_patches, -1))

        return self.wo(attn_out)

    @property
    def sharding_strategy(self) -> ShardingStrategy | None:
        """Gets the attention sharding strategy."""
        return self._sharding_strategy

    @sharding_strategy.setter
    def sharding_strategy(self, strategy: ShardingStrategy) -> None:
        """Sets the sharding strategy for attention weights.

        Args:
            strategy: The sharding strategy to apply.
        """
        if strategy.is_replicate:
            self.wqkv.sharding_strategy = strategy
            self.wo.sharding_strategy = strategy
        elif strategy.is_tensor_parallel:
            self.wqkv.sharding_strategy = ShardingStrategy.stacked_qkv(
                strategy.num_devices, self.num_heads, self.head_dim
            )
            self.wo.sharding_strategy = ShardingStrategy.head_aware_columnwise(
                strategy.num_devices, self.num_heads, self.head_dim
            )
        else:
            raise ValueError(
                f"{self.__class__.__name__} only supports tensor parallel and"
                f" replicate sharding strategies"
            )

        self._sharding_strategy = strategy

    def shard(self, devices: Iterable[DeviceRef]) -> list[Attention]:
        """Creates sharded views of this attention layer across multiple devices.

        Args:
            devices: Iterable of devices to place the shards on.

        Returns:
            List of sharded :obj:`Attention` instances, one per device.
        """
        if not self._sharding_strategy:
            raise ValueError(
                "A sharding strategy must be set prior to calling this method"
            )

        wqkv_shards = self.wqkv.shard(devices)
        wo_shards = self.wo.shard(devices)

        shards = []
        for shard_idx, device in enumerate(devices):
            sharded_num_heads = _compute_heads_per_device(
                total_heads=self.num_heads,
                device_idx=shard_idx,
                num_devices=self._sharding_strategy.num_devices,
            )

            sharded = Attention(
                num_heads=sharded_num_heads,
                hidden_dim=self.hidden_dim,
                dtype=self.dtype,
                device=device,
                has_bias=self.has_bias,
                head_dim=self.head_dim,
                _is_sharding=True,
            )
            sharded.wqkv = wqkv_shards[shard_idx]
            sharded.wo = wo_shards[shard_idx]
            shards.append(sharded)

        return shards
