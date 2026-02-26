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

"""Encoder layer for Kimi K2.5 vision tower."""

from __future__ import annotations

from collections.abc import Iterable

from max.dtype import DType
from max.graph import DeviceRef, ShardingStrategy, TensorValue, TensorValueLike
from max.nn.layer import Module, Shardable
from max.nn.norm import LayerNorm

from .attention import Attention
from .mlp import MLP2


class EncoderLayer(Module, Shardable):
    """Vision encoder layer with QKV-packed self-attention and MLP.

    Args:
        num_heads: Number of attention heads.
        hidden_dim: Hidden dimension of the encoder.
        mlp_dim: Inner dimension of the feed-forward MLP.
        dtype: Data type for all layer weights.
        device: Device on which to allocate weights.
        has_bias: Whether linear projections include bias terms.
    """

    def __init__(
        self,
        num_heads: int,
        hidden_dim: int,
        mlp_dim: int,
        dtype: DType,
        device: DeviceRef,
        has_bias: bool = False,
        _is_sharding: bool = False,
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.hidden_dim = hidden_dim
        self.dtype = dtype
        self.mlp_dim = mlp_dim
        self.has_bias = has_bias
        self._sharding_strategy: ShardingStrategy | None = None

        # During sharding, sub-layers are assigned directly from the
        # parent (see shard()), so skip creation here.
        if not _is_sharding:
            self.norm0 = LayerNorm(
                dims=hidden_dim, devices=[device], dtype=dtype
            )
            self.norm1 = LayerNorm(
                dims=hidden_dim, devices=[device], dtype=dtype
            )
            self.attn = Attention(
                num_heads=num_heads,
                hidden_dim=hidden_dim,
                dtype=dtype,
                device=device,
                has_bias=has_bias,
            )
            self.mlp = MLP2(
                dim=(hidden_dim, mlp_dim, hidden_dim),
                dtype=dtype,
                device=device,
                has_bias=has_bias,
            )

    def __call__(
        self,
        x: TensorValueLike,
        input_row_offsets: TensorValue,
        max_seq_len: TensorValue,
        rope_freqs_cis: TensorValue,
    ) -> TensorValue:
        """Full encoder forward pass.

        Args:
            x: Packed input tensor of shape (n_patches, hidden_dim).
            input_row_offsets: Cumulative sequence lengths of shape
                (batch_size + 1,), dtype uint32.
            max_seq_len: Maximum sequence length, shape (1,), dtype uint32.
            rope_freqs_cis: Precomputed [cos, sin] pairs of shape
                (n_patches, head_dim // 2, 2).

        Returns:
            Output tensor of shape (n_patches, hidden_dim).
        """
        x = TensorValue(x)
        residual = x
        x = self.norm0(x)
        x = self.attn(x, input_row_offsets, max_seq_len, rope_freqs_cis)
        x = residual + x

        residual = x
        x = self.norm1(x)
        x = self.mlp(x)
        x = residual + x

        return x

    @property
    def sharding_strategy(self) -> ShardingStrategy | None:
        """Gets the encoder layer sharding strategy."""
        return self._sharding_strategy

    @sharding_strategy.setter
    def sharding_strategy(self, strategy: ShardingStrategy) -> None:
        """Sets the sharding strategy for all sub-layers.

        Args:
            strategy: The sharding strategy to apply.
        """
        if strategy.is_replicate:
            self.norm0.sharding_strategy = strategy
            self.norm1.sharding_strategy = strategy
            self.attn.sharding_strategy = strategy
            self.mlp.sharding_strategy = strategy
        elif strategy.is_tensor_parallel:
            self.norm0.sharding_strategy = ShardingStrategy.replicate(
                strategy.num_devices
            )
            self.norm1.sharding_strategy = ShardingStrategy.replicate(
                strategy.num_devices
            )
            self.attn.sharding_strategy = strategy
            self.mlp.sharding_strategy = ShardingStrategy.tensor_parallel(
                strategy.num_devices
            )
        else:
            raise ValueError(
                f"{self.__class__.__name__} only supports tensor parallel and"
                f" replicate sharding strategies"
            )

        self._sharding_strategy = strategy

    def shard(self, devices: Iterable[DeviceRef]) -> list[EncoderLayer]:
        """Creates sharded views of this encoder layer across multiple devices.

        Args:
            devices: Iterable of devices to place the shards on.

        Returns:
            List of sharded :obj:`EncoderLayer` instances, one per device.
        """
        if not self._sharding_strategy:
            raise ValueError(
                "A sharding strategy must be set prior to calling this method"
            )

        norm0_shards = self.norm0.shard(devices)
        norm1_shards = self.norm1.shard(devices)
        attn_shards = self.attn.shard(devices)
        mlp_shards = self.mlp.shard(devices)

        shards = []
        for shard_idx, device in enumerate(devices):
            sharded = EncoderLayer(
                num_heads=self.num_heads,
                hidden_dim=self.hidden_dim,
                mlp_dim=self.mlp_dim,
                dtype=self.dtype,
                device=device,
                has_bias=self.has_bias,
                _is_sharding=True,
            )
            sharded.norm0 = norm0_shards[shard_idx]
            sharded.norm1 = norm1_shards[shard_idx]
            sharded.attn = attn_shards[shard_idx]
            sharded.mlp = mlp_shards[shard_idx]
            shards.append(sharded)

        return shards
