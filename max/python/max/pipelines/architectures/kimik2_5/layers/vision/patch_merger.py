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

from __future__ import annotations

from collections.abc import Iterable

from max.dtype import DType
from max.graph import DeviceRef, ShardingStrategy, TensorValue, ops
from max.nn.float8_config import Float8Config
from max.nn.layer import Module, Shardable
from max.nn.linear import Linear
from max.nn.norm import LayerNorm


class PatchMergerMLP(Module, Shardable):
    """Two-layer MLP with LayerNorm that merges spatially adjacent patches.

    Expects ragged input of shape ``(total_patches, N_k, mm_hidden_size)``.
    Applies layer normalization, reshapes to merge adjacent patches,
    then projects through a two-layer MLP with GELU activation.
    """

    def __init__(
        self,
        dtype: DType,
        device: DeviceRef,
        mm_hidden_size: int,
        hidden_size: int,
        merge_kernel_size: tuple[int, int],
        eps: float = 1e-5,
        float8_config: Float8Config | None = None,
        _is_sharding: bool = False,
    ) -> None:
        super().__init__()
        self.dtype = dtype
        self.device = device
        self.mm_hidden_size = mm_hidden_size
        self.hidden_size = hidden_size
        self.merge_kernel_size = merge_kernel_size
        self.eps = eps
        self.float8_config = float8_config

        self.input_dim = mm_hidden_size * (
            merge_kernel_size[0] * merge_kernel_size[1]
        )

        if not _is_sharding:
            self.pre_norm = LayerNorm(
                dims=mm_hidden_size,
                devices=[device],
                dtype=dtype,
                eps=eps,
            )

            self.linear1 = Linear(
                in_dim=self.input_dim,
                out_dim=self.input_dim,
                dtype=dtype,
                device=device,
                has_bias=True,
                float8_config=float8_config,
            )

            self.linear2 = Linear(
                in_dim=self.input_dim,
                out_dim=hidden_size,
                dtype=dtype,
                device=device,
                has_bias=True,
                float8_config=float8_config,
            )

        self._sharding_strategy: ShardingStrategy | None = None

    def __call__(self, x: TensorValue) -> TensorValue:
        # x: (total_patches, N_k, mm_hidden_size)
        x = self.pre_norm(x)
        x = x.reshape((x.shape[0], -1))  # (total_patches, input_dim)
        x = self.linear1(x)
        x = ops.gelu(x)
        x = self.linear2(x)
        return x  # (total_patches, hidden_size)

    @property
    def sharding_strategy(self) -> ShardingStrategy | None:
        return self._sharding_strategy

    @sharding_strategy.setter
    def sharding_strategy(self, strategy: ShardingStrategy) -> None:
        self._sharding_strategy = strategy
        self.pre_norm.sharding_strategy = ShardingStrategy.replicate(
            strategy.num_devices
        )
        if strategy.is_replicate:
            self.linear1.sharding_strategy = strategy
            self.linear2.sharding_strategy = strategy
        elif strategy.is_tensor_parallel:
            self.linear1.sharding_strategy = ShardingStrategy.rowwise(
                strategy.num_devices
            )
            self.linear2.sharding_strategy = ShardingStrategy.columnwise(
                strategy.num_devices
            )
        else:
            raise ValueError(f"Unsupported sharding strategy ({strategy})")

    def shard(self, devices: Iterable[DeviceRef]) -> list[PatchMergerMLP]:
        if self._sharding_strategy is None:
            raise ValueError(
                "A sharding strategy must be set prior to calling this method"
            )

        norm_shards = self.pre_norm.shard(devices)
        linear1_shards = self.linear1.shard(devices)
        linear2_shards = self.linear2.shard(devices)

        shards: list[PatchMergerMLP] = []
        for device, norm, l1, l2 in zip(
            devices,
            norm_shards,
            linear1_shards,
            linear2_shards,
            strict=True,
        ):
            sharded = PatchMergerMLP(
                dtype=self.dtype,
                device=device,
                mm_hidden_size=self.mm_hidden_size,
                hidden_size=self.hidden_size,
                merge_kernel_size=self.merge_kernel_size,
                eps=self.eps,
                float8_config=self.float8_config,
                _is_sharding=True,
            )
            sharded.pre_norm = norm
            sharded.linear1 = l1
            sharded.linear2 = l2
            shards.append(sharded)

        return shards
