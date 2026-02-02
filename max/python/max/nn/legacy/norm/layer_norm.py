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

"""Layer Normalization layer."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Any

import numpy as np
import numpy.typing as npt
from max.dtype import DType
from max.graph import DeviceRef, ShardingStrategy, TensorValue, Weight, ops

from ..layer import Module, Shardable


class LayerNorm(Module, Shardable):
    """Layer normalization block."""

    def __init__(
        self,
        dims: int,
        devices: Sequence[DeviceRef],
        dtype: DType,
        eps: float = 1e-5,
        use_bias: bool = True,
    ) -> None:
        super().__init__()
        self.devices = devices
        self.weight = Weight("weight", dtype, (dims,), device=self.devices[0])
        self.bias = (
            Weight("bias", dtype, (dims,), device=self.devices[0])
            if use_bias
            else None
        )
        self.eps = eps
        self.dim = dims
        self.dtype = dtype
        self._sharding_strategy: ShardingStrategy | None = None

    def __call__(self, input: TensorValue):
        # TODO: AIPIPE-95 Replace with a broadcasting rmo.layer_norm
        bias = (
            ops.cast(self.bias, DType.float32)
            if self.bias
            # If bias wasn't passed then use bias-less layer norm (beta = 0).
            else ops.broadcast_to(
                ops.constant(0.0, DType.float32, self.weight.device),
                shape=(input.shape[-1],),
            )
        )
        return ops.layer_norm(
            input.cast(DType.float32),
            gamma=ops.cast(self.weight, DType.float32),
            beta=bias,
            epsilon=self.eps,
        ).cast(input.dtype)

    @property
    def sharding_strategy(self) -> ShardingStrategy | None:
        """Get the LayerNorm sharding strategy."""
        return self._sharding_strategy

    @sharding_strategy.setter
    def sharding_strategy(self, strategy: ShardingStrategy) -> None:
        """Set the sharding strategy for the LayerNorm layer.

        Args:
            strategy: The sharding strategy to apply.
        """
        # LayerNorm currently only supports replicate strategy
        if not strategy.is_replicate:
            raise ValueError("LayerNorm only supports replicate strategy")

        self._sharding_strategy = strategy
        self.weight.sharding_strategy = strategy
        if self.bias is not None:
            self.bias.sharding_strategy = strategy

    def shard(self, devices: Iterable[DeviceRef]) -> Sequence[LayerNorm]:
        """Creates sharded views of this LayerNorm across multiple devices.

        Args:
            devices: Iterable of devices to place the shards on.

        Returns:
            List of sharded LayerNorm instances, one for each device.
        """
        if self.sharding_strategy is None:
            raise ValueError("Sharding strategy is not set")

        # Get sharded weights
        weight_shards = list(self.weight.shard(devices))
        if self.bias is not None:
            bias_shards: list[Weight | None] = list(self.bias.shard(devices))
        else:
            # Build a list of Nones matching number of devices
            bias_shards = [None for _ in range(len(weight_shards))]

        shards: list[LayerNorm] = []
        for weight_shard, bias_shard, device in zip(
            weight_shards, bias_shards, devices, strict=True
        ):
            # Create new LayerNorm instance with the same configuration
            sharded = LayerNorm(
                dims=self.dim,
                devices=[device],
                dtype=self.dtype,
                eps=self.eps,
                use_bias=self.bias is not None,
            )

            # Assign the sharded parameters
            sharded.weight = weight_shard
            sharded.bias = bias_shard

            shards.append(sharded)

        return shards


class ConstantLayerNorm(Module):
    """Layer normalization block with constant gamma and beta values."""

    gamma: npt.NDArray[np.floating[Any]]
    beta: npt.NDArray[np.floating[Any]]
    eps: float = 1e-5
    device: DeviceRef
    dtype: DType

    def __init__(
        self,
        dims: int | tuple[int, ...],
        device: DeviceRef,
        dtype: DType,
        eps: float = 1e-5,
    ) -> None:
        super().__init__()
        self.gamma = np.ones(dims, dtype=dtype.to_numpy())
        self.beta = np.zeros(dims, dtype=dtype.to_numpy())
        self.eps = eps
        self.device = device
        self.dtype = dtype

    def __call__(self, input: TensorValue) -> TensorValue:
        gamma = ops.constant(self.gamma, self.dtype, self.device)
        beta = ops.constant(self.beta, self.dtype, self.device)
        return ops.cast(
            ops.layer_norm(
                ops.cast(input, DType.float32),
                gamma=gamma,
                beta=beta,
                epsilon=self.eps,
            ),
            input.dtype,
        )
