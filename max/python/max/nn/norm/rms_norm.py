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

"""Normalization layer."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass

from max.dtype import DType
from max.graph import (
    DeviceRef,
    ShardingStrategy,
    TensorType,
    TensorValue,
    TensorValueLike,
    Weight,
    ops,
)

from ..layer import Layer, Module, Shardable


@dataclass
class RMSNormV1(Layer):
    """Computes the Root Mean Square normalization on inputs.

    Deprecated: Use `RMSNorm` instead.
    """

    weight: TensorValueLike
    eps: float = 1e-6
    weight_offset: float = 0.0
    multiply_before_cast: bool = True

    def __call__(self, x: TensorValue) -> TensorValue:
        return ops.custom(
            "rms_norm",
            x.device,
            [
                x,
                TensorValue(self.weight).cast(x.dtype),
                ops.constant(self.eps, dtype=x.dtype, device=DeviceRef.CPU()),
                ops.constant(
                    self.weight_offset, dtype=x.dtype, device=DeviceRef.CPU()
                ),
            ],
            [TensorType(dtype=x.dtype, shape=x.shape, device=x.device)],
            parameters={"multiply_before_cast": self.multiply_before_cast},
        )[0].tensor


class RMSNorm(Module, Shardable):
    """Computes the Root Mean Square normalization on inputs.

    Args:
        dim: Size of last dimension of the expected input.
        eps: Value added to denominator for numerical stability.
        weight_offset: Constant offset added to the learned weights at runtime.
            For Gemma-style RMSNorm, this should be set to 1.0.
        multiply_before_cast: True if we multiply the inputs by the learned
            weights before casting to the input type (Gemma3-style). False if we
            cast the inputs to the input type first, then multiply by the learned
            weights (Llama-style).
    """

    def __init__(
        self,
        dim: int,
        dtype: DType,
        eps: float = 1e-6,
        weight_offset: float = 0.0,
        multiply_before_cast: bool = True,
    ) -> None:
        super().__init__()
        self.weight = Weight("weight", dtype, [dim], device=DeviceRef.CPU())
        self.dim = dim
        self.dtype = dtype
        self.eps = eps
        self.weight_offset = weight_offset
        self.multiply_before_cast = multiply_before_cast
        self._sharding_strategy: ShardingStrategy | None = None

    def __call__(self, x: TensorValue) -> TensorValue:
        # Validate that weight dimension matches input's last dimension if
        # statically known.
        input_last_dim = x.shape[-1]
        weight_dim = self.weight.shape[0]

        if input_last_dim != weight_dim:
            raise ValueError(
                f"RMSNorm weight dimension ({weight_dim}) must match the input's "
                f"last dimension ({input_last_dim})"
            )

        weight: TensorValue = self.weight.cast(x.dtype)
        if x.device:
            weight = weight.to(x.device)

        return ops.custom(
            "rms_norm",
            x.device,
            [
                x,
                weight,
                ops.constant(self.eps, dtype=x.dtype, device=DeviceRef.CPU()),
                ops.constant(
                    self.weight_offset, dtype=x.dtype, device=DeviceRef.CPU()
                ),
            ],
            [TensorType(dtype=x.dtype, shape=x.shape, device=x.device)],
            parameters={"multiply_before_cast": self.multiply_before_cast},
        )[0].tensor

    @property
    def sharding_strategy(self) -> ShardingStrategy | None:
        """Get the RMSNorm sharding strategy."""
        return self._sharding_strategy

    @sharding_strategy.setter
    def sharding_strategy(self, strategy: ShardingStrategy) -> None:
        """Set the sharding strategy for the RMSNorm layer.

        Args:
            strategy: The sharding strategy to apply.
        """
        # RMSNorm always uses replicate strategy
        if not strategy.is_replicate:
            raise ValueError("RMSNorm only supports replicate strategy")

        self._sharding_strategy = strategy
        self.weight.sharding_strategy = strategy

    def shard(self, devices: Iterable[DeviceRef]) -> Sequence[RMSNorm]:
        """Creates sharded views of this RMSNorm across multiple devices.

        Args:
            devices: Iterable of devices to place the shards on.

        Returns:
            List of sharded RMSNorm instances, one for each device.
        """
        if self.sharding_strategy is None:
            raise ValueError("Sharding strategy is not set")

        # Get sharded weights
        weight_shards = self.weight.shard(devices)

        shards = []
        for weight_shard in weight_shards:
            # Create new RMSNorm instance with the same configuration
            sharded = RMSNorm(
                dim=self.dim,
                dtype=self.dtype,
                eps=self.eps,
                weight_offset=self.weight_offset,
                multiply_before_cast=self.multiply_before_cast,
            )

            # Assign the sharded weight
            sharded.weight = weight_shard

            shards.append(sharded)

        return shards
