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

from collections.abc import Callable, Iterable

from max.dtype import DType
from max.graph import (
    DeviceRef,
    ShardingStrategy,
    TensorValue,
    TensorValueLike,
)
from max.graph.quantization import QuantizationEncoding
from max.nn.activation import activation_function_from_name
from max.nn.float8_config import Float8Config
from max.nn.layer import Module, Shardable
from max.nn.linear import Linear


# TODO: (MODELS-1084) generalize this (non-gated) MLP layer and move it somewhere central
class MLP2(Module, Shardable):
    """Simple multi-layer perceptron composed of two :obj:`Linear` layers."""

    def __init__(
        self,
        dim: tuple[int, int, int],
        dtype: DType,
        device: DeviceRef,
        quantization_encoding: QuantizationEncoding | None = None,
        linear_cls: Callable[..., Linear] = Linear,
        has_bias: bool = False,
        activation_function: str = "gelu_tanh",
        float8_config: Float8Config | None = None,
        _is_sharding: bool = False,
    ) -> None:
        """Initializes the MLP2 layer.

        Args:
            dim: (in_dim, hidden_dim, out_dim) of the MLP2.
            dtype: :obj:`DType` to use for the linear layers, which must match the input dtype.
            device: :obj:`DeviceRef` device to run the ``MLP2`` layer on.
            quantization_encoding: :obj:`QuantizationEncoding` of the layer weights.
            linear_cls: :obj:`Linear` class to use to create the projection layers.
            has_bias: Whether to include bias terms in the linear layers.
            activation_function: Activation function to use. Options are:

                - ``gelu_tanh``

            float8_config: :obj:`Float8Config` for float8 quantization.
            _is_sharding: Used internally to disable child layer creation during sharding.
        """
        super().__init__()

        if not _is_sharding:
            common = dict(
                dtype=dtype,
                device=device,
                quantization_encoding=quantization_encoding,
                has_bias=has_bias,
                float8_config=float8_config,
            )
            self.down_proj = linear_cls(in_dim=dim[0], out_dim=dim[1], **common)
            self.up_proj = linear_cls(in_dim=dim[1], out_dim=dim[2], **common)

        self.dim = dim
        self.dtype = dtype
        self.quantization_encoding = quantization_encoding
        self.has_bias = has_bias
        self._activation_function_name = activation_function
        self.activation_function = activation_function_from_name(
            activation_function
        )
        self.float8_config = float8_config
        self._sharding_strategy: ShardingStrategy | None = None

    def __call__(self, x: TensorValueLike) -> TensorValue:
        """Applies the MLP2 transformation to the input.

        Args:
            x: Input tensor to transform.

        Returns:
            The transformed tensor after applying the MLP2 layers.
        """
        value = TensorValue(x)
        assert value.dtype == self.dtype, (
            f"Input dtype does not match {self.__class__.__name__} dtype (received {value.dtype}, expected {self.dtype})"
        )
        value = self.up_proj(value)
        value = self.activation_function(value)
        return self.down_proj(value)

    @property
    def sharding_strategy(self) -> ShardingStrategy | None:
        """Get the MLP2 sharding strategy."""
        return self._sharding_strategy

    @sharding_strategy.setter
    def sharding_strategy(self, strategy: ShardingStrategy) -> None:
        """Set the sharding strategy for the MLP2 layers.

        Args:
            strategy: The sharding strategy to apply.
        """
        self._sharding_strategy = strategy

        if strategy.is_replicate:
            self.down_proj.sharding_strategy = strategy
            self.up_proj.sharding_strategy = strategy
        elif strategy.is_tensor_parallel:
            self.down_proj.sharding_strategy = ShardingStrategy.columnwise(
                strategy.num_devices
            )
            self.up_proj.sharding_strategy = ShardingStrategy.rowwise(
                strategy.num_devices
            )
        else:
            raise ValueError(f"Unsupported sharding strategy ({strategy})")

    def shard(self, devices: Iterable[DeviceRef]) -> list[MLP2]:
        """Creates sharded views of this MLP2 across multiple devices.

        Args:
            devices: Iterable of devices to place the shards on.

        Returns:
            List of sharded MLP2 instances, one for each device.
        """
        if self._sharding_strategy is None:
            raise ValueError(
                "A sharding strategy must be set prior to calling this method"
            )

        sharded_down_projs = self.down_proj.shard(devices)
        sharded_up_projs = self.up_proj.shard(devices)

        shards = []
        for device, down_proj, up_proj in zip(
            devices,
            sharded_down_projs,
            sharded_up_projs,
            strict=True,
        ):
            sharded = MLP2(
                dim=self.dim,
                dtype=self.dtype,
                device=device,
                quantization_encoding=self.quantization_encoding,
                has_bias=self.has_bias,
                activation_function=self._activation_function_name,
                float8_config=self.float8_config,
                _is_sharding=True,
            )
            sharded.down_proj = down_proj
            sharded.up_proj = up_proj
            shards.append(sharded)

        return shards
