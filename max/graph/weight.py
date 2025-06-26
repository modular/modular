# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
from typing import Callable, Optional, Union

import numpy.typing as npt
from max._core import Value as _Value
from max._core.dialects import mo
from max._core_types.driver import DLPackArray
from max.dtype import DType

from . import graph
from .quantization import QuantizationEncoding
from .type import DeviceRef, Shape, ShapeLike
from .value import TensorValue, Value

DLPackCompatible = Union[DLPackArray, npt.NDArray]


def _compute_shard_range(
    shard_dim: int, shard_idx: int, num_devices: int
) -> tuple[int, int]:
    base_size, remainder = divmod(shard_dim, num_devices)

    # Give first 'remainder' devices an extra column/row.
    if shard_idx < remainder:
        start = shard_idx * (base_size + 1)
        end = start + base_size + 1
    else:
        start = (
            remainder * (base_size + 1) + (shard_idx - remainder) * base_size
        )
        end = start + base_size

    return start, end


def col_sharding_strategy(
    weight: Weight, i: int, num_devices: int
) -> TensorValue:
    """Shards a weight tensor by column for a given device.

    Args:
        weight: The :obj:`Weight` to shard.
        i: The index of the current device.
        num_devices: The total number of devices to shard across.

    Returns:
        A :obj:`TensorValue` representing the sharded portion of the weight
        for the i-th device.
    """
    start, end = _compute_shard_range(
        shard_dim=int(weight.shape[1]), shard_idx=i, num_devices=num_devices
    )

    return weight[:, start:end]


def row_sharding_strategy(
    weight: Weight, i: int, num_devices: int
) -> TensorValue:
    """Shards a weight tensor by row for a given device.

    Args:
        weight: The :obj:`Weight` to shard.
        i: The index of the current device.
        num_devices: The total number of devices to shard across.

    Returns:
        A :obj:`TensorValue` representing the sharded portion of the weight
        for the i-th device.
    """
    start, end = _compute_shard_range(
        shard_dim=int(weight.shape[0]), shard_idx=i, num_devices=num_devices
    )

    return weight[start:end]


def replicate_sharding_strategy(
    weight: Weight, i: int, num_devices: int
) -> TensorValue:
    """Replicates the entire weight tensor for a given device.

    Args:
        weight: The :obj:`Weight` to replicate.
        i: The index of the current device (unused in this strategy).
        num_devices: The total number of devices (unused in this strategy).

    Returns:
        A :obj:`TensorValue` representing the full weight tensor.
    """
    return weight[:]


@dataclass(frozen=True)
class ShardingStrategy:
    """Specifies how a :obj:`Weight` should be sharded across multiple devices.

    This class encapsulates a sharding function and the number of devices
    over which to shard. It provides static methods for common sharding
    patterns like row-wise, column-wise, and replication.
    """

    num_devices: int
    """The number of devices to shard the weight across."""

    shard: Callable[[Weight, int, int], TensorValue]
    """A callable that takes a :obj:`Weight`, a device index, and the total
    number of devices, and returns the sharded :obj:`TensorValue` for that
    device.
    """

    def __call__(self, weight: Weight, i: int) -> TensorValue:
        """Applies the sharding strategy to a given weight for a specific device.

        Args:
            weight: The :obj:`Weight` to be sharded.
            i: The index of the device for which to get the shard.

        Returns:
            A :obj:`TensorValue` representing the portion of the weight for the
            i-th device.
        """
        return self.shard(weight, i, self.num_devices)

    @property
    def is_rowwise(self) -> bool:
        """Whether the sharding strategy is row-wise."""
        return self.shard is row_sharding_strategy

    @property
    def is_colwise(self) -> bool:
        """Whether the sharding strategy is column-wise."""
        return self.shard is col_sharding_strategy

    @property
    def is_replicate(self) -> bool:
        """Whether the sharding strategy is replicate."""
        return self.shard is replicate_sharding_strategy

    @staticmethod
    def rowwise(num_devices: int) -> ShardingStrategy:
        """Creates a row-wise sharding strategy.

        This strategy shards the weight along its first axis (axis=0).

        Args:
            num_devices: The number of devices to shard the weight across.

        Returns:
            A :obj:`ShardingStrategy` instance configured for row-wise sharding.
        """
        return ShardingStrategy(
            num_devices=num_devices, shard=row_sharding_strategy
        )

    @staticmethod
    def columnwise(num_devices: int) -> ShardingStrategy:
        """Creates a column-wise sharding strategy.

        This strategy shards the weight along its second axis (axis=1).

        Args:
            num_devices: The number of devices to shard the weight across.

        Returns:
            A :obj:`ShardingStrategy` instance configured for column-wise sharding.
        """
        return ShardingStrategy(
            num_devices=num_devices, shard=col_sharding_strategy
        )

    @staticmethod
    def replicate(num_devices: int) -> ShardingStrategy:
        """Creates a replication strategy.

        This strategy replicates the entire weight on each device.

        Args:
            num_devices: The number of devices (primarily for consistency, as
                the weight is replicated).

        Returns:
            A :obj:`ShardingStrategy` instance configured for replication.
        """
        return ShardingStrategy(
            num_devices=num_devices, shard=replicate_sharding_strategy
        )


@dataclass
class _ShardingStrategyContainer:
    """Container for the sharding strategy and the host weight."""

    host_weight: Weight
    shard_value: ShardingStrategy
    """Defines how to shard a weight onto multiple devices.

    Should be a callable that take the host weight, and the shard index, and
    returns the shard value.
    """


class Weight(TensorValue):
    """Represents a value in a Graph that can be loaded at a later time.

    Weights can be initialized outside of a `Graph` and are lazily-added to
    the parent graph when used. If there is no parent graph when a weight is
    used, an error will be raised.
    """

    _dtype: DType
    _shape: ShapeLike
    _device: DeviceRef
    quantization_encoding: Optional[QuantizationEncoding]
    align: Optional[int]
    _sharding_strategy: Optional[_ShardingStrategyContainer]
    shard_idx: Optional[int]

    def __new__(cls, *args, **kwargs):
        # Skip the `Value.__new__` method to avoid staging a `TensorValue`.
        # A `Weight` can be initialized outside of a graph, but must be located
        # within a graph when operating on it.
        return super(Value, Weight).__new__(cls)

    def __init__(
        self,
        name: str,
        dtype: DType,
        shape: ShapeLike,
        device: DeviceRef,
        quantization_encoding: Optional[QuantizationEncoding] = None,
        align: Optional[int] = None,
        sharding_strategy: Optional[ShardingStrategy] = None,
        _placeholder: bool = False,
    ):
        self.name = name
        self._dtype = dtype
        self._shape = shape
        self._device = device
        self.quantization_encoding = quantization_encoding
        self.align = align
        self._sharding_strategy = (
            _ShardingStrategyContainer(self, sharding_strategy)
            if sharding_strategy
            else None
        )
        self.shard_idx = None
        self._placeholder = _placeholder

    @property
    def dtype(self) -> DType:
        return self._dtype

    @property
    def shape(self) -> Shape:
        if self.shard_idx is not None:
            # If this is a weight shard, then the weight shape will not
            # match the actual shard shape. The correct shape is the shape of
            # the TensorValue computed from the sharding strategy.
            return super().shape
        return Shape(self._shape)

    @cached_property
    def original_dtype_and_shape(self) -> tuple[DType, Shape]:
        """The original dtype and shape of this weight.

        This property should be used to store the original weight's dtype and
        shape the quantization encoding forces the weight to be loaded as uint8.
        """
        return self.dtype, self.shape

    @property
    def device(self) -> DeviceRef:
        return self._device

    @cached_property
    def _mlir_value(self) -> _Value[mo.TensorType]:  # type: ignore[override]
        if not self._sharding_strategy or self.shard_idx is None:
            return _add_weight_to_graph(self)._mlir_value
        host_weight = self._sharding_strategy.host_weight
        tensor_value = self._sharding_strategy.shard_value(
            host_weight, self.shard_idx
        )
        tensor_value = tensor_value.to(self._device)
        return tensor_value._mlir_value

    def __repr__(self):
        device_str = f", {self._device}"
        if self.quantization_encoding:
            return f"Weight({self.name}, {self.dtype}, {self.shape}{device_str}, {self.quantization_encoding})"
        else:
            return (
                f"Weight({self.name}, {self.dtype}, {self.shape}{device_str})"
            )

    @property
    def sharding_strategy(self) -> Optional[ShardingStrategy]:
        """Gets the weight sharding strategy."""
        return (
            self._sharding_strategy.shard_value
            if self._sharding_strategy
            else None
        )

    @sharding_strategy.setter
    def sharding_strategy(self, strategy: ShardingStrategy) -> None:
        """Sets the weight sharding strategy.

        Args:
            strategy: A ShardingStrategy that defines how to shard the weight.
        """
        # Reset device to CPU to avoid implicit transfer behavior
        # from `force_initial_weight_on_host`.
        # TODO(GEX-2121): Remove this hack
        self._device = DeviceRef.CPU()
        self._sharding_strategy = _ShardingStrategyContainer(self, strategy)

    def shard(self, shard_idx: int, device: DeviceRef) -> Weight:
        """Gets a specific shard from the Weight.

        This `Weight` must have `sharding_strategy` defined. The shard object
        returned is also a `Weight` object, but cannot be sharded further.

        Args:
            shard_idx: int value of the shard.
            device: device to place the shard.

        Returns:
            The sharded weight.
        """
        if not self.sharding_strategy:
            raise ValueError(
                f"Weight {self.name} cannot be sharded because no sharding strategy was provided."
            )
        if self.shard_idx is not None:
            raise ValueError(
                f"Weight {self.name} was already sharded. Use __getitem__ instead."
            )
        weight = Weight(
            name=f"{self.name}[{shard_idx}]",
            dtype=self._dtype,
            shape=self._shape,
            device=device,
            quantization_encoding=self.quantization_encoding,
            align=self.align,
        )

        # Copy the sharding strategy container directly to preserve the original host_weight
        # reference. Using the property setter would create a new container with this shard
        # as the host_weight, causing infinite recursion when the shard tries to compute
        # its shape by calling the sharding function on itself.
        weight._sharding_strategy = self._sharding_strategy
        weight.shard_idx = shard_idx
        return weight


def _add_weight_to_graph(weight: Weight):
    try:
        current_graph = graph.Graph.current
    except LookupError:
        raise ValueError(
            "Cannot operate on a `max.graph.Weight` when there is no parent graph."
        )

    # If the weight doesn't exist on the graph, `Graph.add_weight` will
    # return a new `TensorValue`. Otherwise, this will return the existing
    # `TensorValue`.
    return current_graph.add_weight(weight)
