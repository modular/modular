# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from functools import cached_property
from typing import Callable, Optional, Union

import max.graph
import numpy.typing as npt
from max import mlir
from max._core_types.driver import DLPackArray
from max.driver import CPU, Accelerator, Tensor
from max.dtype import DType

from . import graph
from .quantization import QuantizationEncoding
from .type import DeviceKind, DeviceRef, Shape, ShapeLike
from .value import TensorValue, Value

ShardingStrategy = Callable[["Weight", int], TensorValue]
DLPackCompatible = Union[DLPackArray, npt.NDArray]


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
    sharding_strategy: Optional[_ShardingStrategyContainer]
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
    ):
        self.name = name
        self._dtype = dtype
        self._shape = shape
        self._device = device
        self.quantization_encoding = quantization_encoding
        self.align = align
        self.sharding_strategy = (
            _ShardingStrategyContainer(self, sharding_strategy)
            if sharding_strategy
            else None
        )
        self.shard_idx = None

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

    @property
    def _mlir_value(self) -> mlir.Value:
        if self.sharding_strategy and self.shard_idx is not None:
            host_weight = self.sharding_strategy.host_weight
            tensor_value = self.sharding_strategy.shard_value(
                host_weight, self.shard_idx
            )
            tensor_value = tensor_value.to(self._device)
            return tensor_value._mlir_value
        else:
            res = _add_weight_to_graph(self)._mlir_value
            return res

    @_mlir_value.setter
    def _mlir_value(self, value: mlir.Value) -> None:
        raise ValueError("Cannot re-define Weight._mlir_value.")

    def __repr__(self):
        device_str = f", {self._device}"
        if self.quantization_encoding:
            return f"Weight({self.name}, {self.dtype}, {self.shape}{device_str}, {self.quantization_encoding})"
        else:
            return (
                f"Weight({self.name}, {self.dtype}, {self.shape}{device_str})"
            )

    def set_sharding_strategy(
        self, sharding_strategy: ShardingStrategy
    ) -> None:
        """Set the weight sharding strategy.

        Args:
            sharding_strategy: A callable that takes the host weight and shard
              index, and returns the sharded value.
        """
        # Reset device to CPU to avoid implicit transfer behavior
        # from `force_initial_weight_on_host`.
        # TODO(GEX-2121): Remove this hack
        self._device = DeviceRef.CPU()
        self.sharding_strategy = _ShardingStrategyContainer(
            self, sharding_strategy
        )

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
        weight.sharding_strategy = self.sharding_strategy
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


# TODO(GEX-2126): Remove this
def _reconcile_weights(
    model: max.graph.Graph, weights_registry: Mapping[str, DLPackCompatible]
) -> Mapping[str, Tensor]:
    """Returns a new weight registry where weights are on the proper device.

    This is mainly as a utility internal testing.

    Args:
        model: The model containing information on which device weights should be.
        weights_registry: The weight registry. Not all weights have ot be on the
            proper device.

    Returns:
        A new weight registry where all weights are on the proper device.

    Raises:
        ValueError: If model does not contain any weight metadata. If
            weights_registry does not contain all weights in model, and
            only weights in model. If an unknown device is required.
            If model is not a max graph.
    """
    if not hasattr(model, "_weights"):
        raise ValueError("Given model has no weights metadata!")

    # TODO(GEX-2095): Enable this
    # if weights_registry.keys() != model._weights.keys():
    #     raise ValueError(
    #         f"Weight registry passed contains keys {sorted(weights_registry.keys())} but model expects {sorted(model._weights.keys())}"
    #     )

    new_registry = dict()
    for weight_name, weight in model._weights.items():
        if weight_name not in weights_registry:
            raise ValueError(
                f"Weight '{weight_name}' is not in the weights registry."
            )

        # TODO: just use dlpack devices to handle translation layer between MAX and torch stack
        def to_tensor() -> Tensor:
            registered_weight = Tensor.from_dlpack(
                weights_registry[weight_name]
            )

            expected_device = weight.value.device
            assert expected_device is not None, (
                "Expect all weights to have device now."
            )
            if expected_device.device_type is DeviceKind.CPU:
                assert expected_device.id == 0, "Expect only one host device."
                return registered_weight.to(CPU(expected_device.id))
            elif expected_device.device_type is DeviceKind.GPU:
                return registered_weight.to(Accelerator(expected_device.id))
            else:
                raise ValueError(
                    f"Do not know how to handle device {expected_device}"
                )

        new_registry[weight_name] = to_tensor()
    return new_registry
