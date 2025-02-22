# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

from functools import cached_property
from typing import Optional

from max import mlir
from max.dtype import DType

from . import graph
from .quantization import QuantizationEncoding
from .type import DeviceRef, Shape, ShapeLike
from .value import TensorValue, Value


class Weight(TensorValue):
    """Represents a value in a Graph that can be loaded at a later time.

    Weights can be initialized outside of a `Graph` and are lazily-added to
    the parent graph when used. If there is no parent graph when a weight is
    used, an error will be raised.
    """

    _dtype: DType
    _shape: ShapeLike
    _device: Optional[DeviceRef]
    quantization_encoding: Optional[QuantizationEncoding]
    align: Optional[int]

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
        device: Optional[DeviceRef] = None,
        quantization_encoding: Optional[QuantizationEncoding] = None,
        align: Optional[int] = None,
    ):
        self.name = name
        self._dtype = dtype
        self._shape = shape
        self._device = device
        self.quantization_encoding = quantization_encoding
        self.align = align

    @property
    def dtype(self) -> DType:
        return self._dtype

    @property
    def shape(self) -> Shape:
        return Shape(self._shape)

    @cached_property
    def original_dtype_and_shape(self) -> tuple[DType, Shape]:
        """The original dtype and shape of this weight.

        This property should be used to store the original weight's dtype and
        shape the quantization encoding forces the weight to be loaded as uint8.
        """
        return self.dtype, self.shape

    @property
    def _mlir_value(self) -> mlir.Value:
        try:
            current_graph = graph.Graph.current
        except LookupError:
            raise ValueError(
                "Cannot operate on a `max.graph.Weight` when there is no parent graph."
            )

        # If the weight doesn't exist on the graph, `Graph.add_weight` will
        # return a new `TensorValue`. Otherwise, this will return the existing
        # `TensorValue`.
        return current_graph.add_weight(self, self._device)._mlir_value

    @_mlir_value.setter
    def _mlir_value(self, value: mlir.Value) -> None:
        raise ValueError("Cannot re-define Weight._mlir_value.")

    def __repr__(self):
        if self.quantization_encoding:
            return f"Weight({self.dtype}, {self.shape}, {self.device}, {self.quantization_encoding})"
        else:
            return f"Weight({self.dtype}, {self.shape}, {self.device})"
