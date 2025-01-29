# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

from typing import Optional

from max import mlir
from max.dtype import DType

from . import graph
from .quantization import QuantizationEncoding
from .type import Shape, ShapeLike
from .value import TensorValue, Value


class Weight(TensorValue):
    """Represents a value in a Graph that can be loaded at a later time.

    Weights can be initialized outside of a `Graph` and are lazily-added to
    the parent graph when used. If there is no parent graph when a weight is
    used, an error will be raised.
    """

    _name: str
    _dtype: DType
    _shape: ShapeLike
    quantization_encoding: Optional[QuantizationEncoding]
    align: Optional[int]
    __mlir_value: Optional[mlir.Value]

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
        quantization_encoding: Optional[QuantizationEncoding] = None,
        align: Optional[int] = None,
    ):
        self.name = name
        self._dtype = dtype
        self._shape = shape
        self.quantization_encoding = quantization_encoding
        self.align = align
        self.__mlir_value = None

    @property
    def dtype(self) -> DType:
        return self._dtype

    @property
    def shape(self) -> Shape:
        return Shape(self._shape)

    @property
    def _mlir_value(self) -> mlir.Value:
        if not self.__mlir_value:
            try:
                self.__mlir_value = graph.Graph.current.add_weight(
                    self
                )._mlir_value
            except LookupError:
                raise ValueError(
                    "Cannot operate on a `max.graph.Weight` when there is no parent graph."
                )
        assert self.__mlir_value is not None
        return self.__mlir_value

    @_mlir_value.setter
    def _mlir_value(self, value: mlir.Value) -> None:
        raise ValueError("Cannot re-define Weight._mlir_value.")
