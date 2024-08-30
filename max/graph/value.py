# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

from __future__ import annotations

import inspect
import sys
from typing import Any, Iterable, Union

if sys.version_info >= (3, 10):
    from typing import TypeGuard
else:
    from typing_extensions import TypeGuard

import numpy as np
from max import _graph, mlir
from max.dtype import DType

from . import graph, ops
from .type import DimLike, Shape, ShapeLike, TensorType
from .weight import Weight


class Value:
    """Represents a symbolic value within a `Graph`.

    A `Value` can represent the output of a node, the arguments of a
    `Graph` (as seen from within its body), and more generally any symbolic
    value available within the `Graph`. Other nodes receive `Value`
    values as inputs to form a computation graph.

    A `Value` may also refer to an existing input or output of a node,
    and you can change them, such as by swapping a new `Value`.

    Conceptually, think of a `Value` as an edge in the dataflow graph,
    with the other end being the user of that value.

    Similar to a regular variable, a `Value` has a data type.

    Note: All the methods in this type are documented as "Creates foo". This is
    a shorthand notation for "Adds a node representing an op that returns foo".
    """

    _mlir_value: mlir.Value

    def __new__(cls, value: ValueLike):
        if isinstance(value, mlir.Value):
            return super().__new__(TensorValue)
        elif isinstance(value, Value):
            return super().__new__(type(value))
        elif isinstance(value, (np.ndarray, Weight)):
            return super().__new__(TensorValue)
        else:
            raise TypeError(
                "Value() argument must be a mlir.Value, a graph.Value, or an"
                f" np.ndarray, not '{type(value).__name__}'"
            )

    def __repr__(self):
        return str(self._mlir_value.type)

    @property
    def tensor(self) -> TensorValue:
        """Returns the the Value as a TensorValue.

        Raise an exception if the Value is not a TensorValue.
        """
        if isinstance(self, TensorValue):
            return self

            raise TypeError(
                f"Value is not a TensorValue, was '{type(self).__name__}'"
            )


class TensorValue(Value):
    """Represents a value semantic tensor within a `Graph`."""

    def __init__(self, value: ValueLike) -> None:
        if isinstance(value, mlir.Value):
            if _graph.type_is_tensor(value.type):
                self._mlir_value = value
            else:
                raise TypeError(
                    "TensorValue() argument must be a mlir.Value of tensor"
                    " type, a graph.TensorValue, or an np.ndarray, not"
                    f" '{type(value).__name__}'"
                )
        elif isinstance(value, TensorValue):
            self._mlir_value = value._mlir_value
        elif isinstance(value, np.ndarray):
            self._mlir_value = ops.constant(value)._mlir_value
        elif isinstance(value, Weight):
            self._mlir_value = value.add_to_graph(
                graph.Graph.current
            )._mlir_value
        else:
            raise TypeError(
                "TensorValue() argument must be a mlir.Value of tensor type, a"
                " graph.TensorValue, or an np.ndarray, not"
                f" '{type(value).__name__}'"
            )

    # TODO(MSDK-662): Should both DimLike and ShapeLike be considered ValueLike now?
    @staticmethod
    def from_dim(dim: DimLike) -> TensorValue:
        return ops.shape_to_tensor([dim]).reshape(())

    @staticmethod
    def from_shape(shape: ShapeLike) -> TensorValue:
        return ops.shape_to_tensor(shape)

    def __repr__(self):
        dtype = self.dtype
        shape = self.shape
        return f"{type(self).__name__}({dtype=}, {shape=})"

    @property
    def type(self) -> TensorType:
        """Returns the type of the TensorValue as a TensorType."""
        return TensorType.from_mlir(self._mlir_value.type)

    @property
    def shape(self) -> Shape:
        """Returns the shape of the TensorValue."""
        return self.type.shape

    # dtype and rank are special.
    # They use _graph directly to avoid loading the shape dimension if they aren't needed.
    # This also avoids accidentally loading algebraic expression dimensions (which will throw an exception).
    @property
    def dtype(self) -> DType:
        t = self._mlir_value.type
        if not _graph.type_is_tensor(t):
            raise TypeError(f"Expected TensorType, got: {t}")

        return DType(_graph.tensor_type_get_dtype(t))

    @property
    def rank(self) -> int:
        t = self._mlir_value.type
        if not _graph.type_is_tensor(t):
            raise TypeError(f"Expected TensorType, got: {t}")

        return _graph.tensor_type_get_rank(t)

    def print(self, label: str = "debug_tensor"):
        ops.print(self, label=label)

    def reshape(self, shape: ShapeLike) -> TensorValue:
        return ops.reshape(self, shape)

    def rebind(self, shape: ShapeLike) -> TensorValue:
        # For rebind, we create a runtime stack location as the message.
        frame = inspect.currentframe()
        return ops.rebind(self, shape, graph._frame_str(frame))

    def transpose(self, dim_1: int, dim_2: int) -> TensorValue:
        return ops.transpose(self, dim_1, dim_2)

    @property
    def T(self) -> TensorValue:
        return self.transpose(-1, -2)

    def __getitem__(self, index):
        return ops.slice_tensor(
            self, index if isinstance(index, Iterable) else (index,)
        )

    def __eq__(self, rhs: Any) -> TensorValue:  # type: ignore[override]
        if _is_value_like(rhs):
            return ops.equal(self, TensorValue(rhs))
        else:
            raise TypeError(
                "'==' not supported between instance of"
                f" '{type(self).__name__}' and '{type(rhs).__name__}'"
            )

    def __neg__(self) -> TensorValue:
        return ops.negate(self)

    def __ne__(self, rhs: Any) -> TensorValue:  # type: ignore[override]
        if _is_value_like(rhs):
            return ops.not_equal(self, TensorValue(rhs))
        else:
            raise TypeError(
                "'!=' not supported between instance of"
                f" '{type(self).__name__}' and '{type(rhs).__name__}'"
            )

    def __ge__(self, rhs: Any) -> TensorValue:
        if _is_value_like(rhs):
            return ops.greater_equal(self, TensorValue(rhs))
        else:
            raise TypeError(
                "'>=' not supported between instance of"
                f" '{type(self).__name__}' and '{type(rhs).__name__}'"
            )

    def __gt__(self, rhs: Any) -> TensorValue:
        if _is_value_like(rhs):
            return ops.greater(self, TensorValue(rhs))
        else:
            raise TypeError(
                f"'>' not supported between instance of '{type(self).__name__}'"
                f" and '{type(rhs).__name__}'"
            )

    def __lt__(self, rhs: Any) -> TensorValue:
        return ops.logical_not(self >= rhs)

    def __le__(self, rhs: Any) -> TensorValue:
        return ops.logical_not(self > rhs)

    def __add__(self, rhs: ValueLike) -> TensorValue:
        return ops.add(self, TensorValue(rhs))

    def __radd__(self, lhs: ValueLike) -> TensorValue:
        return ops.add(TensorValue(lhs), self)

    def __sub__(self, rhs: ValueLike) -> TensorValue:
        return ops.sub(self, TensorValue(rhs))

    def __rsub__(self, lhs: ValueLike) -> TensorValue:
        return ops.sub(TensorValue(lhs), self)

    def __mul__(self, rhs: ValueLike) -> TensorValue:
        return ops.mul(self, TensorValue(rhs))

    def __rmul__(self, lhs: ValueLike) -> TensorValue:
        return ops.mul(TensorValue(lhs), self)

    def __truediv__(self, rhs: ValueLike) -> TensorValue:
        return ops.div(self, TensorValue(rhs))

    def __rtruediv__(self, lhs: ValueLike) -> TensorValue:
        return ops.div(TensorValue(lhs), self)

    def __floordiv__(self, rhs: ValueLike) -> TensorValue:
        return ops.floor(ops.div(self, TensorValue(rhs)))

    def __rfloordiv__(self, lhs: ValueLike) -> TensorValue:
        return ops.floor(ops.div(TensorValue(lhs), self))

    def __mod__(self, rhs: ValueLike) -> TensorValue:
        return ops.mod(self, TensorValue(rhs))

    def __rmod__(self, lhs: ValueLike) -> TensorValue:
        return ops.mod(TensorValue(lhs), self)

    def __divmod__(self, rhs: ValueLike) -> tuple[TensorValue, TensorValue]:
        rhs = TensorValue(rhs)
        return (self // rhs, self % rhs)

    def __rdivmod__(self, lhs: ValueLike) -> tuple[TensorValue, TensorValue]:
        lhs = TensorValue(lhs)
        return (lhs // self, lhs % self)

    def __matmul__(self, rhs: ValueLike) -> TensorValue:
        return ops.matmul(self, TensorValue(rhs))

    def __rmatmul__(self, lhs: ValueLike) -> TensorValue:
        return ops.matmul(TensorValue(lhs), self)

    def __pow__(self, rhs: ValueLike) -> TensorValue:
        return ops.pow(self, TensorValue(rhs))

    def __rpow__(self, lhs: ValueLike) -> TensorValue:
        return ops.pow(TensorValue(lhs), self)


ValueLike = Union[mlir.Value, Value, np.ndarray, Weight]


def _is_value_like(obj: Any) -> TypeGuard[ValueLike]:
    return isinstance(obj, (mlir.Value, Value, np.ndarray, Weight))
