# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

from __future__ import annotations

import inspect
from typing import Any, Iterable, Union, TypeGuard

from max import _graph, mlir
import numpy as np

from . import graph
from . import ops
from .dtype import DType
from .type import Shape, ShapeLike, TensorType


class GraphValue:
    """Represents a symbolic value within a `Graph`.

    A `GraphValue` can represent the output of a node, the arguments of a
    `Graph` (as seen from within its body), and more generally any symbolic
    value available within the `Graph`. Other nodes receive `GraphValue`
    values as inputs to form a computation graph.

    A `GraphValue` may also refer to an existing input or output of a node,
    and you can change them, such as by swapping a new `GraphValue`.

    Conceptually, think of a `GraphValue` as an edge in the dataflow graph,
    with the other end being the user of that value.

    Similar to a regular variable, a `GraphValue` has a data type.

    Note: All the methods in this type are documented as "Creates foo". This is
    a shorthand notation for "Adds a node representing an op that returns foo".
    """

    _mlir_value: mlir.Value

    def __init__(self, value: ValueLike) -> None:
        if isinstance(value, mlir.Value):
            self._mlir_value = value
        elif isinstance(value, GraphValue):
            self._mlir_value = value._mlir_value
        elif isinstance(value, np.ndarray):
            self._mlir_value = ops.constant(value)._mlir_value
        else:
            raise ValueError(f"can't construct GraphValue from {value}")

    @property
    def tensor_type(self) -> TensorType:
        """Returns the type of the GraphValue as a TensorType.

        Will raise if the type is not TensorType.
        """
        return TensorType.from_mlir(self._mlir_value.type)

    @property
    def shape(self) -> Shape:
        """Returns the shape of the GraphValue.

        Will raise if the type is not TensorType.
        """
        return self.tensor_type.shape

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

    def reshape(self, shape: ShapeLike) -> GraphValue:
        return ops.reshape(self, shape)

    def rebind(self, shape: ShapeLike) -> GraphValue:
        # For rebind, we create a runtime stack location as the message.
        frame = inspect.currentframe()
        return ops.rebind(self, shape, graph._frame_str(frame))

    def transpose(self, dim_1: int, dim_2: int) -> GraphValue:
        return ops.transpose(self, dim_1, dim_2)

    def __getitem__(self, index):
        if isinstance(index, Iterable):
            return ops.slice_tensor(self, index)
        else:
            # Need to wrap it to make it iterable
            return ops.slice_tensor(self, [index])

    def __eq__(self, rhs: Any) -> GraphValue:  # type: ignore[override]
        if _is_value_like(rhs):
            return ops.equal(self, GraphValue(rhs))
        else:
            raise ValueError(f"can't compare GraphValue to {rhs}")

    def __neg__(self) -> GraphValue:
        return ops.negate(self)

    def __ne__(self, rhs: Any) -> GraphValue:  # type: ignore[override]
        if _is_value_like(rhs):
            return ops.not_equal(self, GraphValue(rhs))
        else:
            raise ValueError(f"can't compare GraphValue to {rhs}")

    def __ge__(self, rhs: Any) -> GraphValue:
        if _is_value_like(rhs):
            return ops.greater_equal(self, GraphValue(rhs))
        else:
            raise ValueError(f"can't compare GraphValue to {rhs}")

    def __gt__(self, rhs: Any) -> GraphValue:
        if _is_value_like(rhs):
            return ops.greater(self, GraphValue(rhs))
        else:
            raise ValueError(f"can't compare GraphValue to {rhs}")

    def __lt__(self, rhs: Any) -> GraphValue:
        return ops.logical_not(self >= rhs)

    def __le__(self, rhs: Any) -> GraphValue:
        return ops.logical_not(self > rhs)

    def __add__(self, rhs: ValueLike) -> GraphValue:
        return ops.add(self, GraphValue(rhs))

    def __radd__(self, lhs: ValueLike) -> GraphValue:
        return ops.add(GraphValue(lhs), self)

    def __sub__(self, rhs: ValueLike) -> GraphValue:
        return ops.sub(self, GraphValue(rhs))

    def __rsub__(self, lhs: ValueLike) -> GraphValue:
        return ops.sub(GraphValue(lhs), self)

    def __mul__(self, rhs: ValueLike) -> GraphValue:
        return ops.mul(self, GraphValue(rhs))

    def __rmul__(self, lhs: ValueLike) -> GraphValue:
        return ops.mul(GraphValue(lhs), self)

    def __truediv__(self, rhs: ValueLike) -> GraphValue:
        return ops.div(self, GraphValue(rhs))

    def __rtruediv__(self, lhs: ValueLike) -> GraphValue:
        return ops.div(GraphValue(lhs), self)

    def __floordiv__(self, rhs: ValueLike) -> GraphValue:
        return ops.floor(ops.div(self, GraphValue(rhs)))

    def __rfloordiv__(self, lhs: ValueLike) -> GraphValue:
        return ops.floor(ops.div(GraphValue(lhs), self))

    def __mod__(self, rhs: ValueLike) -> GraphValue:
        return ops.mod(self, GraphValue(rhs))

    def __rmod__(self, lhs: ValueLike) -> GraphValue:
        return ops.mod(GraphValue(lhs), self)

    def __divmod__(self, rhs: ValueLike) -> tuple[GraphValue, GraphValue]:
        rhs = GraphValue(rhs)
        return (self // rhs, self % rhs)

    def __rdivmod__(self, lhs: ValueLike) -> tuple[GraphValue, GraphValue]:
        lhs = GraphValue(lhs)
        return (lhs // self, lhs % self)

    def __matmul__(self, rhs: ValueLike) -> GraphValue:
        return ops.matmul(self, GraphValue(rhs))

    def __rmatmul__(self, lhs: ValueLike) -> GraphValue:
        return ops.matmul(GraphValue(lhs), self)

    def __pow__(self, rhs: ValueLike) -> GraphValue:
        return ops.pow(self, GraphValue(rhs))

    def __rpow__(self, lhs: ValueLike) -> GraphValue:
        return ops.pow(GraphValue(lhs), self)


ValueLike = Union[mlir.Value, GraphValue, np.ndarray]


def _is_value_like(obj: Any) -> TypeGuard[ValueLike]:
    return isinstance(obj, (mlir.Value, GraphValue, np.ndarray))
