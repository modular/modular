# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

from __future__ import annotations

from typing import Union, Iterable

import numpy as np
from max import mlir

from . import ops
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

    def print(self, label: str = "debug_tensor"):
        ops.print(self, label=label)

    def reshape(self, shape: ShapeLike) -> GraphValue:
        return ops.reshape(self, shape)

    def transpose(self, dim_1: int, dim_2: a) -> GraphValue:
        return ops.transpose(self, dim_1, dim_2)

    def __getitem__(self, index):
        if isinstance(index, Iterable):
            return ops.slice_tensor(self, index)
        else:
            # Need to wrap it to make it iterable
            return ops.slice_tensor(self, [index])

    def __eq__(self, rhs: any) -> GraphValue:
        if isinstance(rhs, ValueLike):
            return ops.equal(self, GraphValue(rhs))
        else:
            raise ValueError(f"can't compare GraphValue to {rhs}")

    def __neg__(self) -> GraphValue:
        return ops.negate(self)

    def __ne__(self, rhs: any) -> GraphValue:
        if isinstance(rhs, ValueLike):
            return ops.not_equal(self, GraphValue(rhs))
        else:
            raise ValueError(f"can't compare GraphValue to {rhs}")

    def __ge__(self, rhs: any) -> GraphValue:
        if isinstance(rhs, ValueLike):
            return ops.greater_equal(self, GraphValue(rhs))
        else:
            raise ValueError(f"can't compare GraphValue to {rhs}")

    def __gt__(self, rhs: any) -> GraphValue:
        if isinstance(rhs, ValueLike):
            return ops.greater(self, GraphValue(rhs))
        else:
            raise ValueError(f"can't compare GraphValue to {rhs}")

    def __lt__(self, rhs: any) -> GraphValue:
        return ops.logical_not(self >= rhs)

    def __le__(self, rhs: any) -> GraphValue:
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

    def __divmod__(self, rhs: ValueLike) -> (GraphValue, GraphValue):
        rhs = GraphValue(rhs)
        return (self // rhs, self % rhs)

    def __rdivmod__(self, lhs: ValueLike) -> (GraphValue, GraphValue):
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
