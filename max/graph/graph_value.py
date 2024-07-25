# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

from __future__ import annotations

from dataclasses import dataclass

from max import mlir

from . import ops
from .type import ShapeLike


@dataclass
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

    def __add__(self, other: GraphValue) -> GraphValue:
        return ops.add(self, other)

    def reshape(self, shape: ShapeLike) -> GraphValue:
        return ops.reshape(self, shape)
