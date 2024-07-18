# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
"""Elementwise ops."""

from ..graph import Graph
from ..graph_value import GraphValue
from ..mlir.dialects import rmo


def _elementwise_binary(op):
    def elementwise_op(lhs: GraphValue, rhs: GraphValue) -> GraphValue:
        return Graph.current._add_op(op, lhs, rhs)[0]

    elementwise_op.__name__ = op.__name__
    return elementwise_op


def _elementwise_unary(op):
    def elementwise_op(x: GraphValue) -> GraphValue:
        return Graph.current._add_op(op, x)[0]

    elementwise_op.__name__ = op.__name__
    return elementwise_op


add = _elementwise_binary(rmo.add)
mul = _elementwise_binary(rmo.mul)
