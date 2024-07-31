# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
"""Slicing ops."""

from typing import Union

from max import _graph, mlir
from max.mlir.dialects import rmo

from ..graph import Graph
from ..graph_value import GraphValue, ValueLike
from ..type import ShapeLike, StaticDim


def select(cond: ValueLike, x: ValueLike, y: ValueLike):
    return Graph.current._add_op(
        rmo.select, GraphValue(cond), GraphValue(x), GraphValue(y)
    )[0]
