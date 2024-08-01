# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
"""Ops to help with debugging."""
from max.mlir.dialects import mo

from ..graph import Graph
from ..graph_value import GraphValue


def print(value: GraphValue, label: str = "debug_tensor"):
    Graph.current._add_op(mo.debug_tensor_print, value, label=label)
