# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
"""Ops to help with debugging."""
from max.mlir.dialects import mo

from ..graph import Graph
from ..value import TensorValue


def print(value: TensorValue, label: str = "debug_tensor"):
    Graph.current._add_op(mo.debug_tensor_print, value, label=label)
