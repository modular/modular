# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
"""Ops to help with debugging."""

from __future__ import annotations

from max.mlir.dialects import mo

from ..graph import Graph
from ..value import TensorValue


def print(value: str | TensorValue, label: str = "debug_tensor"):
    in_chain = Graph.current._current_chain

    op = mo.debug_print if isinstance(value, str) else mo.debug_tensor_print

    output = Graph.current._add_op(op, in_chain, value, label=label)[0]
    Graph.current._update_chain(output)
