# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
"""Test the max.graph Python bindings."""

import sys
from random import Random
from operator import mul

import pytest
from hypothesis import strategies as st
from hypothesis import given, settings
from max import mlir
from max.graph import DType, Graph, GraphValue, TensorType, graph, ops
from max.graph.type import shape, Shape, dim, StaticDim, Dim


# Instead of testing mlir string escaping, just limit the label to something reasonable.
printable_ascii = st.characters(min_codepoint=ord(" "), max_codepoint=ord("~"))


@given(input_type=..., label=printable_ascii)
def test_prints(input_type: TensorType, label: str):
    with Graph(
        "print",
        input_types=[input_type],
    ) as graph:
        out = graph.inputs[0]
        out.print(label)

        graph.output(out)
        graph._mlir_op.verify()
        assert "mo.debug.tensor.print" in str(graph._mlir_op)
        assert label in str(graph._mlir_op)
