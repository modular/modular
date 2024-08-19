# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
"""Test the max.graph Python bindings."""

from hypothesis import given
from hypothesis import strategies as st
from max.graph import Graph, TensorType

# Instead of testing mlir string escaping, just limit the label to something reasonable.
printable_ascii = st.characters(min_codepoint=ord(" "), max_codepoint=ord("~"))


@given(input_type=..., label=printable_ascii)
def test_prints(input_type: TensorType, label: str):
    with Graph("print", input_types=[input_type]) as graph:
        out = graph.inputs[0]
        out.print(label)

        graph.output(out)
        assert "mo.debug.tensor.print" in str(graph._mlir_op)
        assert label in str(graph._mlir_op)
