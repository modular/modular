# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
"""Test the max.graph Python bindings."""

from hypothesis import given
from hypothesis import strategies as st
from max.graph import Graph, TensorType, ops

# Instead of testing mlir string escaping, just limit the label to something reasonable.
printable_ascii = st.characters(min_codepoint=ord(" "), max_codepoint=ord("~"))


@given(input_type=..., label1=printable_ascii, label2=printable_ascii)
def test_tensor_prints(input_type: TensorType, label1: str, label2: str):
    with Graph("print_tensors", input_types=[input_type]) as graph:
        out = graph.inputs[0]
        chain_0 = graph._current_chain
        out.print(label1)
        chain_1 = graph._current_chain
        out.print(label2)
        chain_2 = graph._current_chain

        graph.output(out)

        assert str(graph._mlir_op).count("mo.debug.tensor.print") == 2
        assert label1 in str(graph._mlir_op)
        assert label2 in str(graph._mlir_op)

        assert chain_0 != chain_1
        assert chain_1 != chain_2


@given(
    msg1=printable_ascii,
    label1=printable_ascii,
    msg2=printable_ascii,
    label2=printable_ascii,
)
def test_prints(msg1: str, label1: str, msg2: str, label2: str):
    with Graph("print") as graph:
        chain_0 = graph._current_chain
        ops.print(msg1, label1)
        chain_1 = graph._current_chain
        ops.print(msg2, label2)
        chain_2 = graph._current_chain

        graph.output()

        assert str(graph._mlir_op).count("mo.debug.print") == 2
        assert msg1 in str(graph._mlir_op)
        assert label1 in str(graph._mlir_op)
        assert msg2 in str(graph._mlir_op)
        assert label2 in str(graph._mlir_op)

        assert chain_0 != chain_1
        assert chain_1 != chain_2
