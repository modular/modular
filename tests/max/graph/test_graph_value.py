# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
"""Test the max.graph Python bindings."""

from hypothesis import assume, given
from max.graph import Graph, TensorType
from max.graph.type import Shape


@given(input_type=...)
def test_graph_value__T(input_type: TensorType):
    assume(input_type.rank >= 2)
    with Graph("transpose", input_types=[input_type]) as graph:
        out = graph.inputs[0].T
        expected = Shape(input_type.shape)
        expected[-1], expected[-2] = expected[-2], expected[-1]
        assert out.shape == expected

        graph.output(out)
        graph._mlir_op.verify()
