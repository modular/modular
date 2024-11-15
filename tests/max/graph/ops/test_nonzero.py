# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
"""Test the max.graph Python bindings."""

from hypothesis import given
from max.dtype import DType
from max.graph import Graph, TensorType, ops


@given(input_type=...)
def test_nonzero(input_type: TensorType):
    with Graph("nonzero ", input_types=[input_type]) as graph:
        out = ops.nonzero(graph.inputs[0], "nonzero")
        assert out.dtype == DType.int64
        assert out.shape == ["nonzero", input_type.rank]
        graph.output(out)
