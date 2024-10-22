# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
"""Test the max.graph Python bindings."""

from functools import reduce

from conftest import broadcast_shapes, broadcastable_tensor_types
from hypothesis import given
from hypothesis import strategies as st
from max.dtype import DType
from max.graph import Graph, TensorType, ops


@given(input_types=broadcastable_tensor_types(3))
def test_select(input_types: list[TensorType]):
    input_types[0].dtype = DType.bool

    with Graph("select", input_types=input_types) as graph:
        cond, x, y = graph.inputs
        out = ops.select(cond, x, y)

        expected = reduce(broadcast_shapes, (t.shape for t in input_types))
        assert out.shape == expected
        assert out.dtype in (t.dtype for t in input_types)

        graph.output(out)
