# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
"""Test the max.graph Python bindings."""

from conftest import axes, tensor_types
from hypothesis import assume, given
from hypothesis import strategies as st
from max.graph import Graph, TensorType

shared_types = st.shared(tensor_types())


@given(input_type=shared_types, a=axes(shared_types), b=axes(shared_types))
def test_transpose__output_shape(input_type: TensorType, a: int, b: int):
    assume(input_type.rank > 0)
    with Graph("transpose", input_types=[input_type]) as graph:
        out = graph.inputs[0].transpose(a, b)
        target_shape = list(input_type.shape)
        target_shape[a], target_shape[b] = target_shape[b], target_shape[a]
        assert out.shape == target_shape

        graph.output(out)
