# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
"""Test the max.graph Python bindings."""

from conftest import tensor_types
from hypothesis import assume, given, strategies as st

from max.dtype import DType
from max.graph import Graph, TensorType, ops
from max.graph.type import Shape


@given(input_type=...)
def test_tensor_value__T(input_type: TensorType):
    assume(input_type.rank >= 2)
    with Graph("transpose", input_types=[input_type]) as graph:
        out = graph.inputs[0].T
        expected = Shape(input_type.shape)
        expected[-1], expected[-2] = expected[-2], expected[-1]
        assert out.shape == expected

        graph.output(out)


@given(tensor_type=tensor_types(dtypes=st.just(DType.bool)))
def test_tensor_value__operator_logical_and(tensor_type: TensorType):
    with Graph("and", input_types=[tensor_type]) as graph:
        x, = graph.inputs
        assert (x & x).type == ops.logical_and(x, x).type


@given(tensor_type=tensor_types(dtypes=st.just(DType.bool)))
def test_tensor_value__operator_logical_or(tensor_type: TensorType):
    with Graph("or", input_types=[tensor_type]) as graph:
        x, = graph.inputs
        assert (x | x).type == ops.logical_or(x, x).type


@given(tensor_type=tensor_types(dtypes=st.just(DType.bool)))
def test_tensor_value__operator_logical_xor(tensor_type: TensorType):
    with Graph("xor", input_types=[tensor_type]) as graph:
        x, = graph.inputs
        assert (x ^ x).type == ops.logical_xor(x, x).type
