# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
"""test the max.graph python bindings."""

import pytest
from conftest import tensor_types
from hypothesis import assume, given
from hypothesis import strategies as st
from max.dtype import DType
from max.graph import TensorType
from max.graph.ops import logical_not


@given(tensor_type=tensor_types(dtypes=st.just(DType.bool)))
def test_logical_not__same_type(graph_builder, tensor_type: TensorType):
    with graph_builder(input_types=[tensor_type]) as graph:
        x = graph.inputs[0]
        op = logical_not(x)
        assert op.type == tensor_type
        assert op.shape == x.shape

        op2 = ~x
        assert op2.type == tensor_type
        assert op2.shape == x.shape


@given(tensor_type=...)
def test_logical_not__invalid_dtype(graph_builder, tensor_type: TensorType):
    assume(tensor_type.dtype != DType.bool)
    with graph_builder(input_types=[tensor_type]) as graph:
        x = graph.inputs[0]
        with pytest.raises(ValueError):
            logical_not(x)

        with pytest.raises(ValueError):
            ~x
