# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
"""ops.outer tests."""

import pytest
from conftest import shapes, tensor_types
from hypothesis import assume, given
from hypothesis import strategies as st
from max.dtype import DType
from max.graph import Graph, TensorType, ops

shared_dtypes = st.shared(st.from_type(DType))
tensor_types_1d = tensor_types(
    dtypes=shared_dtypes,
    shapes=shapes(min_rank=1, max_rank=1),
)

tensor_types_nd = tensor_types(dtypes=shared_dtypes, shapes=shapes())


@given(lhs_type=tensor_types_1d, rhs_type=tensor_types_1d)
def test_outer_valid(lhs_type: TensorType, rhs_type: TensorType):
    with Graph("outer", input_types=[lhs_type, rhs_type]) as graph:
        out = ops.outer(graph.inputs[0], graph.inputs[1])
        assert out.shape == [lhs_type.shape[0], rhs_type.shape[0]]
        graph.output(out)


@given(lhs_type=tensor_types_nd, rhs_type=tensor_types_nd)
def test_outer_nd_tensors(lhs_type: TensorType, rhs_type: TensorType):
    assume(lhs_type.rank != 1 or rhs_type.rank != 1)

    with Graph("outer", input_types=[lhs_type, rhs_type]) as graph:
        with pytest.raises(ValueError):
            out = ops.outer(graph.inputs[0], graph.inputs[1])
