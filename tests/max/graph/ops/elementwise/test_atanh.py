# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
"""test the max.graph python bindings."""

from conftest import tensor_types
from hypothesis import given
from hypothesis import strategies as st
from max.dtype import DType
from max.graph import TensorType
from max.graph.ops import atanh


@given(
    tensor_type=tensor_types(dtypes=st.just(DType.float32)),
)
def test_atanh_same_type(graph_builder, tensor_type: TensorType):
    with graph_builder(input_types=[tensor_type]) as graph:
        (x,) = graph.inputs
        op = atanh(x)
        assert op.type == tensor_type
