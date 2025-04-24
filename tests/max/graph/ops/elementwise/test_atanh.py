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
from max.graph import Graph, TensorType
from max.graph.ops import atanh


@given(
    tensor_type=tensor_types(dtypes=st.just(DType.float32)),
)
def test_atanh_same_type(tensor_type: TensorType):
    with Graph("atanh", input_types=[tensor_type]) as graph:
        (x,) = graph.inputs
        op = atanh(x)
        assert op.type == tensor_type
