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
from max.graph.ops import gelu

approximate = st.sampled_from(["none", "tanh", "quick"])


@given(
    tensor_type=tensor_types(dtypes=st.just(DType.float32)),
    approximate=approximate,
)
def test_gelu__same_type(tensor_type: TensorType, approximate: str):
    with Graph("gelu", input_types=[tensor_type]) as graph:
        (x,) = graph.inputs
        op = gelu(x, approximate)
        assert op.type == tensor_type
