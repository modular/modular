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
from max.graph.ops import is_inf


@given(
    tensor_type=tensor_types(
        dtypes=st.sampled_from([DType.float32, DType.bfloat16, DType.float16])
    ),
)
def test_is_inf_returns_bool(graph_builder, tensor_type: TensorType) -> None:
    with graph_builder(input_types=[tensor_type]) as graph:
        (x,) = graph.inputs
        op = is_inf(x)

        # is_inf should always return boolean tensors regardless of input dtype
        expected_type = TensorType(
            dtype=DType.bool, shape=tensor_type.shape, device=tensor_type.device
        )
        assert op.type == expected_type
        assert op.shape == x.shape
