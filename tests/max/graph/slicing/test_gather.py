# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
"""ops.gather tests."""

from conftest import axes, dims, tensor_types
from hypothesis import assume, given
from hypothesis import strategies as st
from max.dtype import DType
from max.graph import Graph, TensorType, ops

# gather not meaningful for scalar inputs
input_types = tensor_types(shapes=st.lists(dims, min_size=1))


@given(
    input_type=st.shared(input_types, key="input"),
    indices_type=tensor_types(dtypes=st.just(DType.int64)),
    axis=axes(st.shared(input_types, key="input")),
)
def test_gather(input_type: TensorType, indices_type: TensorType, axis: int):
    assume(indices_type.rank > 0)
    with Graph("gather", input_types=[input_type, indices_type]) as graph:
        input, indices = graph.inputs
        out = ops.gather(input, indices, axis)
        target_shape = [
            *input.shape[:axis],
            *indices.shape,
            *input.shape[axis + 1 :],
        ]
        assert out.tensor_type == TensorType(
            input.tensor_type.dtype, target_shape
        )
        graph.output(out)
