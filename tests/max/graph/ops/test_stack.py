# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
"""ops.stack tests."""

from conftest import new_axes, tensor_types
from hypothesis import given
from hypothesis import strategies as st
from max.graph import Graph, StaticDim, TensorType, ops


@given(
    type=st.shared(tensor_types(), key="type"),
    # Stack is currently "slow" for larger lists because of MLIR interop.
    # Stack currently fails on an empty list.
    stack_size=st.integers(min_value=1, max_value=20),
    axis=new_axes(st.shared(tensor_types(), key="type")),
)
def test_stack(type: TensorType, stack_size: int, axis: int):
    with Graph("stack", input_types=[type] * stack_size) as graph:
        out = ops.stack(graph.inputs, axis)
        target_shape = list(type.shape)
        target_shape.insert(
            axis + type.rank + 1 if axis < 0 else axis, StaticDim(stack_size)
        )
        assert out.shape == target_shape
        graph.output(out)
