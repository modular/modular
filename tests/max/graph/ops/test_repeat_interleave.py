# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
"""ops.repeat_interleave tests."""

import random

from conftest import tensor_types
from hypothesis import given
from hypothesis import strategies as st
from max.graph import Graph, TensorType, ops


@given(
    type=st.shared(tensor_types(), key="type"),
    # Stack is currently "slow" for larger lists because of MLIR interop.
    # Stack currently fails on an empty list.
    repeats=st.integers(min_value=1, max_value=20),
)
def test_repeat_interleave(type: TensorType, repeats: int):
    axis = random.randint(0, type.rank - 1)
    with Graph("repeat_interleave", input_types=[type]) as graph:
        out = ops.repeat_interleave(graph.inputs[0], repeats, axis)
        target_shape = list(type.shape)
        target_shape[axis] *= repeats
        assert out.shape == target_shape
        graph.output(out)
