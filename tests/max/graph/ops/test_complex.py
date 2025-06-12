# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
"""Tests for ops.complex."""

import pytest
from conftest import shapes, tensor_types
from hypothesis import assume, given
from max.graph import StaticDim, TensorType, ops


@given(base_type=tensor_types(shapes=shapes(min_rank=1)))
def test_as_interleaved_complex__valid(graph_builder, base_type: TensorType):
    """Test as_interleaved_complex with valid inputs."""
    *_, last = base_type.shape
    # Ensure last dimension is even and static
    assume(isinstance(last, StaticDim))
    assume(int(last) % 2 == 0)

    with graph_builder(input_types=[base_type]) as graph:
        out = ops.as_interleaved_complex(graph.inputs[0])
        # Output shape should be same except last dim is halved and new dim of 2 added
        expected_shape = base_type.shape[:-1] + [int(last) // 2, 2]
        assert out.type.shape == expected_shape
        graph.output(out)


@given(base_type=tensor_types(shapes=shapes(min_rank=1)))
def test_as_interleaved_complex__error__odd_last_dim(
    graph_builder, base_type: TensorType
):
    """Test that as_interleaved_complex raises an error when last dimension is odd."""
    *_, last = base_type.shape
    assume(isinstance(last, StaticDim))
    assume(int(last) % 2 != 0)

    with graph_builder(input_types=[base_type]) as graph:
        with pytest.raises(ValueError, match="must be divisible by 2"):
            ops.as_interleaved_complex(graph.inputs[0])


@given(base_type=tensor_types(shapes=shapes(min_rank=1)))
def test_as_interleaved_complex__error__dynamic_last_dim(
    graph_builder, base_type: TensorType
):
    """Test that as_interleaved_complex raises an error when last dimension is dynamic."""
    *_, last = base_type.shape
    assume(not isinstance(last, StaticDim))

    with graph_builder(input_types=[base_type]) as graph:
        with pytest.raises(TypeError, match="must be static"):
            ops.as_interleaved_complex(graph.inputs[0])
