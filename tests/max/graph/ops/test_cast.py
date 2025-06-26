# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
"""Tests for ops.cast."""

from hypothesis import given
from max.dtype import DType
from max.graph import (
    TensorType,
    ops,
)


@given(base_type=..., target_dtype=...)
def test_cast__tensor(
    graph_builder, base_type: TensorType, target_dtype: DType
):
    """Test that cast correctly converts tensor values between different data types."""
    expected_type = base_type.cast(target_dtype)
    with graph_builder(input_types=[base_type]) as graph:
        out = ops.cast(graph.inputs[0], target_dtype)
        assert out.type == expected_type
        graph.output(out)
