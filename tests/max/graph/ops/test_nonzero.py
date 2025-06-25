# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
"""Test the max.graph Python bindings."""

import pytest
from hypothesis import given
from hypothesis import strategies as st
from max.dtype import DType
from max.graph import DeviceRef, Graph, TensorType, ops

shared_dtypes = st.shared(st.from_type(DType))


@given(input_type=...)
def test_nonzero(input_type: TensorType) -> None:
    with Graph("nonzero ", input_types=[input_type]) as graph:
        out = ops.nonzero(graph.inputs[0], "nonzero")
        assert out.dtype == DType.int64
        assert out.shape == ["nonzero", input_type.rank]
        graph.output(out)


@given(dtype=shared_dtypes)
def test_nonzero_scalar_error(dtype: DType) -> None:
    """Test that nonzero raises an error with a scalar input for any dtype."""
    scalar_type = TensorType(dtype, [], device=DeviceRef.CPU())
    with Graph("nonzero_scalar", input_types=[scalar_type]) as graph:
        with pytest.raises(ValueError, match="Scalar inputs not supported"):
            ops.nonzero(graph.inputs[0], "nonzero")
