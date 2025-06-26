# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
"""Tests for ops.hann_window."""

import pytest
from hypothesis import given
from hypothesis import strategies as st
from max.dtype import DType
from max.graph import DeviceRef, Shape, StaticDim, TensorType, ops


@given(
    window_length=st.integers(min_value=0, max_value=1000),
    periodic=st.booleans(),
    dtype=st.sampled_from([DType.float32, DType.bfloat16, DType.float64]),
)
def test_hann_window_valid_inputs(
    graph_builder, window_length: int, periodic: bool, dtype: DType
):
    """Test hann_window with valid inputs using property-based testing."""
    with graph_builder(input_types=[]) as graph:
        result = ops.hann_window(
            window_length=window_length,
            device=DeviceRef.CPU(),
            periodic=periodic,
            dtype=dtype,
        )

        # Check output shape
        expected_shape = Shape([StaticDim(window_length)])
        assert result.shape == expected_shape

        # Check output type
        expected_type = TensorType(dtype, expected_shape, DeviceRef.CPU())
        assert result.type == expected_type

        graph.output(result)


@pytest.mark.parametrize("window_length", [0, 1, 2, 5, 10, 100])
@pytest.mark.parametrize("periodic", [True, False])
@pytest.mark.parametrize("dtype", [DType.float32, DType.bfloat16])
def test_hann_window_specific_cases(
    graph_builder, window_length: int, periodic: bool, dtype: DType
):
    """Test hann_window with specific parameter combinations."""
    with graph_builder(input_types=[]) as graph:
        result = ops.hann_window(
            window_length=window_length,
            device=DeviceRef.CPU(),
            periodic=periodic,
            dtype=dtype,
        )

        # Verify shape
        assert result.shape == Shape([StaticDim(window_length)])

        # Verify type
        assert result.type.dtype == dtype
        assert result.type.device == DeviceRef.CPU()

        graph.output(result)


@given(window_length=st.integers(max_value=-1))
def test_hann_window_negative_length_raises(graph_builder, window_length: int):
    """Test that negative window_length raises ValueError."""
    with pytest.raises(ValueError):
        with graph_builder(input_types=[]) as graph:
            ops.hann_window(
                window_length=window_length,
                device=DeviceRef.CPU(),
                periodic=True,
                dtype=DType.float32,
            )


def test_hann_window_negative_length_error_message(graph_builder):
    """Test specific error message for negative window_length."""
    with pytest.raises(ValueError, match="window_length must be non-negative"):
        with graph_builder(input_types=[]) as graph:
            ops.hann_window(
                window_length=-1,
                device=DeviceRef.CPU(),
                periodic=True,
                dtype=DType.float32,
            )


def test_hann_window_non_integer_length_raises(graph_builder):
    """Test that non-integer window_length raises TypeError."""
    with pytest.raises(TypeError):
        with graph_builder(input_types=[]) as graph:
            ops.hann_window(
                window_length=5.5,  # float instead of int
                device=DeviceRef.CPU(),
                periodic=True,
                dtype=DType.float32,
            )


def test_hann_window_non_integer_length_error_message(graph_builder):
    """Test specific error message for non-integer window_length."""
    with pytest.raises(
        TypeError, match="window_length must be an integer, got float"
    ):
        with graph_builder(input_types=[]) as graph:
            ops.hann_window(
                window_length=5.5,
                device=DeviceRef.CPU(),
                periodic=True,
                dtype=DType.float32,
            )


@pytest.mark.parametrize("device", [DeviceRef.CPU(), DeviceRef.GPU()])
def test_hann_window_different_devices(graph_builder, device: DeviceRef):
    """Test hann_window works with different devices."""
    with graph_builder(input_types=[]) as graph:
        result = ops.hann_window(
            window_length=10,
            device=device,
            periodic=True,
            dtype=DType.float32,
        )

        assert result.type.device == device
        graph.output(result)


def test_hann_window_edge_case_zero_length(graph_builder):
    """Test hann_window with zero length returns empty tensor."""
    with graph_builder(input_types=[]) as graph:
        result = ops.hann_window(
            window_length=0,
            device=DeviceRef.CPU(),
            periodic=True,
            dtype=DType.float32,
        )

        assert result.shape == Shape([StaticDim(0)])
        graph.output(result)


def test_hann_window_edge_case_length_one(graph_builder):
    """Test hann_window with length one returns single value."""
    with graph_builder(input_types=[]) as graph:
        result = ops.hann_window(
            window_length=1,
            device=DeviceRef.CPU(),
            periodic=True,
            dtype=DType.float32,
        )

        assert result.shape == Shape([StaticDim(1)])
        graph.output(result)


@pytest.mark.parametrize(
    "dtype", [DType.float32, DType.float64, DType.bfloat16, DType.float16]
)
def test_hann_window_supported_dtypes(graph_builder, dtype: DType):
    """Test hann_window works with various floating point dtypes."""
    with graph_builder(input_types=[]) as graph:
        result = ops.hann_window(
            window_length=5,
            device=DeviceRef.CPU(),
            periodic=True,
            dtype=dtype,
        )

        assert result.type.dtype == dtype
        graph.output(result)


@given(
    window_length=st.integers(min_value=2, max_value=100),
    periodic=st.booleans(),
)
def test_hann_window_graph_invariants(
    graph_builder, window_length: int, periodic: bool
):
    """Test that hann_window maintains graph construction invariants."""
    with graph_builder(input_types=[]) as graph:
        result = ops.hann_window(
            window_length=window_length,
            device=DeviceRef.CPU(),
            periodic=periodic,
            dtype=DType.float32,
        )

        # The operation should create a valid tensor
        assert hasattr(result, "shape")
        assert hasattr(result, "type")

        # The tensor should have the expected rank
        assert result.shape.rank == 1

        # The tensor should have static dimensions
        assert all(isinstance(dim, StaticDim) for dim in result.shape)

        graph.output(result)


def test_hann_window_large_window_length(graph_builder):
    """Test hann_window with large window length."""
    large_length = 10000
    with graph_builder(input_types=[]) as graph:
        result = ops.hann_window(
            window_length=large_length,
            device=DeviceRef.CPU(),
            periodic=False,
            dtype=DType.float32,
        )

        assert result.shape == Shape([StaticDim(large_length)])
        graph.output(result)
