# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
"""Tests for the scatter op."""

import re

import pytest
from conftest import axes, dims, tensor_types
from hypothesis import assume, given
from hypothesis import strategies as st
from max.dtype import DType
from max.graph import DeviceRef, Graph, TensorType, ops

input_types = tensor_types(shapes=st.lists(dims, min_size=1))


@given(
    input_type=st.shared(input_types, key="input"),
    indices_type=tensor_types(
        dtypes=st.sampled_from([DType.int32, DType.int64])
    ),
    axis=axes(st.shared(input_types, key="input")),
)
def test_scatter(
    input_type: TensorType,
    indices_type: TensorType,
    axis: int,
):
    """Tests that the scatter op preserves shape and dtype."""
    updates_type = input_type

    with Graph(
        "scatter", input_types=[input_type, updates_type, indices_type]
    ) as graph:
        input_tensor, updates, indices = graph.inputs
        scatter_result = ops.scatter(input_tensor, updates, indices, axis=axis)
        graph.output(scatter_result)

        assert scatter_result.type.dtype == input_tensor.type.dtype, (
            "DType should be preserved."
        )
        assert scatter_result.type.shape == input_tensor.type.shape, (
            "Shape should be preserved."
        )


@given(
    input_type=st.shared(input_types, key="input"),
    updates_type=st.shared(input_types, key="updates"),
    indices_type=tensor_types(
        dtypes=st.sampled_from([DType.int32, DType.int64])
    ),
    axis=axes(st.shared(input_types, key="input")),
)
def test_scatter_input_and_updates_different_dtypes(
    input_type: TensorType,
    updates_type: TensorType,
    indices_type: TensorType,
    axis: int,
):
    """Tests that the scatter op raises an error with different input and updates dtypes."""
    assume(input_type.dtype != updates_type.dtype)

    with Graph(
        "scatter_input_and_updates_different_dtypes",
        input_types=[input_type, updates_type, indices_type],
    ) as graph:
        input_tensor, updates, indices = graph.inputs
        with pytest.raises(ValueError):
            ops.scatter(input_tensor, updates, indices, axis=axis)


def test_scatter_input_and_updates_different_dtypes_specific_error_message():
    """Test that the scatter op raises an error with different input and updates dtypes."""
    with Graph(
        "scatter_input_and_updates_different_dtypes",
        input_types=[
            TensorType(DType.float32, [1, 2, 3], device=DeviceRef.CPU()),
            TensorType(DType.float64, [1, 2, 3], device=DeviceRef.CPU()),
            TensorType(DType.int32, [1, 2, 3], device=DeviceRef.CPU()),
        ],
    ) as graph:
        with pytest.raises(
            ValueError,
            match=re.escape(
                "The input dtype 'DType.float32' and updates dtype 'DType.float64' must match."
            ),
        ):
            ops.scatter(graph.inputs[0], graph.inputs[1], graph.inputs[2])


@given(
    input_type=st.shared(input_types, key="input"),
    indices_type=tensor_types(
        dtypes=st.sampled_from([DType.float32, DType.float64])
    ),
    axis=axes(st.shared(input_types, key="input")),
)
def test_scatter_invalid_indices_type(
    input_type: TensorType,
    indices_type: TensorType,
    axis: int,
):
    """Test that the scatter op raises an error with an invalid indices type."""
    updates_type = input_type

    with Graph(
        "scatter_with_invalid_indices_type",
        input_types=[input_type, updates_type, indices_type],
    ) as graph:
        input_tensor, updates, indices = graph.inputs
        with pytest.raises(ValueError):
            ops.scatter(input_tensor, updates, indices, axis=axis)


def test_scatter_invalid_indices_type_specific_error_message():
    """Test that the scatter op raises an error with an invalid indices type."""
    with Graph(
        "scatter_with_invalid_indices_type",
        input_types=[
            TensorType(DType.float32, [1, 2, 3], device=DeviceRef.CPU()),
            TensorType(DType.float32, [1, 2, 3], device=DeviceRef.CPU()),
            TensorType(DType.float32, [1, 2, 3], device=DeviceRef.CPU()),
        ],
    ) as graph:
        with pytest.raises(
            ValueError,
            match=re.escape(
                "Invalid indices dtype: 'DType.float32'. Indices must be of type int32 or int64."
            ),
        ):
            ops.scatter(graph.inputs[0], graph.inputs[1], graph.inputs[2])


@given(
    input_type=st.shared(input_types, key="input"),
    indices_type=tensor_types(
        dtypes=st.sampled_from([DType.int32, DType.int64])
    ),
    axis=st.integers(),
)
def test_scatter_invalid_axis(
    input_type: TensorType,
    indices_type: TensorType,
    axis: int,
):
    """Test that the scatter op raises an error with an invalid axis."""
    updates_type = input_type

    assume(abs(axis) > input_type.rank)

    with Graph(
        "scatter_with_invalid_axis",
        input_types=[input_type, updates_type, indices_type],
    ) as graph:
        input_tensor, updates, indices = graph.inputs
        with pytest.raises(ValueError):
            ops.scatter(input_tensor, updates, indices, axis=axis)


def test_scatter_invalid_axis_specific_error_message():
    """Test that the scatter op raises an error with an invalid axis."""
    with Graph(
        "scatter_with_invalid_axis",
        input_types=[
            TensorType(DType.float32, [1, 2, 3], device=DeviceRef.CPU()),
            TensorType(DType.float32, [1, 2, 3], device=DeviceRef.CPU()),
            TensorType(DType.int32, [1, 2, 3], device=DeviceRef.CPU()),
        ],
    ) as graph:
        with pytest.raises(
            ValueError,
            match=re.escape(
                "Invalid axis value 100. Axis must be in range [-3, 2]"
            ),
        ):
            ops.scatter(
                graph.inputs[0], graph.inputs[1], graph.inputs[2], axis=100
            )
