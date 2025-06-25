# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
"""ops.reduction tests."""

import pytest
from conftest import axes, broadcastable_tensor_types, shapes, tensor_types
from hypothesis import example, given
from hypothesis import strategies as st
from max.dtype import DType
from max.graph import DeviceRef, TensorType, ops

shared_shapes = st.shared(shapes(min_rank=1))


@given(input_types=broadcastable_tensor_types(2))
def test_min__elementwise(graph_builder, input_types) -> None:
    with graph_builder(input_types=input_types) as graph:
        x, y = graph.inputs
        result = ops.min(x, y)
        expected = ops.elementwise.min(x, y)
        assert result.shape == expected.shape
        assert result.dtype == expected.dtype


@given(input_types=broadcastable_tensor_types(2))
def test_max__elementwise(graph_builder, input_types) -> None:
    with graph_builder(input_types=input_types) as graph:
        x, y = graph.inputs
        result = ops.max(x, y)
        expected = ops.elementwise.max(x, y)
        assert result.shape == expected.shape
        assert result.dtype == expected.dtype


@given(input_type=tensor_types(shapes=shared_shapes), axis=axes(shared_shapes))
def test_min__reduction(graph_builder, input_type, axis) -> None:
    with graph_builder(input_types=[input_type]) as graph:
        (x,) = graph.inputs
        result = ops.min(x, axis=axis)
        expected = ops.reduction.min(x, axis=axis)
        assert result.shape == expected.shape
        assert result.dtype == expected.dtype


@given(input_type=tensor_types(shapes=shared_shapes), axis=axes(shared_shapes))
def test_max__reduction(graph_builder, input_type, axis) -> None:
    with graph_builder(input_types=[input_type]) as graph:
        (x,) = graph.inputs
        result = ops.max(x, axis=axis)
        expected = ops.reduction.max(x, axis=axis)
        assert result.shape == expected.shape
        assert result.dtype == expected.dtype


@given(input_type=tensor_types(shapes=shared_shapes))
def test_min__reduction__no_axis(graph_builder, input_type) -> None:
    with graph_builder(input_types=[input_type]) as graph:
        (x,) = graph.inputs
        result = ops.min(x)
        expected = ops.reduction.min(x)
        assert result.shape == expected.shape
        assert result.dtype == expected.dtype


@given(input_type=tensor_types(shapes=shared_shapes))
def test_max__reduction__no_axis(graph_builder, input_type) -> None:
    with graph_builder(input_types=[input_type]) as graph:
        (x,) = graph.inputs
        result = ops.max(x)
        expected = ops.reduction.max(x)
        assert result.shape == expected.shape
        assert result.dtype == expected.dtype


@example(
    input_types=[
        TensorType(DType.int8, [1], DeviceRef.CPU()),
        TensorType(DType.int8, [1], DeviceRef.CPU()),
    ],
    axis=-1,
).via("ci flake")
@given(input_types=broadcastable_tensor_types(2), axis=st.integers())
def test_min_fail__y_and_axis_provided(
    graph_builder, input_types, axis
) -> None:
    with graph_builder(input_types=input_types) as graph:
        x, y = graph.inputs
        with pytest.raises(ValueError):
            result = ops.min(x, y, axis=axis)


@example(
    input_types=[
        TensorType(DType.int8, [1], DeviceRef.CPU()),
        TensorType(DType.int8, [1], DeviceRef.CPU()),
    ],
    axis=-1,
).via("ci flake")
@given(input_types=broadcastable_tensor_types(2), axis=st.integers())
def test_max_fail__y_and_axis_provided(
    graph_builder, input_types, axis
) -> None:
    with graph_builder(input_types=input_types) as graph:
        x, y = graph.inputs
        with pytest.raises(ValueError):
            result = ops.max(x, y, axis=axis)
