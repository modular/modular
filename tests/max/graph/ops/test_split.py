# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
"""ops.split tests."""

import re

import pytest
from conftest import axes, shapes, static_dims, tensor_types
from hypothesis import assume, given
from hypothesis import strategies as st
from max.graph import Graph, TensorType, ops
from max.graph.type import DeviceRef, DType, StaticDim

shared_shapes = st.shared(shapes(min_rank=1))


@given(
    base_type=tensor_types(shapes=shapes(min_rank=1)),
    split_sizes=st.lists(static_dims(), min_size=1),
    axis=axes(shared_shapes),
)
def test_split_valid_inputs(
    graph_builder,
    base_type: TensorType,
    split_sizes: list[StaticDim],
    axis: int,
):
    # Ensure axis is within bounds
    assume(axis >= 0 and axis < base_type.rank)

    # Ensure split sizes are within valid bounds
    assume(sum(int(s) for s in split_sizes) < 2**63)

    # Create a tensor with a static dimension on the specified axis
    *broadcast, unused = base_type.shape
    input_type = TensorType(
        base_type.dtype,
        [
            *broadcast[:axis],
            StaticDim(sum(int(s) for s in split_sizes)),
            *broadcast[axis:],
        ],
        base_type.device,
    )

    with graph_builder(input_types=[input_type]) as graph:
        output = ops.split(graph.inputs[0].tensor, split_sizes, axis)
        assert len(output) == len(split_sizes)

        for output, size in zip(output, split_sizes):
            expected_shape = list(input_type.shape)
            expected_shape[axis] = size
            assert output.shape == expected_shape
            assert output.dtype == input_type.dtype


@given(
    base_type=tensor_types(shapes=shapes(min_rank=1)),
    split_sizes=st.lists(
        st.one_of(
            static_dims(min=1, max=10),  # Valid sizes
            st.integers(max_value=0),  # Invalid: non-positive sizes
        ),
        min_size=1,
        max_size=5,
    ),
    axis=st.one_of(
        st.integers(min_value=0),  # Valid axes
        st.integers(max_value=-1),  # Invalid: negative axes
        st.integers(min_value=100),  # Invalid: too large axes
    ),
)
def test_split_invalid_inputs(
    graph_builder,
    base_type: TensorType,
    split_sizes: list[StaticDim | int],
    axis: int,
):
    # Create a tensor with a static dimension on the specified axis
    *broadcast, unused = base_type.shape
    input_type = TensorType(
        base_type.dtype,
        [
            *broadcast[
                : axis % base_type.rank
            ],  # Use modulo to handle out-of-bounds axis
            StaticDim(
                20
            ),  # Use a fixed size that won't match most split_sizes sums
            *broadcast[axis % base_type.rank :],
        ],
        base_type.device,
    )

    with graph_builder(input_types=[input_type]) as graph:
        with pytest.raises(ValueError):
            ops.split(graph.inputs[0].tensor, split_sizes, axis)


def test_invalid_split_full_error_message():
    input_shape = [15]
    split_sizes = [10, 6]
    axis = 0
    with Graph(
        "split",
        input_types=[
            TensorType(DType.float32, input_shape, device=DeviceRef.CPU())
        ],
    ) as graph:
        with pytest.raises(
            ValueError,
            match=re.escape(
                "The split_sizes values should sum to 15 (input tensor's size at dimension 0), but got split_sizes=[10, 6]"
            ),
        ):
            output = ops.split(graph.inputs[0].tensor, split_sizes, axis)


def test_invalid_axis_full_error_message():
    input_shape = [15]
    split_sizes = [10, 6]
    axis = 2
    with Graph(
        "split",
        input_types=[
            TensorType(DType.float32, input_shape, device=DeviceRef.CPU())
        ],
    ) as graph:
        with pytest.raises(
            ValueError,
            match=re.escape(
                "Split axis must be within the input rank (1), got 2"
            ),
        ):
            _ = ops.split(graph.inputs[0].tensor, split_sizes, axis)


@given(
    base_type=tensor_types(shapes=shapes(min_rank=1)),
    axis=axes(shared_shapes),
)
def test_split_with_empty_split_sizes(
    base_type: TensorType,
    axis: int,
):
    split_sizes = []
    with Graph(
        "split",
        input_types=[base_type],
    ) as graph:
        output = ops.split(graph.inputs[0].tensor, split_sizes, axis)
        assert len(output) == 1
        assert output[0].shape == base_type.shape
