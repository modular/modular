# ===----------------------------------------------------------------------=== #
# Copyright (c) 2025, Modular Inc. All rights reserved.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions:
# https://llvm.org/LICENSE.txt
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ===----------------------------------------------------------------------=== #
"""ops.split tests."""

import re

import pytest
from conftest import (
    new_axes,
    non_static_axes,
    shapes,
    static_dims,
    tensor_types,
)
from hypothesis import assume, given
from hypothesis import strategies as st
from max.dtype import DType
from max.graph import DeviceRef, Dim, StaticDim, TensorType, ops

shared_shapes = st.shared(shapes())


def with_dim(base_type: TensorType, dim: Dim, axis: int):
    # Create a tensor with a static dimension on the specified axis
    # If negative, update the axis for the new dim
    return TensorType(
        base_type.dtype,
        [
            *base_type.shape[:axis],
            dim,
            *base_type.shape[axis:],
        ],
        base_type.device,
    ), (axis if axis >= 0 else axis - 1)


@given(
    base_type=tensor_types(shapes=shared_shapes),
    split_sizes=shapes(dims=static_dims()),
    axis=new_axes(shared_shapes),
)
def test_split_valid_inputs(
    graph_builder,  # noqa: ANN001
    base_type: TensorType,
    split_sizes: list[StaticDim],
    axis: int,
) -> None:
    assume(-(2**63) <= sum(int(s) for s in split_sizes) < 2**63 - 1)
    input_type, axis = with_dim(base_type, sum(split_sizes, start=Dim(0)), axis)

    # Ensure axis is within bounds
    assert -input_type.rank <= axis < input_type.rank

    with graph_builder(input_types=[input_type]) as graph:
        output = ops.split(graph.inputs[0].tensor, split_sizes, axis)
        assert len(output) == len(split_sizes)

        for output, size in zip(output, split_sizes):  # noqa: B020
            expected_shape = list(input_type.shape)
            expected_shape[axis] = size
            assert output.shape == expected_shape
            assert output.dtype == input_type.dtype
            assert output.device == input_type.device


@given(
    base_type=tensor_types(shapes=shared_shapes),
    split_sizes=shapes(min_rank=1),
    dim=...,
    axis=new_axes(shared_shapes),
)
def test_split__non_static_split_dims(
    graph_builder,  # noqa: ANN001
    base_type: TensorType,
    split_sizes: list[Dim],
    dim: StaticDim,
    axis: int,
) -> None:
    assume(not all(isinstance(dim, StaticDim) for dim in split_sizes))
    input_type, axis = with_dim(base_type, dim, axis)
    with graph_builder(input_types=[input_type]) as graph:
        with pytest.raises(TypeError):
            ops.split(graph.inputs[0].tensor, split_sizes, axis)


@given(
    input_type=tensor_types(shapes=shared_shapes),
    split_sizes=shapes(dims=static_dims()),
    axis=...,
)
def test_split__invalid_axis(
    graph_builder,  # noqa: ANN001
    input_type: TensorType,
    split_sizes: list[StaticDim],
    axis: int,
) -> None:
    assume(not -input_type.rank <= axis < input_type.rank)
    with graph_builder(input_types=[input_type]) as graph:
        with pytest.raises(Exception):
            ops.split(graph.inputs[0].tensor, split_sizes, axis)


@given(
    base_type=tensor_types(shapes=shared_shapes),
    split_sizes=shapes(dims=static_dims()),
    split_dim=...,
    axis=new_axes(shared_shapes),
)
def test_split__splits_dont_sum_to_dim(
    graph_builder,  # noqa: ANN001
    base_type: TensorType,
    split_sizes: list[StaticDim],
    split_dim: StaticDim,
    axis: int,
) -> None:
    assume(-(2**63) <= sum(int(s) for s in split_sizes) < 2**63 - 1)
    assume(sum(split_sizes) != split_dim)
    input_type, axis = with_dim(base_type, split_dim, axis)
    with graph_builder(input_types=[input_type]) as graph:
        with pytest.raises(Exception):
            ops.split(graph.inputs[0].tensor, split_sizes, axis)


@given(
    input_type=tensor_types(shapes=shared_shapes),
    split_sizes=shapes(dims=static_dims()),
    axis=non_static_axes(shared_shapes),
)
def test_split__non_static_dim(
    graph_builder,  # noqa: ANN001
    input_type: TensorType,
    split_sizes: list[StaticDim],
    axis: int,
) -> None:
    assume(not isinstance(input_type.shape[axis], StaticDim))
    with graph_builder(input_types=[input_type]) as graph:
        with pytest.raises(Exception):
            ops.split(graph.inputs[0].tensor, split_sizes, axis)


@given(
    base_type=tensor_types(shapes=shared_shapes),
    split_sizes=st.lists(
        st.integers(min_value=-(2**63), max_value=2**63 - 1), min_size=1
    ),
    split_dim=...,
    axis=new_axes(shared_shapes),
)
def test_split__negative_split_sizes(
    graph_builder,  # noqa: ANN001
    base_type: TensorType,
    split_sizes: list[int],
    split_dim: StaticDim,
    axis: int,
) -> None:
    assume(any(dim < 0 for dim in split_sizes))
    assume(-(2**63) <= sum(split_sizes) < 2**63 - 1)

    # This is really dumb.
    #  - the target split_dim needs to be positive
    #  - we can't have any dims outside `long` range
    #  so we generate the split dim, and the splits including
    #  negative numbers, then make them sum to the target
    #  value by adding the difference. Since the range of possible
    #  dims is larger than the largest dim, we may split that dim
    #  in half.
    if sum(split_sizes) != split_dim:
        delta = int(split_dim) - sum(split_sizes)
        if delta > 2**63 - 1:  # ugh stupid longs
            d1 = delta // 2
            d2 = delta - d1
            split_sizes.extend((d1, d2))
        else:
            split_sizes.append(delta)

    assert sum(split_sizes) == split_dim
    input_type, axis = with_dim(base_type, split_dim, axis)
    with graph_builder(input_types=[input_type]) as graph:
        with pytest.raises(Exception):
            ops.split(graph.inputs[0].tensor, split_sizes, axis)


def test_invalid_split_full_error_message(graph_builder) -> None:  # noqa: ANN001
    input_shape = [15]
    split_sizes = [10, 6]
    axis = 0
    input_type = TensorType(DType.float32, input_shape, device=DeviceRef.CPU())
    with graph_builder(input_types=[input_type]) as graph:
        expected_message = "Split sizes must sum to dimension value; x.shape[axis]=Dim(15) != sum(split_sizes=[Dim(10), Dim(6)])"
        with pytest.raises(ValueError, match=re.escape(expected_message)):
            ops.split(graph.inputs[0].tensor, split_sizes, axis)


def test_invalid_axis_full_error_message(graph_builder) -> None:  # noqa: ANN001
    input_shape = [15]
    split_sizes = [10, 6]
    axis = 2
    input_type = TensorType(DType.float32, input_shape, device=DeviceRef.CPU())
    with graph_builder(input_types=[input_type]) as graph:
        expected_message = "Axis out of range axis=2, x.rank=1"
        with pytest.raises(IndexError, match=re.escape(expected_message)):
            ops.split(graph.inputs[0].tensor, split_sizes, axis)


def test_negative_split_size_full_error_message(graph_builder) -> None:  # noqa: ANN001
    input_shape = [4]
    split_sizes = [10, -6]
    axis = 0
    input_type = TensorType(DType.float32, input_shape, device=DeviceRef.CPU())
    with graph_builder(input_types=[input_type]) as graph:
        expected_message = (
            "Split sizes must be positive: split_sizes=[Dim(10), Dim(-6)]"
        )
        with pytest.raises(ValueError, match=re.escape(expected_message)):
            ops.split(graph.inputs[0].tensor, split_sizes, axis)
