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
"""Test the max.graph Python bindings."""

import re

import pytest
from conftest import GraphBuilder, axes, shapes, tensor_types
from hypothesis import assume, given
from hypothesis import strategies as st
from max.dtype import DType
from max.graph import DeviceRef, Graph, TensorType

shared_types = st.shared(tensor_types())


@given(input_type=shared_types, a=axes(shared_types), b=axes(shared_types))
def test_transpose_output_shape(input_type: TensorType, a: int, b: int) -> None:
    assume(input_type.rank > 0)
    with Graph("transpose", input_types=[input_type]) as graph:
        out = graph.inputs[0].tensor.transpose(a, b)
        target_shape = list(input_type.shape)
        target_shape[a], target_shape[b] = target_shape[b], target_shape[a]
        assert out.shape == target_shape

        graph.output(out)


# Rank zero transpose is a little different, since it really is a no-op.
# The tests above filter it out, so we need to test it separately.
def test_transpose_input_with_rank_zero() -> None:
    with Graph(
        "transpose",
        input_types=[
            TensorType(shape=[], dtype=DType.float32, device=DeviceRef.CPU())
        ],
    ) as graph:
        out = graph.inputs[0].tensor.transpose(0, 0)
        assert out.shape == []
        graph.output(out)


def test_transpose_error_axis_1_out_of_bounds_input_with_rank_zero() -> None:
    with Graph(
        "transpose",
        input_types=[
            TensorType(shape=[], dtype=DType.float32, device=DeviceRef.CPU())
        ],
    ) as graph:
        with pytest.raises(
            IndexError,
            match=re.escape(
                "Axis axis_1 out of range (expected to be in range of [-1, 0], but got 1)"
            ),
        ):
            out = graph.inputs[0].tensor.transpose(1, 0)
            graph.output(out)


def test_transpose_error_axis_2_out_of_bounds_input_with_rank_zero() -> None:
    with Graph(
        "transpose",
        input_types=[
            TensorType(shape=[], dtype=DType.float32, device=DeviceRef.CPU())
        ],
    ) as graph:
        with pytest.raises(
            IndexError,
            match=re.escape(
                "Axis axis_2 out of range (expected to be in range of [-1, 0], but got 1)"
            ),
        ):
            out = graph.inputs[0].tensor.transpose(0, 1)
            graph.output(out)


def test_transpose_error_axis_1_out_of_bounds() -> None:
    with Graph(
        "transpose",
        input_types=[
            TensorType(
                shape=[2, 3], dtype=DType.float32, device=DeviceRef.CPU()
            )
        ],
    ) as graph:
        with pytest.raises(
            IndexError,
            match=re.escape(
                "Axis axis_1 out of range (expected to be in range of [-2, 1], but got 2)"
            ),
        ):
            out = graph.inputs[0].tensor.transpose(2, 0)
            graph.output(out)


def test_transpose_error_axis_2_out_of_bounds() -> None:
    with Graph(
        "transpose",
        input_types=[
            TensorType(
                shape=[2, 3], dtype=DType.float32, device=DeviceRef.CPU()
            )
        ],
    ) as graph:
        with pytest.raises(
            IndexError,
            match=re.escape(
                "Axis axis_2 out of range (expected to be in range of [-2, 1], but got 2)"
            ),
        ):
            out = graph.inputs[0].tensor.transpose(0, 2)
            graph.output(out)


shared_shapes_rank_gt_0 = st.shared(shapes(min_rank=1))


def invalid_axes(rank: int):
    return st.one_of(
        st.integers(max_value=-rank - 1), st.integers(min_value=rank)
    )


@given(
    base_type=tensor_types(shapes=shared_shapes_rank_gt_0),
    invalid_axis=shared_shapes_rank_gt_0.flatmap(
        lambda shape: invalid_axes(shape.rank)
    ),
    valid_axis=axes(shared_shapes_rank_gt_0),
)
def test_transpose_error_out_of_bounds_axis_rank_gt_zero(
    graph_builder: GraphBuilder,
    base_type: TensorType,
    valid_axis: int,
    invalid_axis: int,
) -> None:
    """Test that transpose raises an error when an axis is out of bounds."""
    with graph_builder(input_types=[base_type]) as graph:
        with pytest.raises(IndexError, match="out of range"):
            graph.inputs[0].tensor.transpose(valid_axis, invalid_axis)
        with pytest.raises(IndexError, match="out of range"):
            graph.inputs[0].tensor.transpose(invalid_axis, valid_axis)
        with pytest.raises(IndexError, match="out of range"):
            graph.inputs[0].tensor.transpose(invalid_axis, invalid_axis)


@given(
    base_type=tensor_types(shapes=shapes(max_rank=0, min_rank=0)),
    axis=st.integers().filter(lambda x: x != 0 and x != -1),
)
def test_transpose_error_out_of_bounds_rank_zero(
    graph_builder: GraphBuilder,
    base_type: TensorType,
    axis: int,
) -> None:
    """Test that transpose raises an error when an axis is out of bounds for rank 0."""
    # We follow PyTorch's definition for valid axes for transposition on rank 0 tensors
    assume(axis not in (0, -1))
    with graph_builder(input_types=[base_type]) as graph:
        with pytest.raises(IndexError, match="out of range"):
            graph.inputs[0].tensor.transpose(0, axis)
        with pytest.raises(IndexError, match="out of range"):
            graph.inputs[0].tensor.transpose(axis, 0)
        with pytest.raises(IndexError, match="out of range"):
            graph.inputs[0].tensor.transpose(axis, axis)
