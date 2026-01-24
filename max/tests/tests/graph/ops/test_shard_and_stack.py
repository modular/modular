# ===----------------------------------------------------------------------=== #
# Copyright (c) 2026, Modular Inc. All rights reserved.
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
"""ops.shard_and_stack tests."""

import re

import pytest
from conftest import GraphBuilder, axes, tensor_types
from hypothesis import assume, given
from hypothesis import strategies as st
from max.dtype import DType
from max.graph import DeviceRef, Dim, StaticDim, TensorType, ops

shared_types = st.shared(tensor_types())


@given(
    input_type=shared_types,
    num_inputs=st.integers(min_value=1, max_value=1024),
    tp_degree=st.integers(min_value=1, max_value=32),
    axis=axes(shared_types),
)
def test_shard_and_stack_valid_inputs(
    graph_builder: GraphBuilder,
    input_type: TensorType,
    num_inputs: int,
    tp_degree: int,
    axis: int,
) -> None:
    axis_dim = input_type.shape[axis]

    if isinstance(axis_dim, StaticDim):
        assume(int(axis_dim) % tp_degree == 0)

    input_types = [input_type] * num_inputs
    devices = [input_type.device] * tp_degree

    with graph_builder(input_types=input_types) as graph:
        inputs = [inp.tensor for inp in graph.inputs]
        outputs = ops.shard_and_stack(inputs, devices, axis)

        # Should return tp_degree outputs
        assert len(outputs) == tp_degree

        # Calculate expected output shape
        chunk_size = input_type.shape[axis] // tp_degree
        expected_shape = [num_inputs, *input_type.shape]

        if axis < 0:
            axis += input_type.rank

        expected_shape[axis + 1] = chunk_size  # axis+1 due to new dim 0

        # # Check each output has the expected shape
        for output in outputs:
            assert output.shape == expected_shape
            assert output.dtype == input_types[0].dtype
            assert output.device == input_types[0].device


def test_shard_and_stack__empty_devices(
    graph_builder: GraphBuilder,
) -> None:
    input_type = TensorType(DType.float32, [10, 4], device=DeviceRef.CPU())
    with graph_builder(input_types=[input_type]) as graph:
        with pytest.raises(ValueError):
            ops.shard_and_stack([graph.inputs[0].tensor], devices=[])


@given(
    tp_degree=st.integers(min_value=2, max_value=4),
)
def test_shard_and_stack__non_divisible_dim(
    graph_builder: GraphBuilder,
    tp_degree: int,
) -> None:
    # Create a dimension that is NOT divisible by tp_degree
    # Use a static dimension that's definitely not divisible
    non_divisible_size = tp_degree + 1

    input_type = TensorType(
        DType.float32,
        [Dim(non_divisible_size), Dim(4)],
        DeviceRef.CPU(),
    )
    devices = [DeviceRef.CPU()] * tp_degree

    with graph_builder(input_types=[input_type]) as graph:
        with pytest.raises(ValueError):
            ops.shard_and_stack([graph.inputs[0].tensor], devices)


def test_shard_and_stack__shape_mismatch(
    graph_builder: GraphBuilder,
) -> None:
    # Create two tensors with different shapes
    input_type1 = TensorType(DType.float32, [Dim(10), Dim(4)], DeviceRef.CPU())
    input_type2 = TensorType(DType.float32, [Dim(8), Dim(4)], DeviceRef.CPU())
    devices = [DeviceRef.CPU(), DeviceRef.CPU()]

    with graph_builder(input_types=[input_type1, input_type2]) as graph:
        with pytest.raises(ValueError):
            ops.shard_and_stack(
                [graph.inputs[0].tensor, graph.inputs[1].tensor], devices
            )


def test_shard_and_stack_empty_inputs(graph_builder: GraphBuilder) -> None:
    devices = [DeviceRef.CPU(), DeviceRef.CPU()]
    with graph_builder(input_types=[]) as graph:
        expected_message = "Must provide at least one input tensor."
        with pytest.raises(ValueError, match=re.escape(expected_message)):
            ops.shard_and_stack([], devices)


def test_shard_and_stack_empty_devices_full_error_message(
    graph_builder: GraphBuilder,
) -> None:
    input_type = TensorType(DType.float32, [10, 4], device=DeviceRef.CPU())
    with graph_builder(input_types=[input_type]) as graph:
        expected_message = "Must provide at least one device."
        with pytest.raises(ValueError, match=re.escape(expected_message)):
            ops.shard_and_stack([graph.inputs[0].tensor], devices=[])


def test_shard_and_stack_non_divisible_full_error_message(
    graph_builder: GraphBuilder,
) -> None:
    input_type = TensorType(DType.float32, [10, 4], device=DeviceRef.CPU())
    devices = [DeviceRef.CPU()] * 3
    with graph_builder(input_types=[input_type]) as graph:
        expected_message = (
            "Dimension size along axis 0 (10) must be evenly "
            "divisible by number of devices (3)"
        )
        with pytest.raises(ValueError, match=re.escape(expected_message)):
            ops.shard_and_stack([graph.inputs[0].tensor], devices)


def test_shard_and_stack_shape_mismatch_full_error_message(
    graph_builder: GraphBuilder,
) -> None:
    input_type1 = TensorType(DType.float32, [10, 4], device=DeviceRef.CPU())
    input_type2 = TensorType(DType.float32, [8, 4], device=DeviceRef.CPU())
    devices = [DeviceRef.CPU(), DeviceRef.CPU()]
    with graph_builder(input_types=[input_type1, input_type2]) as graph:
        expected_message = (
            "All input tensors must have the same shape. "
            "Input 0 has shape [Dim(10), Dim(4)], but input 1 has shape [Dim(8), Dim(4)]"
        )
        with pytest.raises(ValueError, match=re.escape(expected_message)):
            ops.shard_and_stack(
                [graph.inputs[0].tensor, graph.inputs[1].tensor], devices
            )


def test_shard_and_stack__invalid_axis(graph_builder: GraphBuilder) -> None:
    """Test shard_and_stack with out-of-bounds axis."""
    input_type = TensorType(DType.float32, [10, 4, 8], device=DeviceRef.CPU())
    devices = [DeviceRef.CPU(), DeviceRef.CPU()]

    with graph_builder(input_types=[input_type]) as graph:
        # Test axis >= rank
        expected_message = "Axis must be in range [-3, 3), got 3"
        with pytest.raises(IndexError, match=re.escape(expected_message)):
            ops.shard_and_stack([graph.inputs[0].tensor], devices, axis=3)

        # Test axis < -rank
        expected_message = "Axis must be in range [-3, 3), got -4"
        with pytest.raises(IndexError, match=re.escape(expected_message)):
            ops.shard_and_stack([graph.inputs[0].tensor], devices, axis=-4)
