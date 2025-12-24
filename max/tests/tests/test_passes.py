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
"""Tests for `max.experimental._passes`."""

from collections.abc import Iterator

import pytest
from max.driver import CPU
from max.dtype import DType
from max.engine.api import InferenceSession
from max.experimental import _passes, tensor
from max.graph import Graph, TensorType


@pytest.fixture
def session() -> Iterator[InferenceSession]:
    yield tensor._session()


def test_add_input(session: InferenceSession) -> None:
    type = TensorType(DType.float32, ["a", "b"], CPU())
    with Graph("test_add_input", input_types=[]) as graph:
        graph.output()

    # Basic checks: the updated graph has the right input,
    #  still compiles, and the resulting model
    _passes.add_input(graph, type)
    assert len(graph.inputs) == 1
    assert graph.inputs[0].type == type
    model = session.load(graph)
    assert len(model.input_metadata) == 1


def test_remove_unused_arguments(session: InferenceSession) -> None:
    type_a = TensorType(DType.float32, ["a"], CPU())
    type_b = TensorType(DType.float32, ["b"], CPU())
    with Graph("test_add_input", input_types=[type_a, type_b]) as graph:
        _, b = graph.inputs
        graph.output(b)

    # Basic checks: the updated graph has the right input,
    #  still compiles, and the resulting model
    _passes.remove_unused_arguments(graph)
    assert len(graph.inputs) == 1
    assert graph.inputs[0].type == type_b

    model = session.load(graph)
    assert len(model.input_metadata) == 1


def test_remove_static_shape_info() -> None:
    """Test removing static shape info replaces static dims with symbolic ones."""
    # Create a graph with static shapes
    type_static = TensorType(DType.float32, [3, 4], CPU())
    with Graph(
        "test_remove_static_shape_info", input_types=[type_static]
    ) as graph:
        x = graph.inputs[0]
        graph.output(x)

    original_num_inputs = len(graph.inputs)

    # Apply the pass
    parameters = _passes.remove_static_shape_info(graph)

    # Static dimensions 3 and 4 should be mapped to symbolic dims
    assert 3 in parameters
    assert 4 in parameters
    assert len(parameters) == 2

    # A new input should be added (bool tensor carrying shape parameters)
    assert len(graph.inputs) == original_num_inputs + 1

    # The new input should be a bool tensor
    new_input_type = graph.inputs[-1].type
    assert isinstance(new_input_type, TensorType)
    assert new_input_type.dtype == DType.bool


def test_remove_static_shape_info_repeated_dims() -> None:
    """Test that repeated static dims map to the same symbolic dim."""
    # Create a graph where the same static dimension appears multiple times
    type_static = TensorType(DType.float32, [5, 5], CPU())
    with Graph("test_repeated_dims", input_types=[type_static]) as graph:
        x = graph.inputs[0]
        graph.output(x)

    parameters = _passes.remove_static_shape_info(graph)

    # Only one unique static value (5), so only one parameter entry
    assert 5 in parameters
    assert len(parameters) == 1

    # A new input should be added
    assert len(graph.inputs) == 2


def test_remove_static_shape_info_multiple_inputs() -> None:
    """Test removing static shape info from a graph with multiple inputs."""
    type_a = TensorType(DType.float32, [2, 3], CPU())
    type_b = TensorType(DType.float32, [3, 4], CPU())
    with Graph("test_multiple_inputs", input_types=[type_a, type_b]) as graph:
        a, b = graph.inputs
        graph.output(a, b)

    parameters = _passes.remove_static_shape_info(graph)

    # Static dimensions 2, 3, and 4 should be mapped to symbolic dims
    # Note: dimension 3 appears in both inputs but should map to same symbolic dim
    assert 2 in parameters
    assert 3 in parameters
    assert 4 in parameters
    assert len(parameters) == 3

    # A new input should be added (original 2 + 1 new)
    assert len(graph.inputs) == 3
