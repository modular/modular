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

import pytest
from conftest import axes, shapes, tensor_types
from hypothesis import assume, given
from hypothesis import strategies as st
from max.graph import Graph, Shape, TensorType, ops

shared_squeeze_shape = st.shared(shapes(include_dims=[1]))


@given(
    input_type=tensor_types(shapes=shared_squeeze_shape),
    axis=axes(shared_squeeze_shape),
)
def test_squeeze(input_type: TensorType, axis: int) -> None:
    assume(input_type.shape[axis] == 1)
    with Graph("reshape", input_types=[input_type]) as graph:
        out = ops.squeeze(graph.inputs[0].tensor, axis)
        assert out.dtype == input_type.dtype
        expected_shape = Shape(input_type.shape)
        expected_shape.pop(axis)
        assert out.shape == expected_shape
        graph.output(out)


@given(
    input_type=tensor_types(shapes=shared_squeeze_shape),
    axis=axes(shared_squeeze_shape),
)
def test_squeeze__fails_on_non_static_1(
    input_type: TensorType, axis: int
) -> None:
    assume(input_type.shape[axis] != 1)
    with Graph("reshape", input_types=[input_type]) as graph:
        with pytest.raises(ValueError):
            out = ops.squeeze(graph.inputs[0].tensor, axis)


@given(input_type=tensor_types(), axis=st.integers())
def test_squeeze__fails_on_axis_out_of_bounds(
    input_type: TensorType, axis: int
) -> None:
    assume(not -input_type.rank <= axis < input_type.rank)
    with Graph("reshape", input_types=[input_type]) as graph:
        with pytest.raises(IndexError):
            out = ops.squeeze(graph.inputs[0].tensor, axis)
