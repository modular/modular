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
"""Test TensorValue behaviors."""

import pytest
from conftest import axes, shapes, tensor_types
from hypothesis import assume, given
from hypothesis import strategies as st
from max.dtype import DType
from max.graph import Dim, Graph, Shape, TensorType

shared_shapes = st.shared(shapes())


@given(input_type=...)
def test_tensor_value__mean__no_axis(input_type: TensorType) -> None:
    with Graph("mean", input_types=[input_type]) as graph:
        input = graph.inputs[0].tensor
        out = input.mean()
        expected = Shape(input_type.shape)
        expected[-1] = Dim(1)
        assert out.shape == expected

        graph.output(out)


@given(input_type=tensor_types(shapes=shared_shapes), axis=axes(shared_shapes))
def test_tensor_value__mean__with_axis(
    input_type: TensorType, axis: int
) -> None:
    with Graph("mean", input_types=[input_type]) as graph:
        input = graph.inputs[0].tensor
        out = input.mean(axis=axis)
        expected = Shape(input_type.shape)
        expected[axis] = Dim(1)
        assert out.shape == expected

        graph.output(out)


@given(input_type=...)
def test_tensor_value__argmax__no_axis(input_type: TensorType) -> None:
    with Graph("argmax", input_types=[input_type]) as graph:
        input = graph.inputs[0].tensor
        out = input.argmax()
        expected = Shape(input_type.shape)
        expected[-1] = Dim(1)
        assert out.shape == expected

        graph.output(out)


@given(input_type=tensor_types(shapes=shared_shapes), axis=axes(shared_shapes))
def test_tensor_value__argmax__with_axis(
    input_type: TensorType, axis: int
) -> None:
    with Graph("argmax", input_types=[input_type]) as graph:
        input = graph.inputs[0].tensor
        out = input.argmax(axis=axis)
        expected = Shape(input_type.shape)
        expected[axis] = Dim(1)
        assert out.shape == expected

        graph.output(out)


@given(input_type=...)
def test_tensor_value__max__no_axis(input_type: TensorType) -> None:
    with Graph("max", input_types=[input_type]) as graph:
        input = graph.inputs[0].tensor
        out = input.max()
        expected = Shape(input_type.shape)
        expected[-1] = Dim(1)
        assert out.shape == expected

        graph.output(out)


@given(input_type=tensor_types(shapes=shared_shapes), axis=axes(shared_shapes))
def test_tensor_value__max__with_axis(
    input_type: TensorType, axis: int
) -> None:
    with Graph("max", input_types=[input_type]) as graph:
        input = graph.inputs[0].tensor
        out = input.max(axis=axis)
        expected = Shape(input_type.shape)
        expected[axis] = Dim(1)
        assert out.shape == expected

        graph.output(out)


@given(input_type=...)
def test_tensor_value__min__no_axis(input_type: TensorType) -> None:
    with Graph("min", input_types=[input_type]) as graph:
        input = graph.inputs[0].tensor
        out = input.min()
        expected = Shape(input_type.shape)
        expected[-1] = Dim(1)
        assert out.shape == expected

        graph.output(out)


@given(input_type=tensor_types(shapes=shared_shapes), axis=axes(shared_shapes))
def test_tensor_value__min__with_axis(
    input_type: TensorType, axis: int
) -> None:
    with Graph("min", input_types=[input_type]) as graph:
        input = graph.inputs[0].tensor
        out = input.min(axis=axis)
        expected = Shape(input_type.shape)
        expected[axis] = Dim(1)
        assert out.shape == expected

        graph.output(out)


@given(input_type=...)
def test_tensor_value__stdev__no_axis(input_type: TensorType) -> None:
    # Variance not defined for bools
    assume(input_type.dtype is not DType.bool)
    with Graph("stdev", input_types=[input_type]) as graph:
        input = graph.inputs[0].tensor
        out = input.stdev()
        expected = Shape(input_type.shape)
        expected[-1] = Dim(1)
        assert out.shape == expected

        graph.output(out)


@given(input_type=tensor_types(shapes=shared_shapes), axis=axes(shared_shapes))
def test_tensor_value__stdev__with_axis(
    input_type: TensorType, axis: int
) -> None:
    # Variance not defined for bools
    assume(input_type.dtype is not DType.bool)
    with Graph("stdev", input_types=[input_type]) as graph:
        input = graph.inputs[0].tensor
        out = input.stdev(axis=axis)
        expected = Shape(input_type.shape)
        expected[axis] = Dim(1)
        assert out.shape == expected

        graph.output(out)


@given(input_type=...)
def test_tensor_value__var__no_axis(input_type: TensorType) -> None:
    # Variance not defined for bools
    assume(input_type.dtype is not DType.bool)
    with Graph("var", input_types=[input_type]) as graph:
        input = graph.inputs[0].tensor
        out = input.var()
        expected = Shape(input_type.shape)
        expected[-1] = Dim(1)
        assert out.shape == expected

        graph.output(out)


@given(input_type=tensor_types(shapes=shared_shapes), axis=axes(shared_shapes))
def test_tensor_value__var__with_axis(
    input_type: TensorType, axis: int
) -> None:
    # Variance not defined for bools
    assume(input_type.dtype is not DType.bool)
    with Graph("var", input_types=[input_type]) as graph:
        input = graph.inputs[0].tensor
        out = input.var(axis=axis)
        expected = Shape(input_type.shape)
        expected[axis] = Dim(1)
        assert out.shape == expected

        graph.output(out)


@given(input_type=tensor_types(dtypes=st.just(DType.bool)))
def test_tensor_value__stdev__bool(input_type: TensorType) -> None:
    with Graph("stdev", input_types=[input_type]) as graph:
        input = graph.inputs[0].tensor
        with pytest.raises(TypeError):
            _ = input.stdev()


@given(input_type=tensor_types(dtypes=st.just(DType.bool)))
def test_tensor_value__var__bool(input_type: TensorType) -> None:
    with Graph("stdev", input_types=[input_type]) as graph:
        input = graph.inputs[0].tensor
        with pytest.raises(TypeError):
            _ = input.var()
