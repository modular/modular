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
from conftest import tensor_types
from hypothesis import assume, given
from hypothesis import strategies as st
from max.dtype import DType
from max.graph import BufferType, DeviceRef, Graph, Shape, TensorType, ops
from max.graph.value import TensorValue, _is_strong_tensor_value_like


@given(input_type=...)
def test_tensor_value__T(input_type: TensorType) -> None:
    assume(input_type.rank >= 2)
    with Graph("transpose", input_types=[input_type]) as graph:
        out = graph.inputs[0].tensor.T
        expected = Shape(input_type.shape)
        expected[-1], expected[-2] = expected[-2], expected[-1]
        assert out.shape == expected

        graph.output(out)


@given(input_type=...)
def test_buffer__not_tensorvalue(input_type: BufferType) -> None:
    assert not _is_strong_tensor_value_like(input_type)
    with Graph("buffer", input_types=[input_type]) as graph:
        with pytest.raises(TypeError):
            TensorValue(input_type)  # type: ignore


@given(tensor_type=tensor_types(dtypes=st.just(DType.bool)))
def test_tensor_value__operator_logical_and(tensor_type: TensorType) -> None:
    with Graph("and", input_types=[tensor_type]) as graph:
        x = graph.inputs[0].tensor
        assert (x & x).type == ops.logical_and(x, x).type


@given(tensor_type=tensor_types(dtypes=st.just(DType.bool)))
def test_tensor_value__operator_logical_or(tensor_type: TensorType) -> None:
    with Graph("or", input_types=[tensor_type]) as graph:
        x = graph.inputs[0].tensor
        assert (x | x).type == ops.logical_or(x, x).type


@given(tensor_type=tensor_types(dtypes=st.just(DType.bool)))
def test_tensor_value__operator_logical_xor(tensor_type: TensorType) -> None:
    with Graph("xor", input_types=[tensor_type]) as graph:
        x = graph.inputs[0].tensor
        assert (x ^ x).type == ops.logical_xor(x, x).type


def test_special_methods_error() -> None:
    """Test that we disallow special methods that would hang."""
    with Graph(
        "special_methods",
        input_types=[
            TensorType(DType.float32, shape=["a", "b"], device=DeviceRef.CPU())
        ],
    ) as graph:
        (x,) = graph.inputs
        with pytest.raises(TypeError, match="is not a container"):
            assert 1 in x  # type: ignore

        with pytest.raises(TypeError, match="is not iterable"):

            def func(*args) -> None:
                pass

            func(*x)  # type: ignore
