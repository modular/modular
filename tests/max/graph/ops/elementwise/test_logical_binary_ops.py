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
"""test the max.graph python bindings."""

from __future__ import annotations

from collections.abc import Callable

import pytest
from conftest import (
    GraphBuilder,
    broadcast_shapes,
    broadcastable_shapes,
    tensor_types,
)
from hypothesis import assume, given, reject
from hypothesis import strategies as st
from max.dtype import DType
from max.graph import DeviceRef, Shape, TensorType, TensorValue, Value
from max.graph.ops import logical_and, logical_or, logical_xor
from max.graph.value import Numeric

LOGICAL_BINARY_OPS = [logical_or, logical_and, logical_xor]


@pytest.mark.parametrize("logical_op", LOGICAL_BINARY_OPS)
@given(
    tensor_type=tensor_types(dtypes=st.just(DType.bool)),
)
def test_logical_and__same_type(
    logical_op: Callable[[Value, Value], TensorValue],
    graph_builder: GraphBuilder,
    tensor_type: TensorType,
) -> None:
    with graph_builder(input_types=[tensor_type, tensor_type]) as graph:
        x, y = graph.inputs
        op = logical_op(x, y)
        assert op.type == tensor_type


@pytest.mark.parametrize("logical_op", LOGICAL_BINARY_OPS)
@given(tensor_type=...)
def test_logical_and__invalid_dtype(
    logical_op: Callable[[Value, Value], TensorValue],
    graph_builder: GraphBuilder,
    tensor_type: TensorType,
) -> None:
    assume(tensor_type.dtype != DType.bool)
    with graph_builder(input_types=[tensor_type, tensor_type]) as graph:
        x, y = graph.inputs
        with pytest.raises(ValueError):
            logical_op(x, y)


@pytest.mark.parametrize("logical_op", LOGICAL_BINARY_OPS)
@given(shapes=broadcastable_shapes(2))
def test_logical_and__broadcast(
    logical_op: Callable[[Value, Value], TensorValue],
    graph_builder: GraphBuilder,
    shapes: list[Shape],
) -> None:
    s1, s2 = shapes
    broadcast_shape = broadcast_shapes(s1, s2)
    with graph_builder(
        input_types=[
            TensorType(DType.bool, s1, DeviceRef.CPU()),
            TensorType(DType.bool, s2, DeviceRef.CPU()),
        ],
    ) as graph:
        x, y = graph.inputs
        assert logical_op(x, y).shape == broadcast_shape
        assert logical_op(y, x).shape == broadcast_shape


@pytest.mark.skip("MSDK-1158")
@pytest.mark.parametrize("logical_op", LOGICAL_BINARY_OPS)
@given(s1=..., s2=...)
def test_logical_and__invalid_broadcast(
    logical_op: Callable[[Value, Value], TensorValue],
    graph_builder: GraphBuilder,
    s1: Shape,
    s2: Shape,
) -> None:
    try:
        broadcast_shapes(s1, s2)
    except ValueError:
        pass
    else:
        reject()  # valid broadcast

    with graph_builder(
        input_types=[
            TensorType(DType.bool, s1, DeviceRef.CPU()),
            TensorType(DType.bool, s2, DeviceRef.CPU()),
        ],
    ) as graph:
        x, y = graph.inputs
        with pytest.raises(Exception):
            logical_op(x, y)
        with pytest.raises(Exception):
            logical_op(y, x)


@pytest.mark.parametrize("logical_op", LOGICAL_BINARY_OPS)
@given(tensor_type=tensor_types(dtypes=st.just(DType.bool)), b=...)
def test_logical_and__python_bool(
    logical_op: Callable[[Value | Numeric, Value | Numeric], TensorValue],
    graph_builder: GraphBuilder,
    tensor_type: TensorType,
    b: bool,
) -> None:
    with graph_builder(input_types=[tensor_type]) as graph:
        (x,) = graph.inputs
        assert logical_op(x, b).type == tensor_type
        assert logical_op(b, x).type == tensor_type
