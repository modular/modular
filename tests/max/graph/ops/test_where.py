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
from functools import reduce

import pytest
from conftest import (
    GraphBuilder,
    broadcast_shapes,
    broadcastable_tensor_types,
    shapes_are_broadcastable,
    tensor_types,
    value_in_range,
)
from hypothesis import assume, event, given
from hypothesis import strategies as st
from max.dtype import DType
from max.graph import DeviceRef, Graph, Shape, TensorType, ops


@given(input_types=broadcastable_tensor_types(3))
def test_where(input_types: list[TensorType]) -> None:
    input_types[0].dtype = DType.bool

    with Graph("where", input_types=input_types) as graph:
        cond, x, y = (v.tensor for v in graph.inputs)
        out = ops.where(cond, x, y)

        expected = reduce(broadcast_shapes, (t.shape for t in input_types))  # type: ignore
        assert out.shape == expected
        assert out.dtype in (t.dtype for t in input_types)

        graph.output(out)


shared_dtypes = st.shared(st.from_type(DType))


@given(
    cond_type=tensor_types(dtypes=st.just(DType.bool)),
    x_type=tensor_types(dtypes=shared_dtypes),
    y_type=tensor_types(dtypes=shared_dtypes),
)
def test_where_with_non_broadcastable_shapes(
    graph_builder: GraphBuilder,
    cond_type: TensorType,
    x_type: TensorType,
    y_type: TensorType,
) -> None:
    assume(
        not shapes_are_broadcastable(
            cond_type.shape, x_type.shape, y_type.shape
        )
    )
    with Graph("where", input_types=[cond_type, x_type, y_type]) as graph:
        cond, x, y = (v.tensor for v in graph.inputs)
        with pytest.raises(ValueError):
            ops.where(cond, x, y)


def test_where_error_message_with_non_bool_condition() -> None:
    with Graph(
        "where_non_bool",
        input_types=[
            TensorType(shape=[3], dtype=DType.float32, device=DeviceRef.CPU()),
            TensorType(shape=[3], dtype=DType.int32, device=DeviceRef.CPU()),
            TensorType(shape=[3], dtype=DType.int32, device=DeviceRef.CPU()),
        ],
    ) as graph:
        cond, x, y = graph.inputs
        with pytest.raises(
            ValueError,
            match=r"Expected condition to be a boolean tensor, but got a tensor with dtype DType.float32",
        ):
            ops.where(cond.tensor, x.tensor, y.tensor)


def test_where_error_message_with_mismatched_condition_shape() -> None:
    with Graph(
        "where_mismatched_condition_shape",
        input_types=[
            TensorType(shape=[2, 4], dtype=DType.bool, device=DeviceRef.CPU()),
            TensorType(shape=[2, 6], dtype=DType.int32, device=DeviceRef.CPU()),
            TensorType(shape=[2], dtype=DType.int32, device=DeviceRef.CPU()),
        ],
    ) as graph:
        cond, x, y = graph.inputs
        with pytest.raises(
            ValueError,
            match="are neither equivalent nor broadcastable",
        ):
            ops.where(cond.tensor, x.tensor, y.tensor)


def test_where_error_message_with_mismatched_devices() -> None:
    with Graph(
        "where_mismatched_devices",
        input_types=[
            TensorType(shape=[3], dtype=DType.bool, device=DeviceRef.CPU()),
            TensorType(shape=[3], dtype=DType.int32, device=DeviceRef.CPU()),
            TensorType(shape=[3], dtype=DType.int32, device=DeviceRef.GPU()),
        ],
    ) as graph:
        cond, x, y = graph.inputs
        with pytest.raises(
            ValueError,
            match=re.escape(
                "All tensors must be on the same device, but got devices: cpu:0, cpu:0, gpu:0"
            ),
        ):
            ops.where(cond.tensor, x.tensor, y.tensor)


shared_shapes = st.shared(st.from_type(Shape))


@given(
    cond_type=tensor_types(shapes=shared_shapes, dtypes=st.just(DType.bool)),
    x_type=tensor_types(shapes=shared_shapes),
    y_type=tensor_types(shapes=shared_shapes),
)
def test_where_with_promotable_dtypes(
    graph_builder: GraphBuilder,
    cond_type: TensorType,
    x_type: TensorType,
    y_type: TensorType,
) -> None:
    assume(x_type.dtype != y_type.dtype)

    with graph_builder(input_types=[cond_type, x_type, y_type]) as graph:
        cond, x, y = graph.inputs
        try:
            result = ops.where(cond.tensor, x.tensor, y.tensor)
        except ValueError as e:
            assert "Unsafe cast" in str(e)
            event("types don't promote")
        else:
            assert result.dtype in (x_type.dtype, y_type.dtype)
            assert result.shape == cond_type.shape
            event("types promote")


@given(
    cond_type=tensor_types(shapes=shared_shapes, dtypes=st.just(DType.bool)),
    x_type=tensor_types(shapes=shared_shapes, dtypes=shared_dtypes),
    y=shared_dtypes.flatmap(value_in_range),
)
def test_where_with_constant(
    graph_builder: GraphBuilder,
    cond_type: TensorType,
    x_type: TensorType,
    y: float,
) -> None:
    with graph_builder(input_types=[cond_type, x_type]) as graph:
        cond, x = graph.inputs
        result = ops.where(cond.tensor, x.tensor, y)
        assert result.type == x.type
        result = ops.where(cond.tensor, y, x.tensor)
        assert result.type == x.type


@given(
    cond_type=tensor_types(shapes=shared_shapes, dtypes=st.just(DType.bool)),
    dtype=shared_dtypes,
    y=shared_dtypes.flatmap(value_in_range),
    x=shared_dtypes.flatmap(value_in_range),
)
def test_where_with_two_constants(
    graph_builder: GraphBuilder,
    cond_type: TensorType,
    dtype: DType,
    x: float,
    y: float,
) -> None:
    with graph_builder(input_types=[cond_type]) as graph:
        cond = graph.inputs[0].tensor
        with pytest.raises(TypeError):
            ops.where(cond, x, y)
