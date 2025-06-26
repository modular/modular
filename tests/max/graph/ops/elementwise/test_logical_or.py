# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
"""test the max.graph python bindings."""

import pytest
from conftest import broadcast_shapes, broadcastable_shapes, tensor_types
from hypothesis import assume, given, reject
from hypothesis import strategies as st
from max.dtype import DType
from max.graph import DeviceRef, Shape, TensorType
from max.graph.ops import logical_or


@given(tensor_type=tensor_types(dtypes=st.just(DType.bool)))
def test_logical_or__same_type(graph_builder, tensor_type: TensorType):
    with graph_builder(input_types=[tensor_type, tensor_type]) as graph:
        x, y = graph.inputs
        op = logical_or(x, y)
        assert op.type == tensor_type


@given(tensor_type=...)
def test_logical_or__invalid_dtype(graph_builder, tensor_type: TensorType):
    assume(tensor_type.dtype != DType.bool)
    with graph_builder(input_types=[tensor_type, tensor_type]) as graph:
        x, y = graph.inputs
        with pytest.raises(ValueError):
            logical_or(x, y)


@given(shapes=broadcastable_shapes(2))
def test_logical_or__broadcast(graph_builder, shapes: list[Shape]):
    s1, s2 = shapes
    broadcast_shape = broadcast_shapes(s1, s2)
    with graph_builder(
        input_types=[
            TensorType(DType.bool, s1, DeviceRef.CPU()),
            TensorType(DType.bool, s2, DeviceRef.CPU()),
        ],
    ) as graph:
        x, y = graph.inputs
        assert logical_or(x, y).shape == broadcast_shape
        assert logical_or(y, x).shape == broadcast_shape


@pytest.mark.skip("MSDK-1158")
@given(s1=..., s2=...)
def test_logical_or__invalid_broadcast(graph_builder, s1: Shape, s2: Shape):
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
            logical_or(x, y)
        with pytest.raises(Exception):
            logical_or(y, x)


@given(tensor_type=tensor_types(dtypes=st.just(DType.bool)), b=...)
def test_logical_or__python_bool(
    graph_builder, tensor_type: TensorType, b: bool
):
    with graph_builder(input_types=[tensor_type]) as graph:
        (x,) = graph.inputs
        assert logical_or(x, b).type == tensor_type
        assert logical_or(b, x).type == tensor_type
