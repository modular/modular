# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
"""ops.broadcast_to tests."""

import re

import pytest
from conftest import (
    broadcast_shapes,
    broadcastable_static_positive_shapes,
    shapes,
    valid_broadcast_rank,
)
from hypothesis import assume, given
from hypothesis import strategies as st
from max.dtype import DType
from max.graph import DeviceRef, Graph, ShapeLike, TensorType, ops


def _sorted_by_rank(x: ShapeLike, y: ShapeLike) -> tuple[ShapeLike, ShapeLike]:
    return (x, y) if len(list(x)) <= len(list(y)) else (y, x)


@given(input_shapes=broadcastable_static_positive_shapes(2))
def test_broadcast_to_shape_attr(input_shapes: list[ShapeLike]) -> None:
    """Tests broadcast_to with a shape attribute."""
    from_shape, to_shape = _sorted_by_rank(*input_shapes)
    assume(from_shape and to_shape)
    to_shape = broadcast_shapes(from_shape, to_shape)

    graph = Graph(
        "broadcast_to_shape_attr",
        lambda from_tensor, to_tensor: ops.broadcast_to(
            from_tensor, shape=to_tensor.shape
        ),
        input_types=[
            TensorType(DType.int64, from_shape, device=DeviceRef.CPU()),
            TensorType(DType.int64, to_shape, device=DeviceRef.CPU()),
        ],
    )
    assert "rmo.broadcast_to" in str(graph)


# Set a max rank so that the test doesn't time out.
max_rank = 10
shared_shapes = st.shared(shapes(max_rank=max_rank))


@given(
    input_shape=shared_shapes,
    to_rank=valid_broadcast_rank(shared_shapes, max_size=max_rank),
)
def test_broadcast_to_tensor_value(
    input_shape: ShapeLike, to_rank: int
) -> None:
    """Tests broadcast_to with a tensor value."""
    assume(input_shape)

    out_dims = [f"dim{i}" for i in range(to_rank)]
    graph = Graph(
        "broadcast_to_tensor_value",
        forward=lambda x, y: ops.broadcast_to(x, y, out_dims=out_dims),
        input_types=[
            TensorType(DType.bfloat16, input_shape, device=DeviceRef.CPU()),
            TensorType(DType.int64, (to_rank,), device=DeviceRef.CPU()),
        ],
    )
    assert "rmo.mo.broadcast_to" in str(graph)
    expected_type = TensorType(DType.bfloat16, out_dims, device=DeviceRef.CPU())
    assert graph.output_types == [expected_type]


def test_broadcast_to__error_message():
    input_shape = [6]
    output_shape = [6, 7]

    with Graph(
        "broadcast_to_error_message",
        input_types=[
            TensorType(
                dtype=DType.float32, shape=input_shape, device=DeviceRef.CPU()
            )
        ],
    ) as graph:
        with pytest.raises(
            ValueError,
            match=re.escape(
                "[broadcast_to] input dimension at index 0 (6) must be either 1 or equal to corresponding output dimension at index 1 (7)"
            ),
        ):
            ops.broadcast_to(graph.inputs[0].tensor, output_shape)


def test_broadcast_to__error_message_symbolic_shapes():
    input_shape = ["D0"]
    output_shape = ["D1", "D2"]

    with Graph(
        "broadcast_to_error_message",
        input_types=[
            TensorType(
                dtype=DType.float32, shape=input_shape, device=DeviceRef.CPU()
            )
        ],
    ) as graph:
        with pytest.raises(
            ValueError,
            match=re.escape(
                "[broadcast_to] input dimension at index 0 (D0) must be either 1 or equal to corresponding output dimension at index 1 (D2)"
            ),
        ):
            ops.broadcast_to(graph.inputs[0].tensor, output_shape)
