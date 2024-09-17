# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
"""ops.broadcast_to tests."""

from conftest import (
    broadcast_shapes,
    broadcastable_static_positive_shapes,
    graph_result_type,
    shapes,
    valid_broadcast_rank,
)
from hypothesis import assume, given
from hypothesis import strategies as st
from max.dtype import DType
from max.graph import Graph, ShapeLike, TensorType, ops


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
            TensorType(DType.int64, from_shape),
            TensorType(DType.int64, to_shape),
        ],
    )
    assert "rmo.broadcast_to" in str(graph)


# Set a max rank so that the test doesn't time out.
max_rank = 10
shared_shapes = st.shared(shapes(min_size=0, max_size=max_rank))


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
            TensorType(DType.bfloat16, input_shape),
            TensorType(DType.int64, (to_rank,)),
        ],
    )
    assert "rmo.mo.broadcast_to" in str(graph)
    assert TensorType.from_mlir(graph_result_type(graph)) == TensorType(
        graph.inputs[0].dtype, shape=out_dims
    )
