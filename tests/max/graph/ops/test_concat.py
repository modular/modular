# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
"""ops.concat tests."""

import pytest
from conftest import axes, shapes, symbolic_axes, tensor_types
from hypothesis import assume, given
from hypothesis import strategies as st
from max.dtype import DType
from max.graph import DeviceRef, Dim, Shape, StaticDim, TensorType, ops

shared_dtypes = st.shared(st.from_type(DType))
shared_shapes = st.shared(shapes())
shared_tensor_types = st.shared(
    tensor_types(dtypes=shared_dtypes, shapes=shared_shapes)
)

# For test speed, don't do huge concats.
MAX_CONCAT_SIZE = 100


def with_dim(shape: Shape, axis: int, dim: StaticDim):
    shape = Shape(shape)
    shape[axis] = dim
    return shape


@given(
    base_type=shared_tensor_types,
    axis_sizes=st.lists(st.from_type(StaticDim), max_size=MAX_CONCAT_SIZE),
    axis=axes(shared_tensor_types),
)
def test_concat__static_dim(
    graph_builder, base_type: TensorType, axis_sizes: list[StaticDim], axis: int
):
    assume(axis_sizes)
    merged_size = sum(dim.dim for dim in axis_sizes)
    # TODO: test the error for this case
    assume(merged_size < 2**63)

    input_types = [
        TensorType(
            base_type.dtype,
            with_dim(base_type.shape, axis, dim),
            DeviceRef.CPU(),
        )
        for dim in axis_sizes
    ]

    with graph_builder(input_types=input_types) as graph:
        out = ops.concat(graph.inputs, axis)
        assert out.shape == with_dim(base_type.shape, axis, merged_size)
        graph.output(out)


@given(
    base_type=shared_tensor_types,
    axis=st.integers(),
)
def test_concat__axis_out_of_bounds(
    graph_builder, base_type: TensorType, axis: int
):
    assume(axis < -base_type.rank or axis >= base_type.rank)

    with graph_builder(input_types=[base_type]) as graph:
        with pytest.raises(IndexError):
            out = ops.concat(graph.inputs, axis)


@given(
    type_a=shared_tensor_types,
    type_b=tensor_types(shapes=shared_shapes),
    axis=axes(shared_tensor_types),
)
def test_concat__bad_dtype(
    graph_builder, type_a: TensorType, type_b: TensorType, axis: int
):
    assume(type_a.dtype != type_b.dtype)
    assert type_a.shape == type_b.shape
    assume(
        not isinstance(type_a.shape[axis], StaticDim)
        or 2 * type_a.shape[axis].dim < 2**63
    )

    with graph_builder(input_types=[type_a, type_b]) as graph:
        with pytest.raises(ValueError):
            out = ops.concat(graph.inputs, axis)


@given(axis=...)
def test_concat__no_inputs(graph_builder, axis: int):
    with graph_builder(input_types=[]) as graph:
        with pytest.raises(ValueError):
            out = ops.concat([], axis)


@given(
    type_a=shared_tensor_types,
    type_b=tensor_types(dtypes=shared_dtypes),
    axis=axes(shared_tensor_types),
)
def test_concat__different_ranks(
    graph_builder, type_a: TensorType, type_b: TensorType, axis: int
):
    assert type_a.dtype == type_b.dtype
    assume(type_a.rank != type_b.rank)

    with graph_builder(input_types=[type_a, type_b]) as graph:
        with pytest.raises(ValueError):
            out = ops.concat(graph.inputs, axis)


@given(
    type_a=shared_tensor_types,
    type_b=shared_tensor_types.flatmap(
        lambda t: tensor_types(
            dtypes=shared_dtypes,
            shapes=shapes(min_rank=t.rank, max_rank=t.rank),
        )
    ),
    axis=axes(shared_tensor_types),
)
def test_concat__mismatched_dims(
    graph_builder, type_a: TensorType, type_b: TensorType, axis: int
):
    assert type_a.dtype == type_b.dtype
    assert type_a.rank == type_b.rank
    assume(
        not all(
            d1 == d2
            for i, (d1, d2) in enumerate(zip(type_a.shape, type_b.shape))
            if i != (axis if axis >= 0 else axis + type_a.rank)
        )
    )

    with graph_builder(input_types=[type_a, type_b]) as graph:
        with pytest.raises(ValueError):
            out = ops.concat(graph.inputs, axis)


@given(base_type=shared_tensor_types, axis=symbolic_axes(shared_tensor_types))
def test_concat__symbolic__size_1(
    graph_builder, base_type: TensorType, axis: int
):
    assume(not isinstance(base_type.shape[axis], StaticDim))

    with graph_builder(input_types=[base_type]) as graph:
        out = ops.concat(graph.inputs, axis)
        assert out.shape == base_type.shape
        graph.output(out)


@given(
    base_type=shared_tensor_types,
    axis=axes(shared_tensor_types),
    axis_dims=st.lists(st.from_type(Dim), min_size=2, max_size=MAX_CONCAT_SIZE),
)
def test_concat__symbolic__algebraic_result(
    graph_builder,
    base_type: TensorType,
    axis: int,
    axis_dims: list[Dim],
):
    assume(not all(isinstance(dim, StaticDim) for dim in axis_dims))
    merged_static_size = sum(
        dim.dim for dim in axis_dims if isinstance(dim, StaticDim)
    )
    assume(merged_static_size < 2**63)

    input_types = [
        TensorType(
            base_type.dtype,
            with_dim(base_type.shape, axis, dim),
            DeviceRef.CPU(),
        )
        for dim in axis_dims
    ]

    with graph_builder(input_types=input_types) as graph:
        out = ops.concat(graph.inputs, axis)
        assert out.shape == with_dim(base_type.shape, axis, sum(axis_dims))


def test_oncat_different_devices(graph_builder):
    input_types = [
        TensorType(DType.float32, [12], DeviceRef.CPU(0)),
        TensorType(DType.float32, [13], DeviceRef.CPU(0)),
        TensorType(DType.float32, [14], DeviceRef.CPU(1)),
    ]

    with (
        graph_builder(input_types=input_types) as graph,
        pytest.raises(
            ValueError, match="Cannot concat inputs on different devices .*"
        ),
    ):
        out = ops.concat(graph.inputs, 0)
