# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
"""Test the max.graph Python bindings."""

import pytest
from conftest import shapes, tensor_types
from hypothesis import assume, given
from hypothesis import strategies as st
from max.graph import TensorType

shared_shapes = st.shared(shapes())


def unique_axes_list(shapes):
    def strategy(shape):
        rank = shape.rank
        axis = st.integers(min_value=-rank, max_value=rank - 1)
        return st.lists(
            axis,
            min_size=rank,
            max_size=rank,
            unique_by=lambda x: x + rank if x < 0 else x,
        )

    return shapes.flatmap(strategy)


@given(
    input_type=tensor_types(shapes=shared_shapes),
    dims=unique_axes_list(shared_shapes),
)
def test_permute_success(
    graph_builder, input_type: TensorType, dims: list[int]
):
    with graph_builder(input_types=[input_type]) as graph:
        out = graph.inputs[0].permute(dims)
        target_shape = [input_type.shape[d] for d in dims]
        assert out.shape == target_shape

        graph.output(out)


rank_sized_list_ints = shared_shapes.flatmap(
    lambda shape: st.lists(
        st.integers(), min_size=shape.rank, max_size=shape.rank
    )
)


@given(input_type=tensor_types(shapes=shared_shapes), dims=rank_sized_list_ints)
def test_permute_out_of_range(
    graph_builder, input_type: TensorType, dims: list[int]
):
    rank = input_type.rank
    assume(any(d >= rank or d < -rank for d in dims))
    with graph_builder(input_types=[input_type]) as graph:
        with pytest.raises(IndexError):
            out = graph.inputs[0].permute(dims)


@given(input_type=tensor_types(shapes=shared_shapes), dims=...)
def test_permute_wrong_rank(
    graph_builder, input_type: TensorType, dims: list[int]
):
    rank = input_type.rank
    assume(len(dims) != rank)
    with graph_builder(input_types=[input_type]) as graph:
        with pytest.raises(ValueError):
            out = graph.inputs[0].permute(dims)


def int_in_rank(shapes):
    def strategy(shape):
        rank = shape.rank
        return st.integers(min_value=0, max_value=rank - 1)

    return shapes.flatmap(strategy)


@given(
    input_type=tensor_types(shapes=shared_shapes),
    dims=unique_axes_list(shared_shapes),
    i=int_in_rank(shared_shapes),
    j=int_in_rank(shared_shapes),
)
def test_permute_duplicates(
    graph_builder, input_type: TensorType, dims: list[int], i: int, j: int
):
    assume(i != j)
    dims[i] = dims[j]
    with graph_builder(input_types=[input_type]) as graph:
        with pytest.raises(ValueError):
            out = graph.inputs[0].permute(dims)
