# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
"""Test the max.graph Python bindings."""

from collections.abc import Collection

import pytest
from conftest import (
    shapes,
    static_dims,
    symbolic_dims,
    tensor_types,
)
from hypothesis import assume, example, given
from hypothesis import strategies as st
from max.dtype import DType
from max.graph import Dim, Graph, Shape, StaticDim, TensorType


def test_reshape() -> None:
    """Builds a simple graph with a reshape and checks the IR."""
    with Graph(
        "reshape",
        input_types=[
            TensorType(dtype=DType.float32, shape=[6, 5]),
            TensorType(dtype=DType.float32, shape=["batch", "channels"]),
        ],
    ) as graph:
        static_reshape = graph.inputs[0].reshape((3, 10))
        static_reshape_neg_one = graph.inputs[0].reshape((2, -1))
        assert static_reshape_neg_one.shape == [2, 15]

        symbolic_reshape = graph.inputs[1].reshape(("channels", "batch"))
        symbolic_reshape_neg_one = graph.inputs[1].reshape(("channels", -1))
        assert symbolic_reshape_neg_one.shape == ["channels", "batch"]

        graph.output(
            static_reshape,
            static_reshape_neg_one,
            symbolic_reshape,
            symbolic_reshape_neg_one,
        )


def subseqs(c: Collection):
    if not c:
        return st.just(type(c)())
    subseq_indices = st.sets(st.sampled_from(range(len(c))))
    return subseq_indices.map(
        lambda indices: type(c)(v for i, v in enumerate(c) if i in indices)  # type: ignore
    )


def negative_one_reshape(shapes):
    return (
        shapes.flatmap(subseqs)
        .map(lambda subseq: [*subseq, -1])
        .flatmap(st.permutations)
    )


shared_shapes = st.shared(shapes())
# Use a max rank of 4 to reduce the probability of drawing 1-dims.
shared_static_shapes = st.shared(shapes(dims=static_dims()))


@given(
    input_type=tensor_types(shapes=shared_shapes),
    output_shape=shared_shapes.flatmap(st.permutations),
)
def test_reshape__can_permute_input_shape(
    input_type: TensorType, output_shape: list[Dim]
):
    with Graph("reshape", input_types=[input_type]) as graph:
        out = graph.inputs[0].reshape(output_shape)
        assert out.shape == output_shape
        graph.output(out)


@given(
    input_type=tensor_types(shapes=shared_shapes),
    reshape_shape=negative_one_reshape(shared_shapes),
)
@pytest.mark.skip("MAXPLAT-151")
def test_reshapes__can_replace_any_dims_with_negative_one(
    input_type: TensorType, reshape_shape: list[Dim]
):
    with Graph("reshape", input_types=[input_type]) as graph:
        out = graph.inputs[0].reshape(reshape_shape)
        assert out.dtype == input_type.dtype
        for dim, expected in zip(out.shape, reshape_shape):
            if expected != -1:
                assert dim == expected
        graph.output(out)


@given(
    input_type=tensor_types(shapes=shapes(include_dims=[0])),
    reshape_shape=shapes(include_dims=[0]),
)
def test_reshapes__zero_dim(input_type: TensorType, reshape_shape: list[Dim]):
    assume(0 in input_type.shape)
    assume(0 in reshape_shape)
    assume(  # TODO (MSDK-763): remove this assumption
        all(
            d in input_type.shape
            for d in reshape_shape
            if not isinstance(d, StaticDim)
        )
    )
    with Graph("reshape", input_types=[input_type]) as graph:
        out = graph.inputs[0].reshape(reshape_shape)
        assert out.dtype == input_type.dtype
        assert out.shape == reshape_shape
        graph.output(out)


def shapes_plus_ones(shapes=shapes()):
    ones = st.lists(st.just(1))
    shapes = shapes.flatmap(lambda shape: ones.map(lambda ones: shape + ones))
    return shapes.flatmap(st.permutations)


@given(
    input_type=tensor_types(shapes=shared_shapes),
    reshape_shape=shapes_plus_ones(shared_shapes),
)
def test_reshapes__unsqueeze(input_type: TensorType, reshape_shape: list[Dim]):
    with Graph("reshape", input_types=[input_type]) as graph:
        out = graph.inputs[0].reshape(reshape_shape)
        assert out.dtype == input_type.dtype
        assert out.shape == reshape_shape
        graph.output(out)


@given(
    input_type=tensor_types(shapes=shapes_plus_ones(shared_shapes)),
    reshape_shape=shared_shapes,
)
def test_reshapes__squeeze(input_type: TensorType, reshape_shape: list[Dim]):
    with Graph("reshape", input_types=[input_type]) as graph:
        out = graph.inputs[0].reshape(reshape_shape)
        assert out.dtype == input_type.dtype
        assert out.shape == reshape_shape
        graph.output(out)


@given(
    input_type=tensor_types(shapes=shared_shapes),
    output_shape=shared_shapes.flatmap(st.permutations),
    dim=symbolic_dims,
)
@pytest.mark.skip(reason="MAXPLAT-151")
def test_reshape__fails_with_different_symbolic_dim(
    input_type: TensorType,
    output_shape: list[Dim],
    dim: Dim,
):
    assume(dim not in input_type.shape)
    with Graph("reshape", input_types=[input_type]) as graph:
        with pytest.raises(ValueError):
            graph.inputs[0].reshape([*output_shape, dim])


@given(
    input_type=tensor_types(shapes=shared_static_shapes),
    output_shape=shared_static_shapes.flatmap(st.permutations)
    .filter(lambda shape: shape[-1] > 1)
    .map(lambda shape: shape[:-1]),
)
@example(
    # Specifically test an example whose dim product can be represented by an
    # int64, but not by an int32.
    input_type=TensorType(
        DType.int8,
        Shape([268435456, 17]),
    ),
    output_shape=Shape([268435456]),
)
@pytest.mark.skip(reason="MAXPLAT-151")
def test_reshape__fails_with_different_number_of_elements(
    input_type: TensorType,
    output_shape: Shape,
) -> None:
    with Graph("reshape", input_types=[input_type]) as graph:
        with pytest.raises(ValueError):
            graph.inputs[0].reshape(output_shape)


@given(
    input_type=tensor_types(shapes=st.lists(st.just(1))),
    output_shape=st.lists(st.just(1)),
)
def test_reshape__can_reshape_single_element_tensors(
    input_type: TensorType,
    output_shape: list[Dim],
):
    with Graph("reshape", input_types=[input_type]) as graph:
        out = graph.inputs[0].reshape(output_shape)
        assert out.dtype == input_type.dtype
        assert out.shape == output_shape
        graph.output(out)
