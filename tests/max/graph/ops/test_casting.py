# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
"""Test the max.graph Python bindings."""

import math
from collections.abc import Collection

import pytest
from conftest import axes, shapes, static_dims, symbolic_dims, tensor_types
from hypothesis import assume, given
from hypothesis import strategies as st
from max.dtype import DType
from max.graph import Graph, TensorType
from max.graph.type import Dim, Shape, StaticDim


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


def static_known_shape_size(shape: Shape):
    """Returns the size of a shape only considering static dims"""
    return math.prod(dim.dim for dim in shape if isinstance(dim, StaticDim))


def subseqs(c: Collection):
    if not c:
        return st.just(type(c)())
    subseq_indices = st.sets(st.sampled_from(range(len(c))))
    return subseq_indices.map(
        lambda indices: type(c)(v for i, v in enumerate(c) if i in indices)
    )


def negative_one_reshape(shapes):
    return (
        shapes.flatmap(subseqs)
        .map(lambda subseq: [*subseq, -1])
        .flatmap(st.permutations)
    )


shared_shapes = st.shared(shapes())


@given(
    input_type=tensor_types(shapes=shared_shapes),
    output_shape=shared_shapes.flatmap(st.permutations),
)
def test_reshape__can_permute_input_shape(
    input_type: TensorType, output_shape: list[Dim]
):
    assume(static_known_shape_size(input_type.shape) < 2**63 - 1)
    with Graph("reshape", input_types=[input_type]) as graph:
        out = graph.inputs[0].reshape(output_shape)
        assert out.shape == output_shape
        graph.output(out)


@pytest.mark.skip("MSDK-765")
@given(
    input_type=tensor_types(shapes=shared_shapes),
    reshape_shape=negative_one_reshape(shared_shapes),
)
def test_reshapes__can_replace_any_dims_with_negative_one(
    input_type: TensorType, reshape_shape: list[Dim]
):
    assume(static_known_shape_size(input_type.shape) < 2**63 - 1)
    assume(0 not in input_type.shape)

    # TODO(GRA-864): Remove this assumption
    assume(static_known_shape_size(input_type.shape) < 2**31 - 1)
    # TODO(MSDK-765): Support reshaping multiple dimensions
    assume(len(reshape_shape) == len(input_type.shape))

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
def test_reshape__fails_with_different_symoblic_dim(
    input_type: TensorType,
    output_shape: list[Dim],
    dim: Dim,
):
    assume(static_known_shape_size(input_type.shape) < 2**63 - 1)
    assume(0 not in input_type.shape)
    assume(dim not in input_type.shape)
    with Graph("reshape", input_types=[input_type]) as graph:
        graph.inputs[0].reshape([*output_shape, dim])


@given(
    input_type=tensor_types(shapes=shared_shapes),
    output_shape=shared_shapes.flatmap(st.permutations),
    dim=static_dims,
)
def test_reshape__fails_with_different_number_of_elements(
    input_type: TensorType,
    output_shape: list[Dim],
    dim: Dim,
):
    assume(static_known_shape_size([*input_type.shape, dim]) < 2**63 - 1)
    assume(all(isinstance(d, StaticDim) for d in input_type.shape))
    assume(0 not in input_type.shape)
    assume(dim > 1)  # 0 and 1 should both work
    with Graph("reshape", input_types=[input_type]) as graph:
        with pytest.raises(ValueError):
            graph.inputs[0].reshape([*output_shape, dim])


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


shared_types = st.shared(tensor_types())


@given(input_type=shared_types, a=axes(shared_types), b=axes(shared_types))
def test_transpose__output_shape(input_type: TensorType, a: int, b: int):
    assume(input_type.rank > 0)
    with Graph("transpose", input_types=[input_type]) as graph:
        out = graph.inputs[0].transpose(a, b)
        target_shape = list(input_type.shape)
        target_shape[a], target_shape[b] = target_shape[b], target_shape[a]
        assert out.shape == target_shape

        graph.output(out)


def test_rebind() -> None:
    """Builds a simple graph with a reshape and checks the IR."""
    with Graph(
        "rebind",
        input_types=[
            TensorType(dtype=DType.float32, shape=[6, 5]),
            TensorType(dtype=DType.float32, shape=["batch", "channels"]),
            TensorType(
                dtype=DType.float32, shape=["batch", "channels", "other"]
            ),
        ],
    ) as graph:
        rebind_to_existing_names = graph.inputs[0].rebind(("batch", "channels"))
        assert rebind_to_existing_names.shape == ["batch", "channels"]

        rebind_to_const = graph.inputs[1].rebind((3, 10))
        assert rebind_to_const.shape == [3, 10]

        rebind_to_new_names = graph.inputs[0].rebind(
            ("notbatch", "notchannels")
        )
        assert rebind_to_new_names.shape == ["notbatch", "notchannels"]

        rebind_expression = (
            graph.inputs[2]
            .reshape(("batch", -1))
            .rebind(("batch", "expression"))
        )
        assert rebind_expression.shape == ["batch", "expression"]

        graph.output(
            rebind_to_existing_names,
            rebind_to_const,
            rebind_to_new_names,
            rebind_expression,
        )
