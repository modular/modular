# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
"""Test the max.graph Python bindings."""

import sys
from random import Random
from operator import mul

import pytest
from hypothesis import strategies as st
from hypothesis import HealthCheck, assume, given, settings
from max import mlir
from max.graph import DType, Graph, GraphValue, TensorType, graph, ops
from max.graph.type import shape, Shape, dim, StaticDim, Dim


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
        assert static_reshape_neg_one.shape == shape((2, 15))

        symbolic_reshape = graph.inputs[1].reshape(("channels", "batch"))
        symbolic_reshape_neg_one = graph.inputs[1].reshape(("channels", -1))
        assert symbolic_reshape_neg_one.shape == shape(("channels", "batch"))

        graph.output(
            static_reshape,
            static_reshape_neg_one,
            symbolic_reshape,
            symbolic_reshape_neg_one,
        )

        graph._mlir_op.verify()


def st_limit_shape_size(shape_st, size_limit: int):
    return shape_st.filter(
        lambda shape: static_known_shape_size(shape) < size_limit
    )


def static_known_shape_size(shape: Shape):
    """Returns the size of a shape only considering static dims"""
    size = 1
    for dim in shape:
        if dim.is_static():
            size *= dim.dim
    return size


limited_shape = st_limit_shape_size(st.lists(st.from_type(Dim)), 2**63)


@given(dtype=..., input_shape=limited_shape, random=...)
def test_reshapes_shuffle(dtype: DType, input_shape: Shape, random: Random):
    with Graph(
        "reshape",
        input_types=[
            TensorType(dtype, shape=input_shape),
        ],
    ) as graph:
        target_shape = random.sample(input_shape, len(input_shape))
        out = graph.inputs[0].reshape(target_shape)

        assert out.shape == target_shape

        graph.output(out)
        graph._mlir_op.verify()


# `-1` can't calculate a shape if there is a zero. So filter those out.
non_zero_dim = st.from_type(Dim).filter(lambda x: x != dim(0))
non_zero_static_dim = st.from_type(StaticDim).filter(lambda x: x != dim(0))

# TODO(GRA-830): This should be 2**63, but that fails with `-1`.
limited_shape = st_limit_shape_size(st.lists(non_zero_dim), 2**31)


@given(dtype=..., input_shape=limited_shape, random=...)
def test_reshapes_neg_one(dtype: DType, input_shape: Shape, random: Random):
    # need at least one element to replace with `-1`
    assume(len(input_shape) > 0)

    with Graph(
        "reshape",
        input_types=[
            TensorType(dtype, shape=input_shape),
        ],
    ) as graph:
        i = random.randint(0, len(input_shape) - 1)
        target_shape = input_shape[:i] + [-1] + input_shape[i + 1 :]
        random.shuffle(target_shape)

        out = graph.inputs[0].reshape(target_shape)

        replaced_dim = input_shape[i]
        expected_shape = list(
            map(lambda dim: replaced_dim if dim == -1 else dim, target_shape)
        )

        assert out.shape == expected_shape

        graph.output(out)
        graph._mlir_op.verify()


# TODO(GRA-830): This should be 2**63, but that fails with `-1`.
limited_static_shape = st_limit_shape_size(
    st.lists(non_zero_static_dim), 2**31
)


# TODO(MSDK-662): The first assume that limits shape size filters too much.
# Overall, there is probably a better way to handle limiting shape size.
@settings(suppress_health_check=[HealthCheck.filter_too_much])
@given(
    dtype=...,
    merge_dims=limited_static_shape,
    rest_dims=limited_shape,
    random=...,
)
def test_reshapes_neg_one_multi_replace(
    dtype: DType,
    merge_dims: list[StaticDim],
    rest_dims: list[Dim],
    random: Random,
):
    # Two list of dims together must have a limited size.
    assume(
        static_known_shape_size(merge_dims) * static_known_shape_size(rest_dims)
        < 2**31
    )
    # need at least one element to replace with `-1`
    assume(len(merge_dims) > 0)

    input_shape = merge_dims + rest_dims
    random.shuffle(input_shape)
    with Graph(
        "reshape",
        input_types=[
            TensorType(dtype, shape=input_shape),
        ],
    ) as graph:
        target_shape = rest_dims + [-1]
        random.shuffle(target_shape)

        out = graph.inputs[0].reshape(target_shape)

        replaced_size = dim(static_known_shape_size(merge_dims))
        expected_shape = list(
            map(lambda dim: replaced_size if dim == -1 else dim, target_shape)
        )

        assert out.shape == expected_shape

        graph.output(out)
        graph._mlir_op.verify()


@given(input_type=..., random=...)
def test_transpose_pos(input_type: TensorType, random: Random):
    rank = len(input_type.shape)
    assume(rank > 0)
    a = random.randint(0, rank - 1)
    b = random.randint(0, rank - 1)
    with Graph(
        "transpose",
        input_types=[input_type],
    ) as graph:
        out = graph.inputs[0].transpose(a, b)

        target_shape = input_type.shape
        target_shape[a], target_shape[b] = target_shape[b], target_shape[a]

        assert out.shape == target_shape

        graph.output(out)
        graph._mlir_op.verify()


@given(input_type=..., random=...)
def test_transpose_neg(input_type: TensorType, random: Random):
    rank = len(input_type.shape)
    assume(rank > 0)
    a = random.randint(-rank, -1)
    b = random.randint(-rank, -1)
    with Graph(
        "transpose",
        input_types=[input_type],
    ) as graph:
        out = graph.inputs[0].transpose(a, b)

        target_shape = input_type.shape
        a += rank
        b += rank
        target_shape[a], target_shape[b] = target_shape[b], target_shape[a]

        assert out.shape == target_shape

        graph.output(out)
        graph._mlir_op.verify()
