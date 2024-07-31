# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
"""Test the max.graph Python bindings."""

import sys
from random import Random
from operator import mul
import itertools

import pytest
from hypothesis import strategies as st
from hypothesis import HealthCheck, assume, given, settings
from max import mlir
from max.graph import DType, Graph, GraphValue, TensorType, graph, ops
from max.graph.type import shape, Shape, dim, StaticDim, Dim


def broadcast_dim(dims):
    unique_dims = {d for d in dims if d is not None and d != dim(1)}
    if len(unique_dims) > 1:
        raise ValueError(f"dims not broadcastable: {dims}")
    elif not unique_dims:
        # No dims remaining, must have filtered out all one dims.
        return dim(1)
    return next(iter(unique_dims))


def broadcast_shape(shapes: list[Shape]):
    shape = [
        broadcast_dim(dims)
        for dims in itertools.zip_longest(
            *[reversed(shape) for shape in shapes]
        )
    ]
    shape.reverse()
    return shape


def broadcastable_subtype(tensor_type: TensorType, random: Random):
    n_dims = random.randint(0, len(tensor_type.shape))
    n_ones = random.randint(0, n_dims)
    ones = set(random.sample(range(n_dims), n_ones))
    return TensorType(
        tensor_type.dtype,
        [
            1 if i in ones else dim
            for i, dim in enumerate(tensor_type.shape[n_dims:])
        ],
    )


def co_broadcastable_tensor_types(n):
    return st.tuples(st.from_type(TensorType), st.randoms()).map(
        lambda args: [broadcastable_subtype(args[0], args[1]) for _ in range(n)]
    )


def dtype_is_one_of_input_dtypes(inputs: list[TensorType], out: TensorType):
    return out.dtype in map(lambda tensor: tensor.dtype, inputs)


@given(input_types=co_broadcastable_tensor_types(3))
def test_select(input_types: list[TensorType]):
    input_types[0].dtype = DType.bool

    with Graph(
        "select",
        input_types=input_types,
    ) as graph:
        cond = graph.inputs[0]
        x = graph.inputs[1]
        y = graph.inputs[2]

        out = ops.select(cond, x, y)

        expected_shape = broadcast_shape([type.shape for type in input_types])
        assert out.shape == expected_shape
        assert dtype_is_one_of_input_dtypes(input_types, out.tensor_type)

        graph.output(out)
        graph._mlir_op.verify()
