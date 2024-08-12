# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
"""Test the max.graph Python bindings."""

import itertools
import sys
from operator import mul
from random import Random
from typing import Union

import pytest
from hypothesis import HealthCheck, assume, given, settings
from hypothesis import strategies as st
from max import mlir
from max.graph import DType, Graph, GraphValue, TensorType, graph, ops
from max.graph.type import Dim, Shape, StaticDim, dim, shape


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


def test_slice_basic():
    with Graph(
        "slice",
        input_types=[TensorType(DType.int32, [1, 2, 3, 4, 5])],
    ) as graph:
        out = graph.inputs[0][:, 1, ..., 3]

        assert out.shape == shape([1, 1, 3, 4, 1])
        graph.output(out)
        graph._mlir_op.verify()


def test_slice_with_graph_value():
    with Graph(
        "slice",
        input_types=[TensorType(DType.int32, [5, "in_dim"])],
    ) as graph:
        start = ops.scalar(2, DType.int64)
        out = graph.inputs[0][
            (slice(start, None), 3), (slice(start, None), "out_dim")
        ]

        assert out.shape == shape([3, "out_dim"])
        graph.output(out)
        graph._mlir_op.verify()


def valid_slice_of_tensor(type: TensorType, random: Random):
    # Note, this currently does not generate int slices or any form of graph value.
    # Slice also does not support them yet.
    indices = []
    for dim in type.shape:
        choice = random.randint(0, 1)
        if choice == 0:
            # `:` include whole dim.
            indices.append(slice(None, None, None))
        else:
            # Specific int index.
            if dim.is_static():
                size = dim.dim
                indices.append(random.randint(-size, size - 1))
            else:
                # For dynamic dim, value could any size. So any int works.
                indices.append(random.randint(-(2**63), 2**63 - 1))

    # Add potential ellipsis.
    if random.choice([True, False]):
        ellipsis_end = random.randint(0, len(indices))
        ellipsis_start = random.randint(0, ellipsis_end)
        return indices[:ellipsis_start] + [Ellipsis] + indices[ellipsis_end:]
    else:
        # If there is no ellipsis, we can stop indexing at any point.
        # Should just slice the beginning dimensions
        end = random.randint(0, len(indices))
        return indices[:end]


def non_zero_tensor_and_dims(type: TensorType):
    return type.shape and dim(0) not in type.shape


def tensor_and_slice():
    return st.tuples(
        st.from_type(TensorType).filter(non_zero_tensor_and_dims), st.randoms()
    ).map(lambda args: (args[0], valid_slice_of_tensor(args[0], args[1])))


@given(tensor_and_indices=tensor_and_slice())
def test_slice_valid_ints(
    tensor_and_indices: tuple[TensorType, list[Union[int, Ellipsis, slice]]],
):
    input_type = tensor_and_indices[0]
    indices = tensor_and_indices[1]
    assume(len(input_type.shape) > 0)
    assume(len(indices) <= len(input_type.shape))

    with Graph(
        "slice",
        input_types=[input_type],
    ) as graph:
        out = ops.slice_tensor(graph.inputs[0], indices)

        try:
            ellipsis = indices.index(Ellipsis)
        except:
            # No ellipsis.
            ellipsis = len(indices)

        expected_shape = input_type.shape
        for i, index in enumerate(indices[:ellipsis]):
            if isinstance(index, int):
                expected_shape[i] = dim(1)
        for i, index in enumerate(reversed(indices[ellipsis + 1 :])):
            if isinstance(index, int):
                expected_shape[len(expected_shape) - i - 1] = dim(1)

        assert out.shape == expected_shape

        graph.output(out)
        graph._mlir_op.verify()
