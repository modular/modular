# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

from __future__ import annotations

import itertools
import operator
import os
import random
from functools import reduce
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import pytest
from hypothesis import assume, settings
from hypothesis import strategies as st
from max import mlir
from max._core import graph as _graph
from max.dtype import DType
from max.graph import (
    BufferType,
    Dim,
    Graph,
    Shape,
    StaticDim,
    SymbolicDim,
    TensorType,
)

# When running in CI, graph tests can take around 300ms for a single run.
# These seem to be due to CI running under very high cpu usage.
# A similar effect can be achieved locally be running with each test multible times `--runs_per_test=3`.
# They all launch at the same time leading to exceptionally heavy cpu usage.
# We have reasonable test suite timeouts. Use those instead of hypothesis deadlines.
settings.register_profile("graph_tests", deadline=None)
settings.load_profile("graph_tests")

MAX_INT32 = np.iinfo(np.int32).max
MAX_INT64 = np.iinfo(np.int64).max

# TODO(MSDK-1234): add f8e5m2 and f8e4m3 to test date types
dtypes = st.sampled_from(
    [
        d
        for d in DType
        if d
        not in (
            DType._unknown,
            DType.float8_e5m2,
            DType.float8_e5m2fnuz,
            DType.float8_e4m3,
            DType.float8_e4m3fn,
            DType.float8_e4m3fnuz,
        )
    ]
)


def static_dims(min: int = 0, max: int = 2**63 - 1):
    return st.builds(StaticDim, st.integers(min_value=min, max_value=max))


symbolic_dims = st.builds(
    SymbolicDim,
    st.one_of(
        st.just("batch"),
        st.characters(min_codepoint=ord("a"), max_codepoint=ord("z")),
    ),
)
static_positive_dims = st.builds(
    StaticDim, st.integers(min_value=1, max_value=2**63 - 1)
)
dims = st.one_of(static_dims(), symbolic_dims)


@st.composite
def shapes(
    draw,
    min_rank: int = 1,
    max_rank: int = 10,
    include_dims: Sequence = (),
    is_static: bool = False,
    max_size: int = MAX_INT64,
) -> Shape:
    """A strategy to produce shapes whose product fits within an int64.

    This strategy simplifies downstream tests, which otherwise would all have
    to check for overflow themselves.

    Returns:
        A shape containing a mix of static and symbolic dims.
        The product of static dims in the shape is guaranteed to fit within an
        int64.
    """
    dims = list(include_dims)
    cumulative_product = reduce(
        operator.mul,
        [dim.dim for dim in dims if isinstance(dim, StaticDim)],
        1,
    )

    # Draw a random shape size.
    remaining_rank = draw(
        st.integers(
            min_value=max(0, min_rank - len(dims)),
            max_value=max_rank - len(dims),
        )
    )
    for _ in range(remaining_rank):
        # Decide whether to insert a symbolic dimension.
        if not is_static and draw(st.booleans()):
            dim = draw(symbolic_dims)
            dims.append(dim)
            continue

        # Draw a static dim.
        max_value = max_size // max(1, cumulative_product)
        # Get the max exponent: bits needed to represent the max value,
        # excluding sign and leading zeros.
        max_exponent = max_value.bit_length()

        # We want to draw from the range [2**e, 2**(e + 1) - 1).
        # So draw the exponent from integers {0, 1, ..., max_exponent - 1}.
        exponent = draw(st.integers(min_value=0, max_value=max_exponent - 1))
        low = 2**exponent

        # Ensure that the dim product fits in an int64.
        high = min(2 ** (exponent + 1) - 1, max_value)
        dim_value = draw(st.integers(min_value=low, max_value=high))
        assert (dim_value * cumulative_product) <= max_size

        dims.append(StaticDim(dim_value))
        cumulative_product *= dim_value

    return Shape(dims)


def valid_broadcast_rank(shape_st, max_size: int | None = None):
    """Samples valid ranks to broadcast a shape to.

    Valid ranks are >= len(shape).
    """
    return shape_st.flatmap(lambda shape: st.integers(len(shape), max_size))


def tensor_types(dtypes=dtypes, shapes=shapes()):
    return st.builds(TensorType, dtypes, shapes)


def buffer_types(dtypes=dtypes, shapes=shapes()):
    return st.builds(BufferType, dtypes, shapes)


def axes(shapes):
    def strategy(shape):
        assume(shape.rank > 0)
        return st.integers(min_value=-shape.rank, max_value=shape.rank - 1)

    return shapes.flatmap(strategy)


def new_axes(shapes):
    def strategy(shapes):
        if not shapes.rank:
            return st.sampled_from([0, -1])
        return st.integers(min_value=-shapes.rank, max_value=shapes.rank)

    return shapes.flatmap(strategy)


st.register_type_strategy(DType, dtypes)
st.register_type_strategy(Dim, dims)
st.register_type_strategy(Shape, shapes())
st.register_type_strategy(StaticDim, static_dims())
st.register_type_strategy(SymbolicDim, symbolic_dims)
st.register_type_strategy(TensorType, tensor_types())
st.register_type_strategy(BufferType, buffer_types())


def broadcastable_subshape(shape: list[Dim], random: random.Random):
    shape = shape[random.randint(0, len(shape)) :]
    ones = random.sample(range(len(shape)), random.randint(0, len(shape)))
    for idx in ones:
        shape[idx] = StaticDim(1)
    return shape


def _broadcastable_shapes(n: int, dims_strategy):
    return st.lists(dims_strategy).flatmap(
        lambda shape: st.lists(
            st.builds(broadcastable_subshape, st.just(shape), st.randoms()),
            min_size=n,
            max_size=n,
        )
    )


def broadcastable_shapes(n: int):
    return _broadcastable_shapes(n, dims)


def broadcastable_static_positive_shapes(n: int):
    return _broadcastable_shapes(n, static_positive_dims)


def broadcastable_tensor_types(n: int):
    return dtypes.flatmap(
        lambda dtype: broadcastable_shapes(n).map(
            lambda shapes: [TensorType(dtype, shape) for shape in shapes]
        )
    )


def broadcast_shapes(s1: list[Dim], s2: list[Dim]) -> list[Dim]:
    def broadcast_dim(d1: Optional[Dim], d2: Optional[Dim]):
        if d1 is None:
            return d2
        if d2 is None:
            return d1
        valid = d1 == d2 or d1 == StaticDim(1) or d2 == StaticDim(1)
        if not valid:
            raise ValueError(f"Invalid broadcast: {s1}, {s2}")
        return d1 if d2 == StaticDim(1) else d2

    return list(
        reversed(
            [
                broadcast_dim(d1, d2)
                for d1, d2 in itertools.zip_longest(reversed(s1), reversed(s2))
            ]
        )
    )


def graph_result_type(graph: Graph) -> mlir.Type:
    """Returns the graph's result type."""
    # Get the all the mo.graph body's operations (no nested operations).
    graph_block_ops = graph._mlir_op.regions[0].blocks[0].operations
    # Get the type of the terminator mo.output.
    # This is the output of the graph.
    return graph_block_ops[len(graph_block_ops) - 1].operation.operands[0].type


@pytest.fixture
def modular_path() -> Path:
    """Returns the path to the Modular .derived directory."""
    modular_path = os.getenv("MODULAR_PATH")
    assert modular_path is not None

    return Path(modular_path)


@pytest.fixture(scope="function")
def mlir_context() -> mlir.Context:
    """Set up the MLIR context by registering and loading Modular dialects."""
    with mlir.Context() as ctx, mlir.Location.unknown():
        registry = mlir.DialectRegistry()
        _graph.load_modular_dialects(registry)
        ctx.append_dialect_registry(registry)
        ctx.load_all_available_dialects()
        yield ctx


@pytest.fixture
def testdata_directory() -> Path:
    """Returns the path to the Modular .derived directory."""
    path = os.getenv("TESTDATA_DIRECTORY")
    assert path is not None
    return Path(path)
