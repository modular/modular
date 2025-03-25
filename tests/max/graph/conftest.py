# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

from __future__ import annotations

import builtins
import itertools
import math
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
    DimLike,
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


def uniform_distributed_static_dims(min: int = 0, max: int = 2**63 - 1):
    return st.builds(StaticDim, st.integers(min_value=min, max_value=max))


def clip(v, min, max):
    # Like np.clip, but more stable for python int types.
    # np.clip will cast to a float for values > intmax.
    return min if v < min else max if v > max else v


@st.composite
def log_bucket(draw, e: float, min: int, max: int):
    lower = clip(int(2**e), min, max)
    upper = clip(int(2 ** (e + 1)), min, max)
    return draw(st.integers(min_value=lower, max_value=upper))


def log_distributed_static_dims(min: int = 1, max: int = 2**63 - 1):
    assert min > 0, "can't generate 0 with log distribution"
    return (
        st.floats(min_value=math.log2(min), max_value=math.log2(max))
        .flatmap(lambda e: log_bucket(e, min, max))
        .map(StaticDim)
    )


def static_dims(min: int = 0, max: int = 2**63 - 1):
    return st.one_of(
        uniform_distributed_static_dims(min, max),
        log_distributed_static_dims(builtins.max(1, min), max),
    )


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
small_dims = st.one_of(static_dims(min=1, max=16), symbolic_dims)


@st.composite
def all_shapes(
    draw,
    min_rank: int = 1,
    max_rank: int = 5,
    dims: st.SearchStrategy[Dim] = dims,
    include_dims: Sequence[DimLike | st.SearchStrategy[Dim]] = (),
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
    min_rank -= len(include_dims)
    max_rank -= len(include_dims)
    generated_dims = draw(st.lists(dims, min_size=min_rank, max_size=max_rank))
    generated_include_dims = draw(
        st.tuples(
            *(
                dim if isinstance(dim, st.SearchStrategy) else st.just(dim)
                for dim in include_dims
            )
        )
    )
    all_dims = (*generated_include_dims, *generated_dims)
    product = reduce(
        operator.mul, [int(dim) for dim in Shape(all_dims).static_dims], 1
    )
    assume(product <= max_size)
    return draw(st.permutations(all_dims).map(Shape))


def small_shapes(*args, **kwargs):
    return all_shapes(*args, dims=small_dims, **kwargs)


def shapes(*args, **kwargs):
    if "dims" in kwargs:
        return all_shapes(*args, **kwargs)
    return st.one_of(small_shapes(*args, **kwargs), all_shapes(*args, **kwargs))


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
