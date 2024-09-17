# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

from __future__ import annotations

import itertools
import os
import random
from pathlib import Path
from typing import Optional

import pytest
from hypothesis import assume
from hypothesis import strategies as st
from max import _graph, mlir
from max.dtype import DType
from max.graph import BufferType, TensorType
from max.graph.type import Dim, Shape, StaticDim, SymbolicDim

dtypes = st.sampled_from([d for d in DType if d is not DType._unknown])
static_dims = st.builds(
    StaticDim, st.integers(min_value=0, max_value=2**63 - 1)
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
dims = st.one_of(static_dims, symbolic_dims)


def shapes(min_size=0, max_size=None, include_dims=()):
    return (
        st.lists(
            dims,
            min_size=max(0, min_size - len(include_dims)),
            max_size=None if max_size
            is None else max(0, max_size - len(include_dims)),
        )
        .map(lambda shape: (*shape, *include_dims))
        .flatmap(st.permutations)
        .map(Shape)
    )


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
st.register_type_strategy(StaticDim, static_dims)
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
        assert d1 == d2 or d1 == StaticDim(1) or d2 == StaticDim(1)
        return d1 if d2 == StaticDim(1) else d2

    return list(
        reversed(
            [
                broadcast_dim(d1, d2)
                for d1, d2 in itertools.zip_longest(reversed(s1), reversed(s2))
            ]
        )
    )


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
