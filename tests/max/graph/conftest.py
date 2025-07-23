# ===----------------------------------------------------------------------=== #
# Copyright (c) 2025, Modular Inc. All rights reserved.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions:
# https://llvm.org/LICENSE.txt
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ===----------------------------------------------------------------------=== #

from __future__ import annotations

import builtins
import itertools
import math
import operator
import os
import random
from collections.abc import Generator, Sequence
from functools import reduce
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import pytest
from hypothesis import Phase, assume, settings
from hypothesis import strategies as st
from max import mlir
from max.dtype import DType
from max.graph import (
    BufferType,
    DeviceRef,
    Dim,
    DimLike,
    Graph,
    KernelLibrary,
    Shape,
    StaticDim,
    SymbolicDim,
    TensorType,
    _OpaqueType,
)

# When running in CI, graph tests can take around 300ms for a single run.
# These seem to be due to CI running under very high cpu usage.
# A similar effect can be achieved locally be running with each test multiple times `--runs_per_test=3`.
# They all launch at the same time leading to exceptionally heavy cpu usage.
# We have reasonable test suite timeouts. Use those instead of hypothesis deadlines.
settings.register_profile("graph_tests", deadline=None)
settings.load_profile("graph_tests")

settings.register_profile(
    "failfast", phases=[Phase.explicit, Phase.reuse, Phase.generate]
)


MAX_INT32 = np.iinfo(np.int32).max
MAX_INT64 = np.iinfo(np.int64).max

# TODO(MSDK-1234): add f8e5m2 and f8e4m3fn to test date types
dtypes = st.sampled_from(
    [
        d
        for d in DType
        if d
        not in (
            DType.float8_e5m2,
            DType.float8_e5m2fnuz,
            DType.float8_e4m3fn,
            DType.float8_e4m3fnuz,
        )
    ]
)


def uniform_distributed_static_dims(min: int = 0, max: int = 2**63 - 1):
    return st.builds(StaticDim, st.integers(min_value=min, max_value=max))


def clip(v, min, max):  # noqa: ANN001
    # Like np.clip, but more stable for python int types.
    # np.clip will cast to a float for values > intmax.
    return min if v < min else max if v > max else v  # noqa: FURB136


@st.composite
def log_bucket(draw, e: float, min: int, max: int):  # noqa: ANN001
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
    draw,  # noqa: ANN001
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


def valid_broadcast_rank(shape_st, max_size: int | None = None):  # noqa: ANN001
    """Samples valid ranks to broadcast a shape to.

    Valid ranks are >= len(shape).
    """
    return shape_st.flatmap(lambda shape: st.integers(len(shape), max_size))


def tensor_types(
    dtypes=dtypes,  # noqa: ANN001
    shapes=shapes(),  # noqa: ANN001
    device=DeviceRef.CPU(),  # noqa: ANN001
) -> st.Strategy[TensorType]:
    return st.builds(TensorType, dtypes, shapes, st.just(device))


def opaque_types():
    names = st.text(alphabet="abcdefghijklmnopqrstuvwxyz", min_size=1)
    si64s = st.integers(min_value=-(2**63), max_value=2**63 - 1)
    parameters = st.dictionaries(
        keys=names,
        values=st.one_of(
            dtypes,
            names,
            si64s,
            st.booleans(),
        ),
    )
    return st.builds(_OpaqueType, names, parameters)


def buffer_types(
    dtypes=dtypes,  # noqa: ANN001
    shapes=shapes(),  # noqa: ANN001
    device=DeviceRef.CPU(),  # noqa: ANN001
) -> st.Strategy[BufferType]:
    return st.builds(BufferType, dtypes, shapes, st.just(device))


def axes(shapes):  # noqa: ANN001
    def strategy(shape):  # noqa: ANN001
        assume(shape.rank > 0)
        return st.integers(min_value=-shape.rank, max_value=shape.rank - 1)

    return shapes.flatmap(strategy)


def axes_of(
    shapes: st.Strategy[Shape | TensorType], pred: Callable[[Dim], bool]
):
    """Samples the axes satisfying the given predicate for the  dimensions of
    the given shapes.
    """

    def strategy(shape: Shape | TensorType):
        if isinstance(shape, TensorType):
            shape = shape.shape

        rank = shape.rank
        positives = [i for i, x in enumerate(shape) if pred(x)]
        negatives = [i - rank for i in positives]

        return (
            st.sampled_from(positives + negatives)
            if positives or negatives
            else st.nothing()
        )

    return shapes.flatmap(strategy)


def static_axes(shapes):  # noqa: ANN001
    """Samples the axes corresponding to the static dimensions of the given
    shapes.
    """

    return axes_of(shapes, lambda x: isinstance(x, StaticDim))


def symbolic_axes(shapes):  # noqa: ANN001
    """Samples the axes corresponding to the symbolic dimensions of the given
    shapes.
    """

    return axes_of(shapes, lambda x: isinstance(x, SymbolicDim))


def non_static_axes(shapes):  # noqa: ANN001
    """Samples the axes corresponding to the non-static (SymbolicDim or
    AlgebraicDim) dimensions of the given shapes.
    """

    return axes_of(shapes, lambda x: not isinstance(x, StaticDim))


def new_axes(shapes):  # noqa: ANN001
    def strategy(shapes):  # noqa: ANN001
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
st.register_type_strategy(_OpaqueType, opaque_types())


def broadcastable_subshape(shape: list[Dim], random: random.Random):
    shape = shape[random.randint(0, len(shape)) :]
    ones = random.sample(range(len(shape)), random.randint(0, len(shape)))
    for idx in ones:
        shape[idx] = StaticDim(1)
    return shape


def _broadcastable_shapes(n: int, dims_strategy):  # noqa: ANN001
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
            lambda shapes: [
                TensorType(dtype, shape, device=DeviceRef.CPU())
                for shape in shapes
            ]
        )
    )


def broadcast_shapes(s1: list[Dim], s2: list[Dim]) -> list[Dim | None]:
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


@pytest.fixture
def modular_path() -> Path:
    """Returns the path to the Modular .derived directory."""
    modular_path = os.getenv("MODULAR_PATH")
    assert modular_path is not None

    return Path(modular_path)


@pytest.fixture(scope="module")
def mlir_context() -> Generator[mlir.Context]:
    """Set up the MLIR context by registering and loading Modular dialects."""
    with mlir.Context() as ctx, mlir.Location.unknown():
        yield ctx


@pytest.fixture(scope="module")
def kernel_library(mlir_context: mlir.Context) -> Generator[KernelLibrary]:
    """Set up the kernel library for the current system."""
    yield KernelLibrary(mlir_context)


GraphBuilder = Callable[..., Graph]


@pytest.fixture(scope="module")
def graph_builder(
    request: pytest.FixtureRequest,
    mlir_context: mlir.Context,
    kernel_library: KernelLibrary,
) -> Generator[GraphBuilder]:
    yield lambda *args, **kwargs: Graph(
        request.node.name,
        *args,
        context=mlir_context,
        kernel_library=kernel_library,
        **kwargs,
    )


@pytest.fixture
def testdata_directory() -> Path:
    """Returns the path to the Modular .derived directory."""
    path = os.getenv("TESTDATA_DIRECTORY")
    assert path is not None
    return Path(path)
