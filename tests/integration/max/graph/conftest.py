# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

import functools
import os
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pytest
import torch
from hypothesis import given, reject, settings
from hypothesis import strategies as st
from hypothesis.extra import numpy as nps
from max.driver import Tensor
from max.engine import InferenceSession
from max.graph import TensorType
from max.graph.type import Dim, StaticDim, SymbolicDim

MAX_INPUT_MAGNITUDE = 1e5
MIN_SHAPE_DIM = 1
MAX_SHAPE_DIM = 16

ACCURACY_RTOL = 1e-2
ACCURACY_ATOL = 1e-8


@pytest.fixture(scope="session")
def session() -> InferenceSession:
    return InferenceSession()


@pytest.fixture
def graph_testdata() -> Path:
    """Returns the path to the Modular .derived directory."""
    path = os.getenv("GRAPH_TESTDATA")
    assert path is not None
    return Path(path)


def elements(dtype, max_magnitude=MAX_INPUT_MAGNITUDE, **kwargs):
    if max_magnitude:
        kwargs.setdefault("max_value", max_magnitude)
        kwargs.setdefault("min_value", -max_magnitude)

    return nps.from_dtype(
        dtype, allow_nan=False, allow_infinity=False, **kwargs
    )


@st.composite
def shapes(
    draw: st.DrawFn, tensor_type: TensorType, static_dims: Dict[str, int] = {}
):
    """Defines a strategy for generating a concrete shape from a TensorType.

    Args:
        draw: Used with @st.composite to write type hints.
        tensor_type: Input tensor type.
        static_dims: Map of symbolic dimension name to static dimensions. This
            is used when there are dimensions in the test input arrays that must
            be hardcoded. For example, if the input has dims ["a", "b", "c"],
            and "c" must be "a" + "b".

    Returns:
        Shape tuple
    """

    def draw_dim(dim: Dim):
        if isinstance(dim, SymbolicDim):
            if (static := static_dims.get(dim.name)) is not None:
                return static

            sides = st.integers(
                min_value=MIN_SHAPE_DIM, max_value=MAX_SHAPE_DIM
            )
            # shared ensures symbolic dimensions match
            return draw(st.shared(sides, key=dim.name))
        elif isinstance(dim, StaticDim):
            return int(dim)
        raise NotImplementedError

    return tuple(draw_dim(dim) for dim in tensor_type.shape)


def arrays(tensor_type: TensorType, static_dims={}, **kwargs):
    dtype = np.dtype(tensor_type.dtype.to_numpy())
    return nps.arrays(
        dtype,
        shapes(tensor_type, static_dims),
        elements=elements(dtype, **kwargs),
    )


def assert_allclose(result, expected):
    try:
        np.testing.assert_allclose(
            result,
            expected,
            rtol=ACCURACY_RTOL,
            atol=ACCURACY_ATOL,
            equal_nan=True,
        )
        # TODO(MSDK-830): Don't filter out tests where we NaN and torch doesn't
    except AssertionError:
        if np.isnan(result).any():
            reject()


def given_input_types(input_types, static_dims: Dict[str, int] = {}):
    return given(
        st.tuples(
            *(
                arrays(input_type, static_dims=static_dims)
                for input_type in input_types
            )
        )
    )


def execute(model, inputs):
    results = model.execute(*[Tensor.from_numpy(inp) for inp in inputs])
    if results:
        return results[0].to_numpy()


def modular_graph_test(
    session,
    graph,
    *,
    static_dims: Dict[str, int] = {},
    hypothesis_settings: Optional[settings] = None,
):
    def decorator(test_fn):
        model = session.load(graph)

        @given_input_types(
            (input.type for input in graph.inputs), static_dims=static_dims
        )
        def test_correctness(inputs):
            model_execute = functools.partial(execute, model)
            test_fn(model_execute, inputs, [torch.tensor(i) for i in inputs])

        if hypothesis_settings is not None:
            test_correctness = hypothesis_settings(test_correctness)
        test_correctness()
        return test_correctness

    return decorator
