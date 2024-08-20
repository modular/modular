# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

import os
from pathlib import Path

from hypothesis import given, reject, strategies as st
from hypothesis.extra import numpy as nps
from hypothesis.strategies import integers, lists, shared, tuples
import numpy as np
import pytest
import torch

from max.engine import InferenceSession
from max.graph import DType, Graph, TensorType
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
def shapes(draw: st.DrawFn, tensor_type: TensorType):
    def draw_dim(dim: Dim):
        if isinstance(dim, SymbolicDim):
            sides = st.integers(
                min_value=MIN_SHAPE_DIM, max_value=MAX_SHAPE_DIM
            )
            # shared ensures symbolic dimemnsions match
            return draw(st.shared(sides, key=dim.name))
        elif isinstance(dim, StaticDim):
            return int(dim)
        raise NotImplementedError

    return tuple(draw_dim(dim) for dim in tensor_type.shape)


def arrays(tensor_type: TensorType, **kwargs):
    dtype = np.dtype(tensor_type.dtype.to_numpy())
    return nps.arrays(
        dtype, shapes(tensor_type), elements=elements(dtype, **kwargs)
    )


def modular_vs_torch_test(session, graph, torch_fn):
    input_types = [input.tensor_type for input in graph.inputs]
    model = session.load(graph)

    @given(st.tuples(*(arrays(input_type) for input_type in input_types)))
    def test_correctness(inputs):
        names = [spec.name for spec in model.input_metadata]
        results = model.execute(**dict(zip(names, inputs)))
        result = results["output0"]
        expected = torch_fn(*(torch.tensor(i) for i in inputs)).detach().numpy()
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

    test_correctness()
