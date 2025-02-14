# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

import os
from pathlib import Path

import pytest
from max.driver import Accelerator
from max.engine import InferenceSession


@pytest.fixture(scope="session")
def session() -> InferenceSession:
    return InferenceSession()


@pytest.fixture(scope="module")
def gpu_session() -> InferenceSession:
    return InferenceSession(devices=[Accelerator()])


@pytest.fixture
def graph_testdata() -> Path:
    """Returns the path to the Modular .derived directory."""
    path = os.getenv("GRAPH_TESTDATA")
    assert path is not None
    return Path(path)


@pytest.fixture
def counter_mojopkg() -> Path:
    path = os.getenv("MODULAR_COUNTER_OPS_PATH")
    assert path is not None, "Test couldn't find `MODULAR_COUNTER_OPS_PATH` env"
    return Path(path)
