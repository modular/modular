# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

import os
from pathlib import Path

import pytest
from max.engine import InferenceSession


@pytest.fixture(scope="session")
def session() -> InferenceSession:
    return InferenceSession()


@pytest.fixture
def graph_testdata() -> Path:
    """Returns the path to the Modular .derived directory."""
    path = os.getenv("GRAPH_TESTDATA")
    assert path is not None
    return Path(path)
