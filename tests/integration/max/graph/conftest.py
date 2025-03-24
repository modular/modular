# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

import os
from pathlib import Path

import pytest
from hypothesis import settings
from max.driver import Accelerator
from max.engine import InferenceSession

# When running in CI, graph tests can take around 300ms for a single run.
# These seem to be due to CI running under very high cpu usage.
# A similar effect can be achieved locally be running with each test multible times `--runs_per_test=3`.
# They all launch at the same time leading to exceptionally heavy cpu usage.
# We have reasonable test suite timeouts. Use those instead of hypothesis deadlines.
settings.register_profile("graph_tests", deadline=None)
settings.load_profile("graph_tests")


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


@pytest.fixture
def custom_ops_mojopkg() -> Path:
    path = os.getenv("CUSTOM_OPS_PATH")
    assert path is not None, "Test couldn't find `CUSTOM_OPS_PATH` env"
    return Path(path)
