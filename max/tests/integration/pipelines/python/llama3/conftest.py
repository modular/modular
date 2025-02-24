# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

import os
from pathlib import Path

import pytest
from hypothesis import settings
from max.engine import InferenceSession

pytest_plugins = "test_common.registry"

# When running in CI, graph tests can take around 300ms for a single run.
# These seem to be due to CI running under very high cpu usage.
# A similar effect can be achieved locally be running with each test multible times `--runs_per_test=3`.
# They all launch at the same time leading to exceptionally heavy cpu usage.
# We have reasonable test suite timeouts. Use those instead of hypothesis deadlines.
settings.register_profile("graph_tests", deadline=None)
settings.load_profile("graph_tests")


@pytest.fixture
def modular_path() -> Path:
    """Returns the path to the Modular .derived directory."""
    modular_path = os.getenv("MODULAR_PATH")
    assert modular_path is not None
    return Path(modular_path)


@pytest.fixture(scope="session")
def testdata_directory() -> Path:
    """Returns the path to the Modular .derived directory."""
    path = os.getenv("PIPELINES_TESTDATA")
    assert path is not None
    return Path(path)


@pytest.fixture(scope="session")
def session() -> InferenceSession:
    return InferenceSession()
