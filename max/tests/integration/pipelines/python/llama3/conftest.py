# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

import os
from pathlib import Path

import pytest


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
