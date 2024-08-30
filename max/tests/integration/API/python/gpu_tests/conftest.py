# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
from os import getenv
from pathlib import Path

import pytest
from max.driver import CUDA
from max.engine import InferenceSession


@pytest.fixture
def modular_path() -> Path:
    """Returns the path to the Modular .derived directory."""
    modular_path = getenv("MODULAR_PATH")
    assert modular_path is not None

    return Path(modular_path)


@pytest.fixture
def mo_model_path(modular_path: Path) -> Path:
    """Returns the path to the generated BasicMLP model."""
    return (
        modular_path
        / "SDK"
        / "integration-test"
        / "EngineAPI"
        / "c"
        / "mo-model.api"
    )


@pytest.fixture(scope="module")
def gpu_session() -> InferenceSession:
    return InferenceSession(device=CUDA())
