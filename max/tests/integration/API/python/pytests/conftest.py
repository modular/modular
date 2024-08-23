# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

import os
from pathlib import Path

import max.driver as md
import pytest
from max.engine import InferenceSession


@pytest.fixture
def modular_path() -> Path:
    """Returns the path to the Modular .derived directory."""
    modular_path = os.getenv("MODULAR_PATH")
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


@pytest.fixture
def dynamic_model_path(modular_path: Path) -> Path:
    """Returns the path to the dynamic shape model."""
    return (
        modular_path
        / "SDK"
        / "integration-test"
        / "EngineAPI"
        / "Inputs"
        / "dynamic-model.mlir"
    )


def pytest_addoption(parser):
    parser.addoption(
        "--custom-ops-path",
        type=str,
        default="",
        help="Path to custom Ops package",
    )


@pytest.fixture(scope="module")
def session() -> InferenceSession:
    return InferenceSession()


@pytest.fixture(scope="module")
def gpu_session() -> InferenceSession:
    return InferenceSession(device=md.CUDA())
