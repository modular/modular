# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
from os import getenv
from pathlib import Path

import pytest
from max.driver import Accelerator
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
        modular_path / "SDK" / "integration-test" / "API" / "c" / "mo-model.api"
    )


@pytest.fixture
def no_input_path(modular_path: Path) -> Path:
    """Returns the path to a model spec without inputs."""
    return (
        modular_path
        / "SDK"
        / "integration-test"
        / "API"
        / "Inputs"
        / "no-inputs.mlir"
    )


@pytest.fixture
def scalar_input_path(modular_path: Path) -> Path:
    """Returns the path to a model spec with scalar inputs."""
    return (
        modular_path
        / "SDK"
        / "integration-test"
        / "API"
        / "Inputs"
        / "scalar-input.mlir"
    )


@pytest.fixture
def aliasing_outputs_path(modular_path: Path) -> Path:
    """Returns the path to a model spec with outputs that alias each other."""
    return (
        modular_path
        / "SDK"
        / "integration-test"
        / "API"
        / "Inputs"
        / "aliasing-outputs.mlir"
    )


@pytest.fixture(scope="module")
def gpu_session() -> InferenceSession:
    return InferenceSession(devices=[Accelerator()])
