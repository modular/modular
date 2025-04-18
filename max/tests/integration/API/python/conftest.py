# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

from os import getenv
from pathlib import Path

import max.driver as md
import pytest
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
def dynamic_model_path(modular_path: Path) -> Path:
    """Returns the path to the dynamic shape model."""
    return (
        modular_path
        / "SDK"
        / "integration-test"
        / "API"
        / "Inputs"
        / "dynamic-model.mlir"
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


@pytest.fixture
def named_inputs_path(modular_path: Path) -> Path:
    """Returns the path to a model spec that adds a series of named tensors."""
    return (
        modular_path
        / "SDK"
        / "integration-test"
        / "API"
        / "Inputs"
        / "named-inputs.mlir"
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
    devices: list[md.Device] = []
    for i in range(md.accelerator_count()):
        devices.append(md.Accelerator(i))

    devices.append(md.CPU())

    return InferenceSession(devices=devices)


def pytest_collection_modifyitems(items):
    # Prevent pytest from trying to collect Click commands and dataclasses as tests
    for item in items:
        if item.name.startswith("Test"):
            item.add_marker(pytest.mark.skip)


@pytest.fixture
def graph_testdata() -> Path:
    """Returns the path to the Modular .derived directory."""
    path = getenv("GRAPH_TESTDATA")
    assert path is not None
    return Path(path)
