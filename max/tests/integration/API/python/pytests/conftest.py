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


def pytest_addoption(parser):
    parser.addoption(
        "--custom-ops-path",
        type=str,
        default="",
        help="Path to custom Ops package",
    )
