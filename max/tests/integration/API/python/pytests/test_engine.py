# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
"""Test the modular.engine Python bindings with MOF."""

import os
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pytest

import modular.engine as me


@pytest.fixture
def modular_path() -> Path:
    """Returns the path to the Modular .derived directory."""
    modular_path = os.getenv("MODULAR_PATH")
    assert modular_path is not None

    return Path(modular_path)


@pytest.fixture
def mo_model_path(modular_path: Path) -> Path:
    """Returns the path to the generated BasicMLP model."""
    return modular_path / "All" / "test" / "API" / "c" / "mo-model.api"


@pytest.fixture
def mo_custom_ops_model_path(modular_path: Path) -> Path:
    """Returns the path to the generated BasicMLP model."""
    return (
        modular_path / "All" / "test" / "API" / "c" / "custom-ops-override.api"
    )


@pytest.fixture
def custom_ops_package_path(request) -> Path:
    return Path(request.config.getoption("--custom-ops-path"))


def test_execute_success(mo_model_path: Path):
    session = me.InferenceSession()
    model = session.load(mo_model_path)
    output = model.execute(input=np.ones((5)))
    assert "output" in output.keys()
    assert np.allclose(
        output["output"],
        np.array([4.0, 2.0, -5.0, 3.0, 6.0]).astype(np.float32),
    )


# TODO: Move to an experimental modular engine package
@dataclass
class CustomOptions(me.ExperimentalLoadOptions):
    type: str = "experimental"
    custom_ops_path: str = field(default="")


def test_custom_ops(
    mo_custom_ops_model_path: Path, custom_ops_package_path: Path
):
    session = me.InferenceSession()
    model = session.load(mo_custom_ops_model_path)
    inputs = np.ones((1)) * 4
    output = model.execute(input0=inputs)
    assert "output" in output.keys()
    assert np.allclose(
        output["output"],
        np.array([2.0]).astype(np.float32),
    )

    options = CustomOptions()
    options.custom_ops_path = str(custom_ops_package_path)
    model_with_custom_op = session.load(mo_custom_ops_model_path, options)
    inputs = np.ones((1)) * 4
    output = model_with_custom_op.execute(input0=inputs)
    assert "output" in output.keys()
    assert np.allclose(
        output["output"],
        np.array([4.0]).astype(np.float32),
    )
