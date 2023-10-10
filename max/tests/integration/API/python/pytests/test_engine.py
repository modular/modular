# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
"""Test the modular.engine Python bindings with MOF."""

import os
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


def test_execute_success(mo_model_path: Path):
    session = me.InferenceSession()
    model = session.load(mo_model_path)
    output = model.execute(input=np.ones((5)))
    assert "output" in output.keys()
    assert np.allclose(
        output["output"],
        np.array([4.0, 2.0, -5.0, 3.0, 6.0]).astype(np.float32),
    )
