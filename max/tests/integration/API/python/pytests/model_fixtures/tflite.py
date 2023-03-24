# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
"""All fixtures for end-to-end testing"""
from pathlib import Path
from typing import Generator

import pytest
from model_fixtures.utils import (
    ModelFormat,
    fixture_generator,  # cleanup,
    modular_derived_path,
)


@pytest.fixture(scope="session")
def tflite_basic_mlp() -> Path:
    """This is path to tiny generated BasicMLP model (< 1 MB), useful for sanity checking."""
    return (
        modular_derived_path()
        / "build"
        / "GeneratedTests"
        / "BINARIES"
        / "models"
        / "BasicMLP_float32_25x5"
        / "1"
        / "model.tflite"
    )


@pytest.fixture(scope="session")
def tflite_resnet50() -> Generator:
    """This is a path to medium-sized model about 100 MB in size."""

    key = "resnet50-v1.5"
    yield fixture_generator(key, ModelFormat.tflite)


@pytest.fixture(scope="session")
def tflite_camembert() -> Generator:
    """This is a path to large-sized model about 400 MB in size."""

    key = "camembert-base"
    yield fixture_generator(key, ModelFormat.tflite)


@pytest.fixture(scope="session")
def tflite_convnext() -> Generator:
    """This is a path to xlarge-sized model about 1.3 GB in size."""

    key = "facebook-convnext-xlarge-224-22k-1k"
    yield fixture_generator(key, ModelFormat.tflite)
