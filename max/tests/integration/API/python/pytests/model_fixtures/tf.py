# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
"""All fixtures for end-to-end testing"""
from pathlib import Path

import pytest

from model_fixtures.utils import (
    ModelFormat,
    modular_derived_path,
    fixture_generator,
)
from typing import Generator


@pytest.fixture(scope="session")
def tf_basic_mlp() -> Path:
    """This is path to tiny generated BasicMLP model (< 1 MB), useful for sanity checking."""

    return (
        modular_derived_path()
        / "build"
        / "GeneratedTests"
        / "BINARIES"
        / "models"
        / "BasicMLP_float32_25x5"
        / "0"
        / "model.savedmodel"
    )


@pytest.fixture(scope="session")
def tf_efficientnet() -> Generator:
    """This is a path to a model about 26 MB in size."""

    key = "efficientnet_b0"
    yield fixture_generator(key, ModelFormat.tf)


@pytest.fixture(scope="session")
def tf_dlrm() -> Generator:
    """This is a path to a model about 100 MB in size."""

    key = "dlrm-rm1"
    yield fixture_generator(key, ModelFormat.tf)


@pytest.fixture(scope="session")
def tf_bert() -> Generator:
    """This is a path to a model about 400 MB in size."""

    key = "bert-base-uncased-seqlen-128"
    yield fixture_generator(key, ModelFormat.tf)
