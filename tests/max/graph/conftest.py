# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

import os
from pathlib import Path

import pytest
from max import mlir
from max.graph import core as _c


@pytest.fixture
def modular_path() -> Path:
    """Returns the path to the Modular .derived directory."""
    modular_path = os.getenv("MODULAR_PATH")
    assert modular_path is not None

    return Path(modular_path)


@pytest.fixture(scope="module")
def mlir_context() -> mlir.Context:
    """Set up the MLIR context by registering and loading Modular dialects."""
    with mlir.Context() as ctx, mlir.Location.unknown():
        registry = mlir.DialectRegistry()
        _c.load_modular_dialects(registry)
        ctx.append_dialect_registry(registry)
        ctx.load_all_available_dialects()
        yield ctx
