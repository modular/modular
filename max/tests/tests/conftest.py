# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

from __future__ import annotations

from typing import Generator

import pytest
from max import mlir
from max._core import graph


@pytest.fixture(scope="function")
def mlir_context() -> Generator[mlir.Context]:
    """Set up the MLIR context by registering and loading Modular dialects."""
    with mlir.Context() as ctx, mlir.Location.unknown():
        registry = mlir.DialectRegistry()
        graph.load_modular_dialects(registry)
        ctx.append_dialect_registry(registry)
        ctx.load_all_available_dialects()
        yield ctx
