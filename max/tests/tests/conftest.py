# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

from __future__ import annotations

from typing import Generator

import pytest
from max import mlir


@pytest.fixture(scope="function")
def mlir_context() -> Generator[mlir.Context]:
    """Set up the MLIR context by registering and loading Modular dialects."""
    with mlir.Context() as ctx, mlir.Location.unknown():
        yield ctx
