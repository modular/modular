# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
"""Tests attribute factories."""
import array
from pathlib import Path
from tempfile import NamedTemporaryFile

import pytest

import max.graph.core as _c
from max.graph import mlir, DType, TensorType


@pytest.fixture(scope="module")
def mlir_context():
    """Set up the MLIR context by registering and loading Modular dialects."""
    with mlir.Context() as ctx, mlir.Location.unknown():
        registry = mlir.DialectRegistry()
        _c.load_modular_dialects(registry)
        ctx.append_dialect_registry(registry)
        ctx.load_all_available_dialects()
        yield ctx


def test_array_attr(mlir_context) -> None:
    """Tests array attribute creation."""
    buffer = array.array("f", [42, 3.14])

    array_attr = _c.array_attr(
        "foo", buffer, TensorType(DType.float32, (2,)).to_mlir()
    )
    assert "dense_array" in str(array_attr)


def test_weights_attr(mlir_context) -> None:
    """Tests weighst attributes creation."""
    with NamedTemporaryFile("wb") as weights_file:
        weights_file.write(b"Hello, world!\n")

        weights_attr = _c.weights_attr(
            Path(weights_file.name),
            0,
            TensorType(DType.uint8, (2, 2)).to_mlir(),
            "bar",
        )
        assert "dense_resource" in str(weights_attr)
