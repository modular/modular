# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
"""Test the max.graph Python bindings."""
import pytest
from max.graph import mlir, DType, Graph, TensorType


def test_mlir_module_create() -> None:
    """Tests whether we can import mlir and create a Module.

    This is a basic smoke test for max.graph Python packaging.
    """
    with mlir.ir.Context(), mlir.ir.Location.unknown():
        _ = mlir.ir.Module.create()


def test_elementwise_add_graph() -> None:
    """Builds a simple graph with an elementwise addition and checks the IR."""
    with Graph(
        "elementwise_add",
        input_types=[
            TensorType(dtype=DType.float32, dims=["batch", "channels"])
        ],
    ) as graph:
        graph.output(graph.inputs[0] + 1)
