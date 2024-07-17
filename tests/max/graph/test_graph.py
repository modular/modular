# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
"""Test the max.graph Python bindings."""
import pytest
import sys
from max.graph import DType, Graph, TensorType, graph, mlir


def test_mlir_module_create() -> None:
    """Tests whether we can import mlir and create a Module.

    This is a basic smoke test for max.graph Python packaging.
    """
    with mlir.Context(), mlir.Location.unknown():
        _ = mlir.Module.create()


@pytest.mark.skip(reason="max.graph.Graph is currently a work in progress")
def test_elementwise_add_graph() -> None:
    """Builds a simple graph with an elementwise addition and checks the IR."""
    with Graph(
        "elementwise_add",
        input_types=[
            TensorType(dtype=DType.float32, dims=["batch", "channels"])
        ],
    ) as graph:
        graph.output(graph.inputs[0] + 1)


@pytest.mark.skipif(sys.version_info.minor > 10, reason="MSDK-636")
def test_location():
    def bar():
        return graph.location()

    def foo():
        return bar()

    with mlir.Context() as ctx:
        loc = foo()

    # We can't really introspect locations except to get their `str`
    assert f"{__name__}.test_location" in str(loc)
    assert f"{__name__}.foo" in str(loc)
    assert f"{__name__}.bar" in str(loc)
