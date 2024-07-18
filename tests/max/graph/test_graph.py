# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
"""Test the max.graph Python bindings."""
import pytest
import sys
from max.graph import DType, Graph, TensorType, graph, mlir, ops


def test_mlir_module_create() -> None:
    """Tests whether we can import mlir and create a Module.

    This is a basic smoke test for max.graph Python packaging.
    """
    with mlir.Context(), mlir.Location.unknown():
        _ = mlir.Module.create()


@pytest.mark.skip(
    reason=(
        "Implicit conversions from Python builtins not implemented yet"
        " (MSDK-640)."
    )
)
def test_elementwise_add_graph() -> None:
    """Builds a simple graph with an elementwise addition and checks the IR."""
    with Graph(
        "elementwise_add",
        input_types=[
            TensorType(dtype=DType.float32, dims=["batch", "channels"])
        ],
    ) as graph:
        graph.output(graph.inputs[0] + 1)

        graph._mlir_op.verify()


@pytest.mark.skipif(sys.version_info.minor > 10, reason="MSDK-636")
def test_location() -> None:
    def bar():
        return graph.location()

    def foo():
        return bar()

    with mlir.Context():
        loc = foo()

    # We can't really introspect locations except to get their `str`
    assert f"{__name__}.test_location" in str(loc)
    assert f"{__name__}.foo" in str(loc)
    assert f"{__name__}.bar" in str(loc)


def test_add_op() -> None:
    """Builds a simple graph with an elementwise addition and checks the IR."""
    input_type = TensorType(dtype=DType.float32, dims=["batch", "channels"])
    with Graph("add", input_types=(input_type, input_type)) as graph:
        graph.output(ops.add(graph.inputs[0], graph.inputs[1]))

        graph._mlir_op.verify()
