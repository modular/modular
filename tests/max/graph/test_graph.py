# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
"""Test the max.graph Python bindings."""

import sys

import pytest
from max import mlir
from max.graph import DType, Graph, GraphValue, TensorType, graph, ops


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
            TensorType(dtype=DType.float32, shape=["batch", "channels"])
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
    input_type = TensorType(dtype=DType.float32, shape=["batch", "channels"])
    with Graph("add", input_types=(input_type, input_type)) as graph:
        lhs, rhs = graph.inputs
        elemwise_sum = ops.add(lhs, rhs)
        graph.output(elemwise_sum)

        graph._mlir_op.verify()

        # Check that the arg/result name attributes were added.
        assert "argument_names = " in str(graph._mlir_op)
        assert "result_names = " in str(graph._mlir_op)


def test_add_op_closure() -> None:
    """Uses a closure to build a simple graph with an elementwise addition
    and checks the IR.
    """

    def elementwise_add(lhs: GraphValue, rhs: GraphValue) -> GraphValue:
        return ops.add(lhs, rhs)

    input_type = TensorType(dtype=DType.float32, shape=["batch", "channels"])
    add_graph = Graph("add", elementwise_add, (input_type, input_type))

    add_graph._mlir_op.verify()

    assert "rmo.add" in str(add_graph._mlir_op)
    assert "mo.output" in str(add_graph._mlir_op)
