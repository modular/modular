# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
"""Test the max.graph Python bindings."""

import sys

import pytest
from hypothesis import assume, given
from hypothesis import strategies as st
from max import mlir
from max.dtype import DType
from max.graph import (
    Dim,
    Graph,
    Device,
    DeviceType,
    TensorType,
    TensorValue,
    graph,
    ops,
)

empty_graphs = st.builds(
    Graph, st.text(), input_types=st.lists(st.from_type(TensorType))
)


@given(graph=empty_graphs)
def test_simple_graphs(graph: Graph):
    assume(len(graph.inputs) > 0)
    with graph:
        graph.output(graph.inputs[0])


def test_mlir_module_create() -> None:
    """Tests whether we can import mlir and create a Module.

    This is a basic smoke test for max.graph Python packaging.
    """
    with mlir.Context(), mlir.Location.unknown():
        _ = mlir.Module.create()


def test_elementwise_add_graph() -> None:
    """Builds a simple graph with an elementwise addition and checks the IR."""
    with Graph(
        "elementwise_add",
        input_types=[
            TensorType(dtype=DType.float32, shape=["batch", "channels"])
        ],
    ) as graph:
        graph.output(graph.inputs[0] + 1)


def test_elementwise_add_graph_with_device_prop() -> None:
    """Builds a simple graph with explicit device on inputs and checks for output device propagation in the IR.
    """
    with Graph(
        "elementwise_add",
        input_types=[
            TensorType(
                dtype=DType.float32,
                shape=["batch", "channels"],
                device=Device.CUDA(0),
            ),
            TensorType(
                dtype=DType.float32,
                shape=["batch", "channels"],
                device=Device.CUDA(0),
            ),
        ],
    ) as graph:
        graph.output(graph.inputs[0] + graph.inputs[1])
        # Ensure input tensor has cuda
        for input in graph.inputs:
            assert "cuda" in str(input)
        # Ensure output tensor has cuda propagated
        assert " -> !mo.tensor<[batch, channels], f32, cuda:0>" in str(
            graph._mlir_op
        )


def test_elementwise_add_graph_with_device_prop_error() -> None:
    """Builds a simple graph with explicit device on inputs and checks for error conditions on output device propagation in the IR.
    """
    with Graph(
        "elementwise_add",
        input_types=[
            TensorType(
                dtype=DType.float32,
                shape=["batch", "channels"],
                device=Device.CUDA(0),
            ),
            TensorType(
                dtype=DType.float32,
                shape=["batch", "channels"],
                device=Device.CUDA(1),
            ),
        ],
    ) as graph:
        with pytest.raises(ValueError, match="Differing input devices"):
            graph.output(graph.inputs[0] + graph.inputs[1])


def test_transpose_graph_with_device_prop() -> None:
    """Builds a simple graph with an transpose and checks the IR."""
    with Graph(
        "elementwise_add",
        input_types=[
            TensorType(
                dtype=DType.float32,
                shape=["batch", "channels"],
                device=Device.CUDA(0),
            )
        ],
    ) as graph:
        graph.output(ops.transpose(graph.inputs[0], -1, -2))
        for input in graph.inputs:
            assert "cuda" in str(input)
        assert " -> !mo.tensor<[channels, batch], f32, cuda:0>" in str(
            graph._mlir_op
        )


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

        # Check that the arg/result name attributes were added.
        assert "argument_names = " in str(graph._mlir_op)
        assert "result_names = " in str(graph._mlir_op)


def test_add_op_closure() -> None:
    """Uses a closure to build a simple graph with an elementwise addition
    and checks the IR.
    """

    def elementwise_add(lhs: TensorValue, rhs: TensorValue) -> TensorValue:
        return ops.add(lhs, rhs)

    input_type = TensorType(dtype=DType.float32, shape=["batch", "channels"])
    add_graph = Graph("add", elementwise_add, (input_type, input_type))

    assert "rmo.add" in str(add_graph._mlir_op)
    assert "mo.output" in str(add_graph._mlir_op)


def test_unique_symbolic_dim() -> None:
    """Test that unique_symbolic_dim works, even if the counter is reset."""
    graph = Graph("dim_tester", input_types=[TensorType(DType.float32, (50,))])

    def use_dim(dim: Dim) -> None:
        with graph:
            ops.rebind(
                ops.reshape(graph.inputs[0], (-1,)), (dim,), "dim mismatch"
            )

    dim = graph.unique_symbolic_dim("foo")
    use_dim(dim)
    assert dim.name == "unique_foo_0"
    dim = graph.unique_symbolic_dim("bar")
    use_dim(dim)
    assert dim.name == "unique_bar_1"
    # Pretend we forgot the counter.
    assert graph._unique_symbolic_dim_counter == 2
    graph._unique_symbolic_dim_counter = 0
    dim = graph.unique_symbolic_dim("foo")
    use_dim(dim)
    assert dim.name == "unique_foo_1"
