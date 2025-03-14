# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
"""Test while loop operations."""

from max.dtype import DType
from max.graph import Graph, TensorType, ops


def test_while_loop_basic() -> None:
    """Test basic while loop functionality."""
    with Graph(
        "while_loop_basic", input_types=[TensorType(DType.int32, [])]
    ) as graph:
        x = graph.inputs[0]

        def pred(x):
            return x < 10

        def body(x):
            return x + 1

        results = ops.while_loop(x, pred, body)
        graph.output(results[0])

    # Verify MLIR contains while op and expected structure
    mlir_str = str(graph)
    assert "mo.while" in mlir_str
    assert " do " in mlir_str


def test_while_loop_multiple_args() -> None:
    """Test while loop with multiple arguments."""
    with Graph(
        "while_loop_multiple_args",
        input_types=[TensorType(DType.int32, []), TensorType(DType.int32, [])],
    ) as graph:
        x, y = graph.inputs

        def pred(x, y):
            return x < 10 and y < 10

        def body(x, y):
            return [x + 1, y + 1]

        results = ops.while_loop((x, y), pred, body)
        graph.output(results[0], results[1])

    # Verify MLIR contains while op with multiple args
    mlir_str = str(graph)
    assert "mo.while" in mlir_str
    assert " do " in mlir_str


def test_while_loop_empty_init() -> None:
    """Test while loop with empty initial values raises error."""
    with Graph("while_loop_empty_init", input_types=()) as graph:
        try:
            ops.while_loop([], lambda: True, lambda: [])
        except ValueError as e:
            assert "While loops must have at least one iteration value" in str(
                e
            )


def test_while_loop_type_check() -> None:
    """Test type checking in while loop."""
    with Graph(
        "while_loop_type_check", input_types=[TensorType(DType.int32, [])]
    ) as graph:
        x = graph.inputs[0]

        def pred(x):
            return x < 10

        def body(x):
            # Return wrong type
            return ops.cast(x + 1, DType.float32)

        try:
            ops.while_loop(x, pred, body)
        except TypeError as e:
            assert "Results don't match expected types" in str(e)
