# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

"""Shared utilities for normalization layer tests."""

from __future__ import annotations

from max.graph import Graph
from max.mlir import OpView, StringAttr


def find_ops_in_graph(
    graph: Graph, op_name: str, custom_symbol: str | None = None
) -> list[OpView]:
    """Find operations in a graph's MLIR representation.

    Args:
        graph: The Graph object to search
        op_name: The name of the operation (e.g., "mo.custom", "mo.layer_norm")
        custom_symbol: For "mo.custom" ops, the symbol to search for in attributes

    Returns:
        List of operations matching the criteria
    """
    ops: list[OpView] = []
    for op in graph._mlir_op.regions[0].blocks[0].operations:
        if op.name == op_name:
            if custom_symbol and op_name == "mo.custom":
                if custom_symbol in StringAttr(op.attributes["symbol"]).value:
                    ops.append(op)
            else:
                ops.append(op)
    return ops


def assert_single_op(
    graph: Graph, op_name: str, custom_symbol: str | None = None
) -> OpView:
    """Assert that exactly one op of the given type exists in the graph.

    Args:
        graph: The Graph object to search
        op_name: The name of the operation
        custom_symbol: For "mo.custom" ops, the symbol to search for

    Returns:
        The single operation found
    """
    ops = find_ops_in_graph(graph, op_name, custom_symbol)
    assert len(ops) == 1, f"Expected 1 {op_name} op, found {len(ops)}"
    return ops[0]


def assert_op_output_shape(op: OpView, expected_shape_str: str) -> None:
    """Assert that an op has a single output with the expected shape.

    Args:
        op: The operation to check
        expected_shape_str: The expected shape string (e.g., "[2, 10, 64]")
    """
    assert len(op.results) == 1, f"Expected 1 result, found {len(op.results)}"
    assert expected_shape_str in str(op.results[0].type), (
        f"Expected shape {expected_shape_str} not found in {op.results[0].type}"
    )


# Common test shapes for normalization layers
COMMON_NORM_TEST_SHAPES: list[tuple[tuple[int, ...], int]] = [
    ((10,), 10),
    ((5, 10), 10),
    ((2, 5, 10), 10),
    ((1, 2, 5, 10), 10),
]
