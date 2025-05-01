# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
"""Op implementation for calling a graph."""

from __future__ import annotations

from collections.abc import Iterable

from max import mlir
from max.mlir.dialects import mo

from ..graph import Graph
from ..value import Value


def call(graph: Graph, *args: Value | mlir.Value) -> list[Value]:
    """Call a graph with the provided arguments and return its results.

    This function invokes a previously defined graph, passing in the provided
    arguments and the current chain value, and returns the results.

    The body of the graph is ultimately inlined into the caller, so the chain
    value is only used for serialization if the subgraph's body contains an
    operation that makes use of it in the first place.

    The current advantage of using subgraphs is that it offers a way to improve
    compile times for operations that are used repeatedly in a model. As a
    secondary benefit, it also makes the IR more readable by allowing control
    flow to be expressed in a more natural way.

    Args:
        graph: The graph to call
        *args: Arguments to pass to the called graph

    Returns:
        Either a single Value or a list of Values representing the graph outputs
        (excluding the chain value which is handled internally)
    """
    # Get the current graph context
    current_graph = Graph.current

    # Get the symbol name of the target graph
    symbol_name = graph.name

    # Extract MLIR operation details from the graph
    graph_mlir_op = graph._mlir_op
    function_type_attr = graph_mlir_op.attributes["functionType"]
    assert isinstance(function_type_attr, mlir.TypeAttr)
    function_type = function_type_attr.value
    assert isinstance(function_type, mlir.FunctionType)
    output_types = function_type.results

    # TODO: This is a hack to get the chain type that allows for equivalence testing with the function's type arguments.
    # Ideally, we should be able to use _ChainType().to_mlir() directly, but that returns `ChainType(!mo.chain)` which
    # is a max._core.dialects.mo.ChainType instance, not a mlir.Type instance.
    chain_type = Graph.current._current_chain._mlir_value.type

    has_chain_input = (
        len(function_type.inputs) > 0 and function_type.inputs[0] == chain_type
    )
    has_chain_output = (
        len(function_type.results) > 0
        and function_type.results[0] == chain_type
    )
    if has_chain_input != has_chain_output:
        raise ValueError(
            "Subgraphs with chain inputs must also have chain outputs, and vice versa"
        )

    arg_types: Iterable[mlir.Type] = map(
        lambda arg: arg._mlir_value.type
        if isinstance(arg, Value)
        else arg.type,
        args,
    )
    for idx, (arg_type, operand_type) in enumerate(
        zip(arg_types, function_type.inputs[int(has_chain_input) :])
    ):
        if arg_type != operand_type:
            raise ValueError(
                f"Argument {idx} type mismatch: expected {arg_type}, got {operand_type}"
            )

    # Add a call operation to the current graph
    call_results = current_graph._add_op(
        mo.call_,
        symbol=symbol_name,
        results=output_types,
        operands=([Graph.current._current_chain] if has_chain_input else [])
        + list(args),
    )

    # Extract the chain result and return the other results
    results_count = len(call_results)
    if has_chain_output:
        output_chain = call_results[0]  # First result is the chain
        # Update the current chain in the graph
        current_graph._current_chain = output_chain

    output_values = call_results[
        int(has_chain_output) : results_count
    ]  # Other results are the actual outputs

    # Return the actual outputs (excluding the chain)
    return output_values
