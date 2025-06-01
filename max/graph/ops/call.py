# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
"""Op implementation for calling a graph."""

from __future__ import annotations

from max.mlir.dialects import mo

from ..graph import Graph
from ..type import _ChainType
from ..value import Value


def call(graph: Graph, *args: Value, prefix: str = "") -> list[Value]:
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
        prefix: Prefix to add to the names of any weights in the subgraph

    Returns:
        Either a single Value or a list of Values representing the graph outputs
        (excluding the chain value which is handled internally)
    """
    # Get the current graph context
    current_graph = Graph.current
    call_args = list(args)  # mutable so we can add a chain
    # Be careful, input_types are type[Value], output_types are Type
    input_types = [type(input) for input in graph.inputs]
    output_types = [*graph.output_types, _ChainType()]

    # Mostly leave type checking up to the op builder.
    # We can do some basic type checking to improve error messages,
    # but for instance can't check forward shape propagation correctness.
    if len(call_args) != len(input_types):
        raise ValueError(
            f"Expected {len(input_types)} args to call to {graph.name}, got {len(call_args)}. "
            f"\n    {graph.name}{tuple(input_types)}"
        )

    call_args.append(current_graph._current_chain)

    # Add a call operation to the current graph
    call_results = current_graph._add_op(
        mo.call_,
        symbol=graph.name,
        results=output_types,
        operands=call_args,
        prefix=prefix,
    )

    *results, current_graph._current_chain = call_results
    return results
