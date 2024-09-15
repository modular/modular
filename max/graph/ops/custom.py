# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
"""Operations for invoking user-defined operations."""

from max.graph.graph import Graph
from max.graph.type import Type
from max.graph.value import Value
from max.mlir import StringAttr
from max.mlir.dialects import mo


def custom(
    name: str, values: list[Value], out_types: list[Type]
) -> list[Value]:
    """Creates a node to execute a custom graph operation in the graph.

    The custom op should be registered by annotating a function with
    the [`max.register.op`](/max/api/mojo/register/register/op)
    decorator.

    Args:
        name: The op name provided to `max.register.op`.
        values: The op function's arguments.
        out_types: The list of op function's return type.

    Returns:
        Symbolic values representing the outputs of the op in the graph.
        These correspond 1:1 with the types passed as `out_types`.
    """
    graph = Graph.current
    symbol_attr = StringAttr.get(name, graph._context)
    return graph._add_op(
        mo.custom, [t.to_mlir() for t in out_types], values, symbol=symbol_attr
    )
