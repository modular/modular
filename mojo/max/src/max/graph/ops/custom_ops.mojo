# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
"""Helpers for building custom ops."""

from collections import List

from max.graph._attributes import _string_attr
from max.graph.type import Type


def custom[name: StringLiteral](value: Symbol, out_type: Type) -> Symbol:
    """Creates a node to execute a custom graph operation in the graph.

    The custom op should be registered by annotating a function with the
    [`@compiler.register`](/max/api/mojo-decorators/compiler-register/)
    decorator.

    Parameters:
        name: The op name provided to `@compiler.register`.

    Args:
        value: The op function's argument.
        out_type: The op function's return type.

    Returns:
        A symbolic value representing the output of the op in the graph.
    """
    return custom[name](List(value), List(out_type))[0]


def custom[name: StringLiteral](values: List[Symbol], out_type: Type) -> Symbol:
    """Creates a node to execute a custom graph operation in the graph.

    The custom op should be registered by annotating a function with the
    [`@compiler.register`](/max/api/mojo-decorators/compiler-register/)
    decorator.

    Parameters:
        name: The op name provided to `@compiler.register`.

    Args:
        values: The op function's arguments.
        out_type: The op function's return type.

    Returns:
        A symbolic value representing the output of the op in the graph.
    """
    return custom[name](values, List[Type](out_type))[0]


def custom[
    name: StringLiteral
](values: List[Symbol], out_types: List[Type]) -> List[Symbol]:
    """Creates a node to execute a custom graph operation in the graph.

    The custom op should be registered by annotating a function with the
    [`@compiler.register`](/max/api/mojo-decorators/compiler-register/)
    decorator.

    Parameters:
        name: The op name provided to `@compiler.register`.

    Args:
        values: The op function's arguments.
        out_types: The list of op function's return type.

    Returns:
        Symbolic values representing the outputs of the op in the graph.
        These correspond 1:1 with the types passed as `out_types`.
    """
    var g = values[0].graph()
    var symbol_attr = _string_attr(g._context(), "symbol", name)
    return g.nvop("mo.custom", values, out_types, List(symbol_attr))
