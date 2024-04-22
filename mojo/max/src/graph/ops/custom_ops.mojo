# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
"""Helpers for building custom ops."""

from max.graph._attributes import _string_attr
from max.graph.symbol import SymbolTuple
from max.graph.type import AnyMOType, TypeTuple


def custom[
    name: StringLiteral
](values: SymbolTuple, out_type: AnyMOType) -> Symbol:
    """Creates a node to execute a custom graph operation in the graph.

    The custom op should be registered by annotating a function with
    the [`max.register.op`](/engine/reference/mojo/register/register/op)
    decorator.

    Parameters:
        name: The op name provided to `max.register.op`.

    Args:
        values: The op function's arguments.
        out_type: The op function's return type.

    Returns:
        A symbolic value representing the output of the op in the graph.
    """
    return custom_nv[name](values, TypeTuple(out_type))[0]


# We'll be able to make this an overload once we can get rid of `TypeTuple`.
def custom_nv[
    name: StringLiteral
](values: SymbolTuple, out_types: TypeTuple) -> SymbolTuple:
    """Creates a node to execute a custom graph operation in the graph.

    The custom op should be registered by annotating a function with
    the [`max.register.op`](/engine/reference/mojo/register/register/op)
    decorator.

    Parameters:
        name: The op name provided to `max.register.op`.

    Args:
        values: The op function's arguments.
        out_types: The op function's return type.

    Returns:
        Symbolic values representing the outputs of the op in the graph.
        These correspond 1:1 with the types passed as `out_types`.
    """
    var g = values[0].graph()
    var symbol_attr = _string_attr(g._context(), "symbol", name)
    return g.nvop("mo.custom", values, out_types, symbol_attr)
