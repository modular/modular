# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
"""Operations for invoking user-defined operations."""

from typing import Iterable

from max.mlir import StringAttr
from max.mlir.dialects import mo

from ..graph import Graph
from ..type import Type, _ChainType
from ..value import BufferValue, Value


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


def inplace_custom(name: str, values: Iterable[Value]) -> None:
    """Creates a node to execute an in-place custom graph operation in the graph.

    The custom op should be registered by annotating a function with
    the [`max.register.op`](/max/api/mojo/register/register/op)
    decorator.

    Args:
        name: The op name provided to `max.register.op`.
        values: The op function's arguments.
    """
    # Unfortunately there's no existing way to mark a particular NDBuffer input
    # as needing to be backed by a `mo.buffer` value at the graph level.
    #
    # This will change as the new kernel API matures and has support added for
    # marking particular inputs as mutable.
    #
    # Until that switch is made check that at least one input to the custom op
    # is a BufferValue to provide some level of safety.
    if not any(isinstance(val, BufferValue) for val in values):
        raise TypeError(
            "Expected at least one BufferValue as input to an in-place"
            " custom op"
        )

    graph = Graph.current
    current_chain = graph._current_chain
    values.append(current_chain)  # type: ignore

    out_chain = custom(name, values, [_ChainType()])[0]  # type: ignore
    graph._update_chain(out_chain._mlir_value)
