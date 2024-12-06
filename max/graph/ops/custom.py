# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
"""Operations for invoking user-defined operations."""

from __future__ import annotations

from typing import Iterable

from max.mlir import StringAttr
from max.mlir.dialects import mo

from ..graph import Graph
from ..type import Type, _ChainType
from ..value import BufferValue, Value, _OpaqueValue


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

    if any(isinstance(val, (BufferValue, _OpaqueValue)) for val in values):
        msg = (
            "custom ops that take buffers or opaque values to do in-place "
            "updates should use ops.inplace_custom instead"
        )
        raise TypeError(msg)

    return graph._add_op(
        mo.custom, [t.to_mlir() for t in out_types], values, symbol=symbol_attr
    )


def inplace_custom(
    name: str, values: Iterable[Value], out_types: Iterable[Type] | None = None
) -> list[Value]:
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
    if not any(isinstance(val, (BufferValue, _OpaqueValue)) for val in values):
        msg = (
            "expected at least one BufferValue or _OpaqueValue as input to an "
            "in-place custom op"
        )
        raise TypeError(msg)

    # Pass empty out_types if unspecified.
    out_mlir_types = [t.to_mlir() for t in out_types] if out_types else []

    graph = Graph.current
    current_chain = graph._current_chain

    *results, out_chain = graph._add_op(
        mo.custom,
        results_=[*out_mlir_types, _ChainType().to_mlir()],
        operands_=[*values, current_chain],
        symbol=StringAttr.get(name, graph._context),
    )
    graph._update_chain(out_chain._mlir_value)

    return results
