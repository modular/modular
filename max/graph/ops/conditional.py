# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
"""Op implementation for conditional."""

from __future__ import annotations

from typing import Callable, Iterable

from max import mlir
from max.mlir.dialects import mo

from ..graph import Graph
from ..type import Type, _ChainType
from ..value import TensorValue, TensorValueLike


def cond(
    pred: TensorValueLike,
    out_types: Iterable[Type] | None,
    then_fn: Callable,
    else_fn: Callable,
) -> TensorValue:
    """Conditionally execute one of two branches based on a boolean predicate.

    Both branches must return the same number and types of values as specified
    in ``out_types``. Buffer mutations in branches are tracked automatically
    through the chain mechanism.

    Examples:

    1. Basic conditional with return values:

        .. code-block:: python

            def then_fn():
                return ops.constant(1, DType.int32)
            def else_fn():
                return ops.constant(0, DType.int32)

            result = ops.cond(
                pred,
                [TensorType(DType.int32, [])],
                then_fn,
                else_fn
            )

    2. Conditional with buffer mutations:

        .. code-block:: python

            def then_fn():
                ops.inplace_custom("increment", [buffer])
            def else_fn():
                ops.inplace_custom("decrement", [buffer])

            ops.cond(pred, None, then_fn, else_fn)

    ::
    Args:
        pred:
            Boolean scalar tensor of type :obj:`DType.bool` determining branch execution

        out_types:
            Expected output types for both branches. Use :obj:`None` for branches that don't return values

        then_fn:
            Callable executed when ``pred`` is True. Must return values matching ``out_types`` if ``out_types`` is not :obj:`None`

        else_fn:
            Callable executed when ``pred`` is False. Must return values matching ``out_types`` if ``out_types`` is not :obj:`None`

    Returns:
        List of output values from executed branch. Returns empty list when ``out_types``
        is :obj:`None`

    Raises:
        ValueError: If branches return different numbers of results or result types
                  don't match ``out_types``

    Note:
        Buffer operations in branches automatically update the global chain state to
        maintain mutation ordering constraints
    """
    pred = TensorValue(pred)
    out_types_actual = [t.to_mlir() for t in out_types or []] + [
        _ChainType().to_mlir()
    ]

    results, if_op = Graph.current._add_op_get_op_with_results(
        mo.if_, pred, out_types_actual
    )
    results_len = len(results)
    out_chain = results[results_len - 1]
    results = results[: results_len - 1]

    current_graph = Graph.current
    parent_chain = current_graph._current_chain

    def _handle_block(block: mlir.Block, block_fn: Callable, block_name: str):
        current_graph._update_chain(parent_chain)
        with current_graph._enter_block(block), current_graph._location():
            block_results = block_fn()
            if block_results is None:
                block_results = []
            elif not isinstance(block_results, Iterable):
                block_results = [block_results]

            if len(block_results) != len(results):
                raise ValueError(
                    f"{block_name} returned {len(block_results)} results, but "
                    f"expected {len(results)}"
                )
            for idx, (block_result, result) in enumerate(
                zip(block_results, results)
            ):
                if block_result.type != result.type:
                    raise ValueError(
                        f"{block_name} returned result of type {block_result.type} at index {idx}, but "
                        f"expected {result.type}"
                    )

            _ = current_graph._add_op(
                mo.YieldOp,
                block_results + [current_graph._current_chain],
            )

    _handle_block(if_op.thenRegion.blocks[0], then_fn, "then_block")
    _handle_block(if_op.elseRegion.blocks[0], else_fn, "else_block")

    Graph.current._update_chain(out_chain)
    return results
