# ===----------------------------------------------------------------------=== #
# Copyright (c) 2025, Modular Inc. All rights reserved.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions:
# https://llvm.org/LICENSE.txt
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ===----------------------------------------------------------------------=== #
"""Op implementation for conditional."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import Any

from max.mlir.dialects import mo

from ..graph import Graph
from ..type import DeviceRef, Type, _ChainType
from ..value import TensorValue, TensorValueLike, Value, _ChainValue


def cond(
    pred: TensorValueLike,
    out_types: Iterable[Type[Any]] | None,
    then_fn: Callable[[], Iterable[Value[Any]] | Value[Any] | None],
    else_fn: Callable[[], Iterable[Value[Any]] | Value[Any] | None],
) -> list[TensorValue]:
    """Conditionally execute one of two branches based on a boolean predicate.

    Both branches must return the same number and types of values as specified
    in ``out_types``. Buffer mutations in branches are tracked automatically
    through the chain mechanism.

    Examples:

    1. Basic conditional with return values:

        .. code-block:: python

            def then_fn():
                return ops.constant(1, DType.int32, device=DeviceRef.CPU())
            def else_fn():
                return ops.constant(0, DType.int32, device=DeviceRef.CPU())

            result = ops.cond(
                pred,
                [TensorType(DType.int32, [], device=device)],
                then_fn,
                else_fn
            )

    2. Conditional with buffer mutations:

        .. code-block:: python

            def then_fn():
                ops.inplace_custom("increment", device=buffer.device, values=[buffer])
            def else_fn():
                ops.inplace_custom("decrement", device=buffer.device, values=[buffer])

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
    pred = pred.to(DeviceRef.CPU())
    out_types_actual = [
        *(t.to_mlir() for t in out_types or []),
        _ChainType().to_mlir(),
        *(_ChainType().to_mlir() for _ in Graph.current.device_chains),
    ]

    # Pause verification until the operation is fully constructed
    with Graph.current._pause_verification():
        results, if_op = Graph.current._add_op_get_op_with_results(
            mo.if_, pred, out_types_actual
        )

    num_values = len(list(out_types)) if out_types is not None else 0

    results, out_chain, device_chains = (
        results[:num_values],
        results[num_values],
        results[num_values + 1 :],
    )

    def wrap_region_func(  # noqa: ANN202
        user_func,  # noqa: ANN001
    ):
        # Capture the set of live devices before and after constructing the
        # body region. If new per-device chains were introduced, merge them
        # back into the global chain (guaranteed to exist).
        def handle_chains(*args, **kwargs):  # noqa: ANN202
            live_devices_on_entry = dict.fromkeys(Graph.current.device_chains)

            results = user_func(*args, **kwargs)

            live_devices_on_exit = tuple(Graph.current.device_chains)

            for device in live_devices_on_exit:
                if device in live_devices_on_entry:
                    continue

                # Merge this chain with the global chain
                Graph.current._merge_chains(
                    [
                        Graph.current._current_chain,
                        Graph.current.device_chains[device],
                    ]
                )

                # Remove the new device chain from the map
                del Graph.current.device_chains[device]

            return results

        return handle_chains

    try:
        Graph.current._build_block(
            if_op.thenRegion.blocks[0],
            wrap_region_func(then_fn),
            mo.YieldOp,
            "then_block",
            out_types,
        )

        Graph.current._build_block(
            if_op.elseRegion.blocks[0],
            wrap_region_func(else_fn),
            mo.YieldOp,
            "else_block",
            out_types,
        )

        Graph.current._update_chain(out_chain)
        for i, device in enumerate(Graph.current.device_chains):
            new_chain = device_chains[i]
            assert isinstance(new_chain, _ChainValue)
            Graph.current.device_chains[device] = new_chain

        Graph.current._verify_op(if_op)
        return results
    except Exception as e:
        if_op.erase()
        raise e
