# ===----------------------------------------------------------------------=== #
# Copyright (c) 2026, Modular Inc. All rights reserved.
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

from ..graph import Graph, GraphBlock
from ..type import Type
from ..value import TensorValue, TensorValueLike, Value
from .support import as_iterable, as_values


def cond(
    pred: TensorValueLike,
    out_types: Iterable[Type[Any]] | None,
    then_fn: Callable[
        [],
        Iterable[Value[Any] | TensorValueLike]
        | Value[Any]
        | TensorValueLike
        | None,
    ],
    else_fn: Callable[
        [],
        Iterable[Value[Any] | TensorValueLike]
        | Value[Any]
        | TensorValueLike
        | None,
    ],
) -> list[TensorValue]:
    r"""Conditionally execute one of two branches based on a boolean predicate.

    This function provides conditional execution in the computation graph, where
    one of two branches is executed based on the runtime value of a boolean
    predicate. Both branches must return the same number and types of values as
    specified in ``out_types``. Buffer mutations in branches are tracked
    automatically through the chain mechanism.

    The predicate is evaluated at runtime to determine which branch to execute.
    Both branches are compiled but only the selected branch is executed based
    on the predicate value.

    Branch bodies are populated into :class:`GraphBlock`\ s before the
    enclosing ``mo.if`` op is created. If a branch body raises, the partial
    block never becomes visible to the surrounding verifier and no cleanup
    is needed.

    This example shows a basic conditional with return values:

    .. code-block:: python

        def then_fn():
            return ops.constant(1, DType.int32, device=DeviceRef.CPU())

        def else_fn():
            return ops.constant(0, DType.int32, device=DeviceRef.CPU())

        device = DeviceRef.CPU()
        pred = ops.constant(True, DType.bool, device=device)
        result = ops.cond(
            pred,
            [TensorType(DType.int32, [], device=device)],
            then_fn,
            else_fn
        )

    This example shows a conditional with buffer mutations, where branches
    don't return values:

    .. code-block:: python

        def then_fn():
            ops.inplace_custom("increment", device=buffer.device, values=[buffer])

        def else_fn():
            ops.inplace_custom("decrement", device=buffer.device, values=[buffer])

        ops.cond(pred, None, then_fn, else_fn)

    This example shows a conditional with multiple return values:

    .. code-block:: python

        def then_fn():
            a = ops.constant(1, DType.float32, device=device)
            b = ops.constant(2, DType.float32, device=device)
            return a, b

        def else_fn():
            a = ops.constant(0, DType.float32, device=device)
            b = ops.constant(-1, DType.float32, device=device)
            return a, b

        device = DeviceRef.CPU()
        out_types = [
            TensorType(DType.float32, [], device=device),
            TensorType(DType.float32, [], device=device)
        ]
        results = ops.cond(pred, out_types, then_fn, else_fn)

    Args:
        pred: Boolean scalar tensor of type :attr:`DType.bool` determining branch
            execution.
        out_types: Expected output types for both branches. Use :obj:`None` for
            branches that don't return values.
        then_fn: Callable executed when ``pred`` is True. Must return values
            matching ``out_types`` if ``out_types`` is not :obj:`None`.
        else_fn: Callable executed when ``pred`` is False. Must return values
            matching ``out_types`` if ``out_types`` is not :obj:`None`.

    Returns:
        List of output values from executed branch. Returns empty list when
        ``out_types`` is :obj:`None`.

    Raises:
        ValueError: If branches return different numbers of results or result
            types don't match ``out_types``.
    """
    pred = TensorValue(pred)
    if not pred.device.is_cpu():
        raise ValueError(
            "The predicate for `ops.cond` must reside on CPU, but got a"
            f" tensor on {pred.device}. Transfer it explicitly with"
            " `ops.transfer_to(pred, CPU())`."
        )

    out_types_list = list(out_types) if out_types is not None else []

    graph = Graph.current

    def _build_branch(block: GraphBlock, fn: Callable[..., Any]) -> None:
        # Snapshot the device set on entry so we can detect new device
        # chains introduced inside the branch and fold them into the host
        # chain before yielding — otherwise the yield's chain count would
        # exceed what the surrounding scope expects.
        live_on_entry = set(graph.device_chains)
        results = as_values(as_iterable(fn()), out_types_list)
        new_devices = set(graph.device_chains) - live_on_entry
        graph.device_chains.merge_for(new_devices)
        for device in new_devices:
            del graph.device_chains[device]
        block.output(*graph.device_chains.pack(results))

    with GraphBlock() as then_block:
        _build_branch(then_block, then_fn)
    with GraphBlock() as else_block:
        _build_branch(else_block, else_fn)

    # Both blocks are fully populated. Now create the mo.if op with them
    # attached; verification runs normally and recursively re-verifies the
    # yields against the real parent.
    results = graph._add_op(
        mo.if_,
        pred,
        then_block.output_types,
        then_block=then_block.mlir_block,
        else_block=else_block.mlir_block,
    )

    return graph.device_chains.unpack(results)
