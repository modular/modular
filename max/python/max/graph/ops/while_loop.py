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
"""Op implementation for while loop."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import Any

from max._core import Value as _Value
from max.mlir.dialects import mo

from ..graph import Graph, GraphBlock
from ..value import BufferValue, TensorValue, Value, _ChainValue
from .support import as_iterable, as_values


def while_loop(
    initial_values: Iterable[Value[Any]] | Value[Any],
    predicate: Callable[..., TensorValue],
    body: Callable[..., Value[Any] | Iterable[Value[Any]]],
) -> list[TensorValue]:
    """Execute a loop until the predicate evaluates to false.

    Both the predicate and body functions must take in as arguments the same
    number and types of values as specified in the init_args. The predication
    function must return only a boolean scalar tensor of type :attr:`~max.dtype.DType.bool`.
    The body function must return a list of values matching the types of init_args,
    (or may return a value directly if there is only one).

    The following example demonstrates a basic while loop with a single argument:

    .. code-block:: python

        from max.graph import Graph, ops
        from max.dtype import DType

        with Graph("while_loop_example") as g:
            x = ops.constant(0, dtype=DType.int32, device=DeviceRef.CPU())

            def pred(x):
                return x < 10

            def body(x):
                return x + 1

            result = ops.while_loop(x, pred, body)
            print(result)

    The following example shows a while loop with multiple arguments:

    .. code-block:: python

        from max.graph import Graph, ops
        from max.dtype import DType

        with Graph("while_loop_example") as g:
            x = ops.constant(0, dtype=DType.int32, device=DeviceRef.CPU())
            y = ops.constant(5, dtype=DType.int32, device=DeviceRef.CPU())

            def pred(x, y):
                return ops.logical_and(x < 10, y < 15)

            def body(x, y):
                return [x + 1, y + 1]

            results = ops.while_loop((x, y), pred, body)
            print(results)

    Args:
        initial_values:
            Initial values for loop arguments. Must be non-empty.

        predicate:
            Callable that takes loop arguments and returns a boolean scalar tensor
            of type :attr:`~max.dtype.DType.bool` determining loop continuation.

        body:
            Callable that takes loop arguments and returns updated values matching
            the types of init_args.

    Returns:
        List of output values from the final loop iteration.

    Raises:
        ValueError: If init_args is empty.
        NotImplementedError: If any init_arg is a :class:`~max.graph.BufferValue`.

    Note:
        Buffer operations are currently not supported.
    """
    initial_values = (
        list(initial_values)
        if isinstance(initial_values, Iterable)
        else [initial_values]
    )
    if not initial_values:
        raise ValueError("While loops must have at least one iteration value.")
    if any(isinstance(arg, BufferValue) for arg in initial_values):
        raise NotImplementedError(
            "Buffer operations are currently not supported in while loops"
        )

    num_initial_values = len(initial_values)
    graph = Graph.current

    # Loop-carried values are: the user's initial values, then the global
    # execution chain, then per-device chains. The same shape is what the
    # cond/body blocks receive as block arguments and what they yield.
    carried = [
        *initial_values,
        graph._current_chain,
        *graph.device_chains.ordered_values(),
    ]
    carried_types = [v.type for v in carried]

    def adopt_block_args(args: Iterable[Any]) -> list[Value[Any]]:
        """Rebind graph chain state from the block's arguments.

        Returns the user-visible loop variables.
        """
        all_args = [Value.from_mlir(_Value._from_cmlir(a)) for a in args]
        loop_vars = all_args[:num_initial_values]
        execution_chain = all_args[num_initial_values]
        device_chains = all_args[num_initial_values + 1 :]
        assert len(device_chains) == len(graph.device_chains)

        graph._update_chain(execution_chain)
        for i, device in enumerate(graph.device_chains):
            new_chain = device_chains[i]
            assert isinstance(new_chain, _ChainValue)
            graph.device_chains[device] = new_chain
        return loop_vars

    def while_condition_terminator(args: list[Any]):  # noqa: ANN202
        """Build ``mo.while.condition`` from a flat ``[cond, *yielded]`` arg list.

        ``mo.while.condition`` takes its first operand as the condition and
        the rest as yielded values, but ``GraphBlock.output`` passes
        everything as one flat list.
        """
        condition, *yielded = args
        return mo.WhileConditionOp(condition, yielded)

    with GraphBlock(arg_types=carried_types) as pred_block:
        loop_vars = adopt_block_args(pred_block.arguments)
        condition = predicate(*loop_vars)
        if not condition.device.is_cpu():
            raise ValueError(
                "The predicate for `ops.while_loop` must reside on CPU,"
                f" but got a tensor on {condition.device}. Transfer it"
                " explicitly with `ops.transfer_to(pred, CPU())`."
            )
        pred_block.output(
            condition, *loop_vars, terminator=while_condition_terminator
        )

    user_carry_types = [v.type for v in initial_values]
    with GraphBlock(arg_types=carried_types) as body_block:
        loop_vars = adopt_block_args(body_block.arguments)
        body_block.output(
            *as_values(as_iterable(body(*loop_vars)), user_carry_types)
        )

    # The body's output types are what mo.while yields (loop vars + chains);
    # the condition block's first operand is the bool, so its output_types
    # would be wrong here.
    results = graph._add_op(
        mo.while_,
        body_block.output_types,
        carried,
        cond_block=pred_block.mlir_block,
        body_block=body_block.mlir_block,
    )

    user_results, out_chain, device_chains = (
        results[:num_initial_values],
        results[num_initial_values],
        results[num_initial_values + 1 :],
    )
    graph._update_chain(out_chain)
    for i, device in enumerate(graph.device_chains):
        new_chain = device_chains[i]
        assert isinstance(new_chain, _ChainValue)
        graph.device_chains[device] = new_chain

    return user_results
