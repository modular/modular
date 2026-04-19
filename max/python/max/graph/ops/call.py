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
"""Op implementation for calling a graph."""

from __future__ import annotations

from typing import Any

from max.mlir.dialects import mo

from ..graph import Graph
from ..type import DeviceRef, _ChainType
from ..value import Value, _ChainValue


def call(graph: Graph, *args: Value[Any], prefix: str = "") -> list[Value[Any]]:
    """Calls a previously defined graph with the provided arguments.

    Use this function to invoke a subgraph built with
    :meth:`~max.graph.Graph.add_subgraph` or
    :meth:`~max.nn.Module.build_subgraph`. The primary benefit is that the
    compiler processes the subgraph definition once, which reduces compile
    time significantly for models with repeated blocks.

    Examples:
        Call a subgraph and forward its outputs to the parent graph:

        .. code-block:: python

            from max.dtype import DType
            from max.graph import Graph, ops
            from max.graph.type import TensorType, DeviceRef

            input_type = TensorType(DType.float32, [10], DeviceRef.CPU())

            with Graph("main", input_types=[input_type]) as graph:
                with graph.add_subgraph(
                    "add_one", input_types=[input_type]
                ) as sub:
                    x = sub.inputs[0].tensor
                    one = ops.constant(1, DType.float32, device=DeviceRef.CPU())
                    sub.output(ops.elementwise.add(x, one))

                result = ops.call(sub, graph.inputs[0])
                graph.output(*result)

        Call a shared subgraph for each layer of a model, resolving
        different weights at each call site with ``prefix``:

        .. code-block:: python

            # Build the subgraph once from the first layer.
            subgraph = self.layers[0].build_subgraph(
                "transformer_block",
                input_types=input_types,
                weight_prefix="layers.0.",
            )

            # Invoke it once per layer with layer-specific weights.
            for idx in range(num_layers):
                outputs = ops.call(
                    subgraph, *h, prefix=f"layers.{idx}."
                )

    Args:
        graph: The subgraph to call.
        *args: Arguments to pass to the subgraph. Must match the subgraph's
            input types, excluding the chain value (handled internally).
        prefix: A string prepended to all weight names when the subgraph is
            invoked. Use this to distinguish repeated calls to the same
            subgraph. For example, if a transformer block references a weight
            named ``attention.wq``, calling with ``prefix="layers.3."``
            resolves it to ``layers.3.attention.wq`` in the weights registry.
            Leave empty if the subgraph contains no placeholder weights.

    Returns:
        A list of :class:`~max.graph.Value` objects representing the
        subgraph's outputs, excluding any internal chain values.
    """
    # Get the current graph context
    current_graph = Graph.current
    call_args = list(args)  # mutable so we can add a chain
    # Be careful, input_types are type[Value], output_types are Type
    input_types = [type(input) for input in graph.inputs]
    output_types = list(graph.output_types)

    # Mostly leave type checking up to the op builder.
    # We can do some basic type checking to improve error messages,
    # but for instance can't check forward shape propagation correctness.
    if len(call_args) != len(input_types):
        raise ValueError(
            f"Expected {len(input_types)} args to call to {graph.name}, got {len(call_args)}. "
            f"\n    {graph.name}{tuple(input_types)}"
        )

    # Collect all device chains into the call args and output type.
    chain_devices: tuple[DeviceRef, ...] = ()
    chain_args: list[_ChainValue] = []
    if graph._has_chain_input:
        chain_devices = tuple(graph.device_chains)
        chain_args = [current_graph._current_chain]
        chain_args.extend(
            current_graph.device_chains[device] for device in chain_devices
        )
        output_types.extend(_ChainType() for _ in chain_args)
        call_args.extend(chain_args)

    # Add a call operation to the current graph
    call_results = current_graph._add_op(
        mo.call_,
        callee=graph.name,
        results=output_types,
        operands=call_args,
        prefix=prefix,
    )

    chain_result_count = len(chain_args)
    if not chain_result_count:
        return call_results

    # Update the device chains.
    chain_results = call_results[-chain_result_count:]
    current_graph._current_chain = _ChainValue(chain_results[0])
    current_graph.device_chains.update(
        (device, _ChainValue(chain_value))
        for device, chain_value in zip(
            chain_devices, chain_results[1:], strict=True
        )
    )
    return call_results[:-chain_result_count]
