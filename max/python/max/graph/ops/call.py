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
"""Op implementation for calling a graph."""

from __future__ import annotations

from typing import Any

from max.mlir.dialects import mo

from ..graph import Graph
from ..type import DeviceRef, _ChainType
from ..value import Value, _ChainValue


def call(graph: Graph, *args: Value[Any], prefix: str = "") -> list[Value[Any]]:
    """Call a graph with the provided arguments and return its results.

    This function invokes a previously defined graph, passing in the provided
    arguments and the current chain value, and returns the results.

    The body of the graph is ultimately inlined into the caller, so the chain
    value is only used for serialization if the subgraph's body contains an
    operation that makes use of it in the first place.

    The current advantage of using subgraphs is that it offers a way to improve
    compile times for operations that are used repeatedly in a model. As a
    secondary benefit, it also makes the IR more readable by allowing control
    flow to be expressed in a more natural way.

    Args:
        graph: The graph to call
        *args: Arguments to pass to the called graph
        prefix: Prefix to add to the names of any weights in the subgraph

    Returns:
        Either a single Value or a list of Values representing the graph outputs
        (excluding the chain value which is handled internally)
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
        symbol=graph.name,
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
