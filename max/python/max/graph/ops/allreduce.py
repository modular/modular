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
"""Op implementation for allreduce."""

from __future__ import annotations

from collections.abc import Iterable

from max._core.dialects import mo

from ..graph import Graph
from ..type import _ChainType
from ..value import BufferValueLike, TensorValue, TensorValueLike
from .utils import _buffer_values, _tensor_values


def sum(
    inputs: Iterable[TensorValueLike], signal_buffers: Iterable[BufferValueLike]
) -> list[TensorValue]:
    """Collective allreduce summation operation.

    This op is a collective op which takes in tensors from different devices and
    outputs tensors on different devices.
    In particular, this operation will gather the inputs across different
    devices and reduce them via a summation operation.
    The result is then broadcasted back to the same devices that the inputs
    came from.

    Args:
        inputs: The input tensors to reduce.
        signal_buffers: Device buffer values used for synchronization.

    Returns:
        An iterable outputs which all hold the reduction output.
    """
    inputs = _tensor_values(inputs)
    signal_buffers = _buffer_values(signal_buffers)
    if len(inputs) != len(signal_buffers):
        raise ValueError(
            f"expected number of inputs ({len(inputs)}) and number of "
            f"signal buffers ({len(signal_buffers)}) to match"
        )

    devices = [input.device for input in inputs]

    if not all(input.shape == inputs[0].shape for input in inputs[1:]):
        raise ValueError(
            "allreduce.sum operation must have the same shape across all"
            f" input tensors. Got: {inputs=}"
        )
    if len(set(devices)) < len(devices):
        raise ValueError(
            "allreduce.sum operation must have unique devices across its input "
            f"tensors. Got: {devices=}"
        )

    graph = Graph.current

    # Merge all device chains into one input chain.
    in_chain = graph._merge_chains(
        [graph._current_chain, *(graph.device_chains[d] for d in devices)]
    )

    # Stage a single allreduce op across all devices.
    *results, out_chain = graph._add_op_generated(
        mo.DistributedAllreduceSumOp,
        [inp.type for inp in inputs],
        _ChainType(),
        inputs,
        signal_buffers,
        in_chain,
    )

    # Update all chains.
    graph._update_chain(out_chain)
    for device in devices:
        graph.device_chains[device] = out_chain

    return [res.tensor for res in results]
