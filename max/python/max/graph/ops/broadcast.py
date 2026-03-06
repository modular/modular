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
"""Op implementation for distributed broadcast."""

from __future__ import annotations

from collections.abc import Iterable

from max._core.dialects import mo
from max._core.dialects.builtin import IntegerAttr, IntegerType

from ..graph import Graph
from ..type import TensorType, _ChainType
from ..value import BufferValueLike, TensorValue, TensorValueLike
from .utils import _buffer_values


def distributed_broadcast(
    input: TensorValueLike,
    signal_buffers: Iterable[BufferValueLike],
) -> list[TensorValue]:
    """Broadcast tensor from source GPU to all GPUs.

    This op is a collective operation which broadcasts a tensor from the source
    GPU (where the input tensor resides) to all participating GPUs. Each GPU
    receives a copy of the input tensor.

    Args:
        input: Input tensor to broadcast. The device where this tensor resides
            becomes the root/source of the broadcast.
        signal_buffers: Device buffer values used for synchronization.
            The number of signal buffers determines the number of participating
            GPUs.

    Returns:
        List of output tensors, one per device. Each output tensor has the
        same shape and dtype as the input tensor.

    Raises:
        ValueError: If input tensor device is not found in signal buffer devices,
            if devices are not unique, or if there are fewer than 2 signal buffers.
    """
    input = TensorValue(input)
    signal_buffers = _buffer_values(signal_buffers)
    num_devices = len(signal_buffers)

    if num_devices < 2:
        # Single device or empty: no-op, return input as-is
        return [input] if num_devices == 1 else []

    # Get devices and infer root from input tensor's device
    devices = [buf.device for buf in signal_buffers]
    if input.device not in devices:
        raise ValueError(
            f"input tensor device {input.device} not found in signal buffer "
            f"devices: {devices}"
        )
    root = devices.index(input.device)

    # Validate all devices are unique
    if len(set(devices)) < len(devices):
        raise ValueError(
            "distributed_broadcast requires unique devices across signal "
            f"buffers. Got: {devices=}"
        )

    graph = Graph.current

    # Merge all device chains into one input chain.
    in_chain = graph._merge_chains(
        [graph._current_chain, *(graph.device_chains[d] for d in devices)]
    )

    # Output types: one tensor per device with same shape/dtype as input.
    output_types = [
        TensorType(dtype=input.dtype, shape=input.shape, device=device)
        for device in devices
    ]

    # Stage a single broadcast op across all devices.
    root_attr = IntegerAttr(IntegerType(64), root)
    *results, out_chain = graph._add_op_generated(
        mo.DistributedBroadcastOp,
        output_types,
        _ChainType(),
        input,
        signal_buffers,
        in_chain,
        root_attr,
    )

    # Update all chains.
    graph._update_chain(out_chain)
    for device in devices:
        graph.device_chains[device] = out_chain

    return [res.tensor for res in results]
