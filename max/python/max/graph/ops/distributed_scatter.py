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
"""Op implementation for distributed scatter.

Distributes different data chunks from a root GPU to multiple device groups.
Each group (DP replica) gets a different chunk, and all devices within a group
(TP devices) get the same chunk via P2P pull.
"""

from __future__ import annotations

import math
from collections.abc import Iterable

from max._core.dialects import mo
from max._core.dialects.builtin import IntegerAttr, IntegerType

from ..graph import Graph
from ..type import TensorType, _ChainType
from ..value import BufferValueLike, TensorValue, TensorValueLike
from .utils import _buffer_values, _tensor_values


def distributed_scatter(
    input_chunks: Iterable[TensorValueLike],
    signal_buffers: Iterable[BufferValueLike],
) -> list[TensorValue]:
    """Scatter different chunks from root GPU to device groups.

    Each DP replica group receives a different input chunk. All TP devices
    within the same replica get the same chunk. Uses a pull-based approach
    where each GPU reads its chunk from the root GPU via P2P.

    Args:
        input_chunks: Input tensors to scatter, one per DP replica. All must
            reside on the same root device. The number of chunks determines
            ``dp_size``.
        signal_buffers: Device buffer values used for synchronization.
            The number of signal buffers determines the number of
            participating GPUs (``ngpus``).

    Returns:
        List of output tensors, one per device. Each output tensor has the
        same shape and dtype as its replica's input chunk.

    Raises:
        ValueError: If fewer than 2 signal buffers, if input chunks are not
            on the same device, or if devices are not unique.
    """
    input_chunks = _tensor_values(input_chunks)
    signal_buffers = _buffer_values(signal_buffers)
    dp_size = len(input_chunks)
    ngpus = len(signal_buffers)

    if ngpus < 2:
        raise ValueError(
            "distributed_scatter requires at least 2 devices "
            f"(signal_buffers). Got: {ngpus}"
        )

    if dp_size < 1:
        raise ValueError(
            "distributed_scatter requires at least 1 input chunk. "
            f"Got: {dp_size}"
        )

    # All input chunks must be on the same root device.
    root_device = input_chunks[0].device
    for i, chunk in enumerate(input_chunks):
        if chunk.device != root_device:
            raise ValueError(
                f"All input chunks must be on the same device. "
                f"Chunk 0 is on {root_device}, but chunk {i} is on "
                f"{chunk.device}"
            )

    devices = [buf.device for buf in signal_buffers]
    if len(set(devices)) < len(devices):
        raise ValueError(
            "distributed_scatter requires unique devices across signal "
            f"buffers. Got: {devices=}"
        )

    # Infer root from where the input chunks live.
    root = devices.index(root_device)

    tp_size = math.ceil(ngpus / dp_size)

    # Build ngpus-sized padded chunk list so every GPU sees all chunk sizes
    # and computes the same grid dimensions (avoiding barrier deadlocks).
    # padded_chunks[i] is the chunk that GPU i should read.
    padded_chunks = [
        input_chunks[min(i // tp_size, dp_size - 1)] for i in range(ngpus)
    ]

    # Build output types: each GPU gets its replica's chunk shape.
    out_types = [
        TensorType(
            dtype=padded_chunks[i].dtype,
            shape=padded_chunks[i].shape,
            device=device,
        )
        for i, device in enumerate(devices)
    ]

    graph = Graph.current

    # Merge all device chains into one input chain.
    in_chain = graph._merge_chains(
        [graph._current_chain, *(graph.device_chains[d] for d in devices)]
    )

    # Stage a single scatter op across all devices.
    root_attr = IntegerAttr(IntegerType(64), root)
    *results, out_chain = graph._add_op_generated(
        mo.DistributedScatterOp,
        out_types,
        _ChainType(),
        padded_chunks,
        signal_buffers,
        in_chain,
        root_attr,
    )

    # Update all chains.
    graph._update_chain(out_chain)
    for device in devices:
        graph.device_chains[device] = out_chain

    return [res.tensor for res in results]
