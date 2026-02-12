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
"""Op implementation for distributed broadcast.

TODO(SERVOPT-943): Graduate to native MLIR op (mo.DistributedBroadcastOp) to
align with allreduce and allgather. Currently uses inplace_custom for faster
iteration.
"""

from __future__ import annotations

from collections.abc import Iterable

from ..graph import Graph
from ..type import TensorType
from ..value import BufferValue, BufferValueLike, TensorValue, TensorValueLike
from .custom import inplace_custom


def _buffer_values(values: Iterable[BufferValueLike]) -> list[BufferValue]:
    return [BufferValue(v) for v in values]


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

    # TODO(GEX-3209): This is a temporary workaround to avoid a hang
    # 8xB200 for `nvidia/llama-3.1-405b-instruct-nvfp4`.
    # Without this, we see a hang in the model execution when allreduce
    # is followed by broadcast.
    graph = Graph.current
    merged_chain = graph._merge_chains(
        [graph.device_chains[d] for d in devices]
    )
    for d in devices:
        graph.device_chains[d] = merged_chain

    # Per-device execution model:
    # Create one broadcast op per target device.
    # Each op reads the input tensor (from root) and writes to output on its device.
    results = []
    for device in devices:
        out_type = TensorType(
            dtype=input.dtype,
            shape=input.shape,
            device=device,
        )
        # inplace_custom handles chain management internally
        device_results = inplace_custom(
            name="mo.distributed.broadcast",
            device=device,
            values=[input, *signal_buffers],
            out_types=[out_type],
            parameters={"root": root},
        )
        results.append(device_results[0].tensor)

    return results
