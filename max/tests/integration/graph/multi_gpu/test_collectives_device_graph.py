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
"""The distributed collectives on two GPUs, through device-graph synthesis.

The graph compiler expands `mo.distributed.allreduce.sum`, `.broadcast` and
`.allgather` in an `is_device_graph` graph into one bundled op per device,
each reading its peers through stable address slots. These are the end-to-end
checks for that expansion: the same graph that the ordinary lowering runs,
compiled with `is_device_graph`, must produce the same values, and must keep
producing them when the recorded graph replays.

Each device's input is distinct, so a dropped or doubled peer shows up in the
result rather than cancelling out.

The grouped allreduce runs on four GPUs: every bundled copy receives the
whole world and picks its group from `peer_rank` and `group_size`, so a copy
outside the first group that indexed its peers by the wrong rank would read
the wrong buffers or none at all.
"""

from __future__ import annotations

import os
from collections.abc import Callable, Sequence
from pathlib import Path

import numpy as np
import pytest
from max.driver import CPU, Accelerator, Buffer, Device, accelerator_count
from max.dtype import DType
from max.engine import InferenceSession
from max.graph import (
    BufferType,
    BufferValue,
    DeviceRef,
    Graph,
    TensorType,
    TensorValue,
    ops,
)
from max.nn.legacy import Signals

NUM_ELEMENTS = 4
DEVICES = [DeviceRef.GPU(0), DeviceRef.GPU(1)]

GROUP_SIZE = 2
GROUPED_DEVICES = [DeviceRef.GPU(i) for i in range(4)]
# Sibling groups differ in shape, as per-DP-replica allreduces do, so a copy
# that sized itself from world index 0 instead of its own group would be off.
GROUPED_ELEMENTS = [
    NUM_ELEMENTS,
    NUM_ELEMENTS,
    NUM_ELEMENTS + 2,
    NUM_ELEMENTS + 2,
]

Collective = Callable[
    [Sequence[TensorValue], Sequence[BufferValue]], Sequence[TensorValue]
]


def _inputs(elements: Sequence[int] | None = None) -> list[np.ndarray]:
    """One distinct `[1, N_i]` input per device: device `i` holds
    `i * 10 + [0..N_i)`."""
    if elements is None:
        elements = [NUM_ELEMENTS] * len(DEVICES)
    return [
        (np.arange(n, dtype=np.float32) + 10 * i).reshape(1, -1)
        for i, n in enumerate(elements)
    ]


def _allreduce(
    tensors: Sequence[TensorValue], signals: Sequence[BufferValue]
) -> Sequence[TensorValue]:
    return ops.allreduce.sum(tensors, signals)


def _broadcast_from(root: int) -> Collective:
    def broadcast(
        tensors: Sequence[TensorValue], signals: Sequence[BufferValue]
    ) -> Sequence[TensorValue]:
        return ops.distributed_broadcast(tensors[root], signals)

    return broadcast


def _allgather(
    tensors: Sequence[TensorValue], signals: Sequence[BufferValue]
) -> Sequence[TensorValue]:
    # Along the last axis, the shape a column-parallel lm_head gathers in;
    # a leading-axis gather of 1-D pieces would lower the concatenation to a
    # buffer-level copy that device-graph finalization does not record.
    return ops.allgather(tensors, signals, axis=-1)


def _expected(name: str, inputs: Sequence[np.ndarray]) -> list[np.ndarray]:
    if name == "allreduce":
        total = np.sum(np.stack(list(inputs)), axis=0)
        return [total for _ in inputs]
    if name.startswith("broadcast_from_"):
        root = int(name.removeprefix("broadcast_from_"))
        return [inputs[root] for _ in inputs]
    if name == "allgather":
        gathered = np.concatenate(list(inputs), axis=-1)
        return [gathered for _ in inputs]
    raise ValueError(name)


COLLECTIVES: dict[str, Collective] = {
    "allreduce": _allreduce,
    "broadcast_from_0": _broadcast_from(0),
    "broadcast_from_1": _broadcast_from(1),
    "allgather": _allgather,
}


def _graph(
    name: str,
    collective: Collective,
    device_refs: Sequence[DeviceRef],
    elements: Sequence[int],
    signals: Signals,
    is_device_graph: bool,
) -> Graph:
    input_types: list[TensorType | BufferType] = [
        *(
            TensorType(DType.float32, [1, n], device=d)
            for d, n in zip(device_refs, elements, strict=True)
        ),
        *signals.input_types(),
    ]
    num_devices = len(device_refs)
    with Graph(
        f"{name}_{'device_graph' if is_device_graph else 'plain'}",
        input_types=input_types,
        is_device_graph=is_device_graph,
    ) as graph:
        # A relu per device gives every peer a real producer on its own
        # device, the shape the model graphs have, rather than a graph input.
        tensors = [ops.relu(graph.inputs[i].tensor) for i in range(num_devices)]
        signal_buffers = [
            graph.inputs[num_devices + i].buffer for i in range(num_devices)
        ]
        graph.output(*collective(tensors, signal_buffers))
        return graph


def _run(
    graph: Graph,
    device_refs: Sequence[DeviceRef],
    signals: Signals,
    inputs: Sequence[np.ndarray],
    expected: Sequence[np.ndarray],
) -> None:
    host = CPU()
    devices: list[Device] = [Accelerator(i) for i in range(len(device_refs))]
    session = InferenceSession(devices=[host, *devices])
    compiled = session.load(graph)

    device_inputs = [
        Buffer.from_numpy(x).to(device)
        for x, device in zip(inputs, devices, strict=True)
    ]

    # The second execution replays the recorded graph against the same
    # buffers; it has to read their contents again, not a stale copy.
    for _ in range(2):
        outputs = compiled.execute(*device_inputs, *signals.buffers())
        for i, (out, want) in enumerate(zip(outputs, expected, strict=True)):
            assert isinstance(out, Buffer)
            np.testing.assert_array_equal(
                out.to(host).to_numpy(), want, err_msg=f"device {i}"
            )


@pytest.mark.parametrize("is_device_graph", [False, True], ids=["plain", "dg"])
@pytest.mark.parametrize("name", sorted(COLLECTIVES))
def test_collective_two_gpus(name: str, is_device_graph: bool) -> None:
    if accelerator_count() < len(DEVICES):
        pytest.skip(f"Test requires {len(DEVICES)} GPUs")

    signals = Signals(DEVICES)
    elements = [NUM_ELEMENTS] * len(DEVICES)
    graph = _graph(
        name, COLLECTIVES[name], DEVICES, elements, signals, is_device_graph
    )
    inputs = _inputs(elements)
    _run(graph, DEVICES, signals, inputs, _expected(name, inputs))


@pytest.mark.parametrize("name", ["allgather", "allreduce", "broadcast_from_0"])
def test_rebuilt_rank_republishes_its_slot(name: str) -> None:
    """Device 1 replays while device 0 re-records, reading device 0 through
    its slot.

    Device 0's peer comes from `fill_from_host_scalar`, whose host input keys
    only device 0's graph, so a changed value re-records device 0 alone. The
    new recording allocates a new output for the fill, publishes its address
    into device 0's slot, and device 1's cached graph must read the new
    address there. A bundled kernel whose `build_graph` read the slot at
    record time instead would replay against device 0's first buffer and
    return the old value on device 1. Collectives in which device 1 reads
    nothing from device 0 are left out.
    """
    if accelerator_count() < len(DEVICES):
        pytest.skip(f"Test requires {len(DEVICES)} GPUs")

    signals = Signals(DEVICES)
    peer_type = TensorType(DType.float32, [1, NUM_ELEMENTS], device=DEVICES[0])
    with Graph(
        f"{name}_rebuilt_rank",
        input_types=[
            TensorType(DType.float32, [1], device=DeviceRef.CPU()),
            TensorType(DType.float32, [1, NUM_ELEMENTS], device=DEVICES[1]),
            *signals.input_types(),
        ],
        custom_extensions=[
            Path(os.environ["MODULAR_KERNEL_VERIFICATION_OPS_PATH"])
        ],
        is_device_graph=True,
    ) as graph:
        filled = ops.custom(
            "fill_from_host_scalar",
            device=DEVICES[0],
            values=[graph.inputs[0].tensor],
            out_types=[peer_type],
        )[0].tensor
        tensors = [filled, ops.relu(graph.inputs[1].tensor)]
        signal_buffers = [graph.inputs[2 + i].buffer for i in range(2)]
        graph.output(*COLLECTIVES[name](tensors, signal_buffers))

    host = CPU()
    session = InferenceSession(
        devices=[host, *(Accelerator(i) for i in range(len(DEVICES)))]
    )
    compiled = session.load(graph)

    x1 = _inputs()[1]
    device_x1 = Buffer.from_numpy(x1).to(Accelerator(1))

    # 3.0 records both devices, 9.0 re-records device 0 alone, and 3.0 again
    # returns to device 0's first graph, which must publish its own buffer
    # back into the slot.
    for value in (3.0, 9.0, 3.0):
        scalar = Buffer.from_numpy(np.array([value], dtype=np.float32))
        outputs = compiled.execute(scalar, device_x1, *signals.buffers())
        inputs = [np.full([1, NUM_ELEMENTS], value, dtype=np.float32), x1]
        expected = _expected(name, inputs)
        for i, (out, want) in enumerate(zip(outputs, expected, strict=True)):
            assert isinstance(out, Buffer)
            np.testing.assert_array_equal(
                out.to(host).to_numpy(),
                want,
                err_msg=f"device {i} with {value}",
            )


@pytest.mark.parametrize("is_device_graph", [False, True], ids=["plain", "dg"])
def test_grouped_allreduce_four_gpus(is_device_graph: bool) -> None:
    if accelerator_count() < len(GROUPED_DEVICES):
        pytest.skip(f"Test requires {len(GROUPED_DEVICES)} GPUs")

    def grouped_allreduce(
        tensors: Sequence[TensorValue], signals: Sequence[BufferValue]
    ) -> Sequence[TensorValue]:
        return ops.allreduce.sum(tensors, signals, group_size=GROUP_SIZE)

    signals = Signals(GROUPED_DEVICES)
    graph = _graph(
        "grouped_allreduce",
        grouped_allreduce,
        GROUPED_DEVICES,
        GROUPED_ELEMENTS,
        signals,
        is_device_graph,
    )
    inputs = _inputs(GROUPED_ELEMENTS)
    expected = [
        np.sum(np.stack(inputs[start : start + GROUP_SIZE]), axis=0)
        for start in range(0, len(inputs), GROUP_SIZE)
        for _ in range(GROUP_SIZE)
    ]
    _run(graph, GROUPED_DEVICES, signals, inputs, expected)
