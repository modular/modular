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

from collections.abc import Callable, Sequence

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
