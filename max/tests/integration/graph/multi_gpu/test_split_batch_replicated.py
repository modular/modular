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
"""Integration tests for ``split_batch_replicated`` under mixed DP + TP."""

from __future__ import annotations

from typing import Any, cast

import numpy as np
import pytest
from max.driver import CPU, Accelerator, Buffer, accelerator_count
from max.dtype import DType
from max.engine import InferenceSession
from max.graph import DeviceRef, Graph, TensorType, Type, ops
from max.nn import Signals
from max.nn.data_parallelism import split_batch_replicated

HIDDEN = 64


def _session(num_gpus: int) -> tuple[InferenceSession, CPU, list[Accelerator]]:
    host = CPU()
    devices = [Accelerator(n) for n in range(num_gpus)]
    return InferenceSession(devices=[host, *devices]), host, devices


def _input_types(
    signals: Signals, rows: int, offsets_len: int, dp_degree: int
) -> list[Type[Any]]:
    """One replicated copy of the batch and its row offsets per device."""
    cpu = DeviceRef.CPU()
    return cast(
        list[Type[Any]],
        [
            TensorType(dtype=DType.float32, shape=[rows, HIDDEN], device=device)
            for device in signals.devices
        ]
        + [
            TensorType(dtype=DType.uint32, shape=[offsets_len], device=device)
            for device in signals.devices
        ]
        + [
            TensorType(dtype=DType.int64, shape=[offsets_len], device=cpu),
            TensorType(dtype=DType.int64, shape=[dp_degree + 1], device=cpu),
        ]
        + signals.input_types(),
    )


@pytest.mark.parametrize(
    ("num_gpus", "group_size", "split_points"),
    [
        (4, 2, [0, 1, 3]),
        (8, 4, [0, 1, 3]),
        (8, 4, [0, 3, 3]),
        (8, 4, [0, 0, 3]),
    ],
    ids=["4x2", "8x4", "8x4-last-replica-empty", "8x4-first-replica-empty"],
)
def test_group_size_matches_leader_broadcast(
    num_gpus: int, group_size: int, split_points: list[int]
) -> None:
    """A grouped split equals splitting on leaders and broadcasting out.

    The two must agree elementwise on every device, including when a replica
    is handed no requests and its slice has zero rows.
    """
    if num_gpus > accelerator_count():
        pytest.skip(f"Needs {num_gpus} GPUs.")
    signals = Signals(devices=[DeviceRef.GPU(id=i) for i in range(num_gpus)])
    dp_degree = num_gpus // group_size
    rows, offsets_len = 96, 4
    leaders = [i * group_size for i in range(dp_degree)]
    assert len(split_points) == dp_degree + 1

    types = _input_types(signals, rows, offsets_len, dp_degree)
    with Graph("split_batch_equivalence", input_types=types) as graph:
        xs = [v.tensor for v in graph.inputs[:num_gpus]]
        offs = [v.tensor for v in graph.inputs[num_gpus : 2 * num_gpus]]
        offs_i64 = graph.inputs[2 * num_gpus].tensor
        splits = graph.inputs[2 * num_gpus + 1].tensor
        buffers = [v.buffer for v in graph.inputs[2 * num_gpus + 2 :]]

        # Grouped: every device cuts its own replica's range.
        grouped_x, grouped_off = split_batch_replicated(
            signals.devices,
            xs,
            offs,
            offs_i64,
            splits,
            prefix="grouped",
            group_size=group_size,
        )

        # Leader split, then broadcast each replica's slice to its peers.
        leader_x, leader_off = split_batch_replicated(
            [signals.devices[i] for i in leaders],
            [xs[i] for i in leaders],
            [offs[i] for i in leaders],
            offs_i64,
            splits,
            prefix="leader",
        )
        bcast_x: list[Any] = []
        bcast_off: list[Any] = []
        for replica in range(dp_degree):
            lo, hi = replica * group_size, (replica + 1) * group_size
            bcast_x.extend(
                ops.distributed_broadcast(leader_x[replica], buffers)[lo:hi]
            )
            bcast_off.extend(
                ops.distributed_broadcast(leader_off[replica], buffers)[lo:hi]
            )

        graph.output(*grouped_x, *grouped_off, *bcast_x, *bcast_off)

    session, host, devices = _session(num_gpus)
    compiled = session.load(graph)

    batch = np.arange(rows * HIDDEN, dtype=np.float32).reshape(rows, HIDDEN)
    # Three requests of 32 rows each; `split_points` assigns them to replicas,
    # and a replica whose range is empty gets a zero-row slice.
    offsets = np.array([0, 32, 64, 96], dtype=np.uint32)
    splits_np = np.array(split_points, dtype=np.int64)

    results = compiled.execute(
        *[Buffer.from_numpy(batch).to(d) for d in devices],
        *[Buffer.from_numpy(offsets).to(d) for d in devices],
        Buffer.from_numpy(offsets.astype(np.int64)),
        Buffer.from_numpy(splits_np),
        *signals.buffers(),
    )

    assert len(results) == 4 * num_gpus
    for i in range(num_gpus):
        np.testing.assert_array_equal(
            results[i].to(host).to_numpy(),
            results[2 * num_gpus + i].to(host).to_numpy(),
            err_msg=f"hidden states differ on device {i}",
        )
        np.testing.assert_array_equal(
            results[num_gpus + i].to(host).to_numpy(),
            results[3 * num_gpus + i].to(host).to_numpy(),
            err_msg=f"row offsets differ on device {i}",
        )


@pytest.mark.parametrize(
    ("group_size", "match"),
    [(0, "at least 1"), (3, "evenly divide")],
    ids=["below-one", "indivisible"],
)
def test_rejects_unusable_group_size(group_size: int, match: str) -> None:
    """A group size that cannot tile the devices is refused, not rounded."""
    devices = [DeviceRef.GPU(id=i) for i in range(4)]
    cpu = DeviceRef.CPU()
    types = cast(
        list[Type[Any]],
        [
            TensorType(dtype=DType.float32, shape=[8, HIDDEN], device=d)
            for d in devices
        ]
        + [TensorType(dtype=DType.uint32, shape=[3], device=d) for d in devices]
        + [
            TensorType(dtype=DType.int64, shape=[3], device=cpu),
            TensorType(dtype=DType.int64, shape=[3], device=cpu),
        ],
    )
    with Graph("split_batch_group_size_guard", input_types=types) as graph:
        with pytest.raises(ValueError, match=match):
            split_batch_replicated(
                devices,
                [v.tensor for v in graph.inputs[:4]],
                [v.tensor for v in graph.inputs[4:8]],
                graph.inputs[8].tensor,
                graph.inputs[9].tensor,
                group_size=group_size,
            )
