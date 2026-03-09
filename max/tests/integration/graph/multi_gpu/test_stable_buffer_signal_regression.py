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

"""Regression tests for stable-address signal buffer layout.

These tests guard against a deadlock that occurred when stable-address ops
(allreduce, allgather, broadcast) were followed by old-input-scheme ops
(reducescatter, distributed_scatter) that share the same signal buffers.

Root cause: the old buffer layout [ stable inputs | comm buffers ] placed the
stable input copy at byte 0, overwriting the Signal struct that old-scheme ops
read for their barrier.  The fix [ comm buffers | stable inputs ] reserves the
lower half exclusively for communication so byte 0 is never clobbered.
"""

from __future__ import annotations

import numpy as np
import pytest
from max.driver import CPU, Accelerator, Buffer, accelerator_count
from max.dtype import DType
from max.engine import InferenceSession
from max.graph import DeviceRef, Graph, TensorType, ops
from max.nn import Signals


def test_allreduce_stable_followed_by_reducescatter() -> None:
    """Regression: allreduce (stable-address) must not corrupt signal buffers for reducescatter.

    Graph: allreduce.sum (stable-address scheme) → reducescatter.sum (old
    input scheme), both sharing the same signal buffers. Executed with plain
    model.execute() — no graph capture.

    In the old buffer layout ([ stable inputs | comm buffers ]) the allreduce
    stable-input copy wrote to byte 0, overwriting the Signal struct that the
    subsequent reducescatter barrier reads from — deadlock.

    The fix ([ comm buffers | stable inputs ]) puts stable inputs in the upper
    half so the Signal struct at byte 0 is untouched when reducescatter runs.
    """
    available_gpus = accelerator_count()
    if available_gpus < 2:
        pytest.skip("Test requires at least 2 GPUs")

    host = CPU()
    devices = [Accelerator(i) for i in range(available_gpus)]
    graph_devices = [DeviceRef.GPU(id=i) for i in range(available_gpus)]
    signals = Signals(devices=graph_devices)

    M, N = 64, 128
    input_types = [
        TensorType(dtype=DType.float32, shape=[M, N], device=graph_devices[i])
        for i in range(available_gpus)
    ]

    with Graph(
        "allreduce_stable_then_reducescatter",
        input_types=[*input_types, *signals.input_types()],
    ) as graph:
        signal_bufs = [inp.buffer for inp in graph.inputs[available_gpus:]]
        tensor_inputs = [inp.tensor for inp in graph.inputs[:available_gpus]]

        # First op: allreduce via the stable-address scheme.
        reduced = ops.allreduce.sum(tensor_inputs, signal_bufs)

        # Second op: reducescatter (old input scheme) on the same signal
        # buffers. Deadlocks if the first op's stable copy corrupted byte 0.
        scattered = ops.reducescatter.sum(reduced, signal_bufs, axis=0)

        graph.output(*scattered)

    session = InferenceSession(devices=[host, *devices])
    model = session.load(graph)

    input_data = np.ones((M, N), dtype=np.float32)
    input_tensors = [Buffer.from_numpy(input_data).to(d) for d in devices]
    signal_buffers = signals.buffers()

    # If the signal buffer layout is wrong this will deadlock.
    model.execute(*input_tensors, *signal_buffers)


def test_allgather_stable_followed_by_distributed_scatter() -> None:
    """Regression: allgather (stable-address) must not corrupt signal buffers for scatter.

    Graph: allgather (stable-address scheme) → distributed_scatter (old input
    scheme), both sharing the same signal buffers. Executed with plain
    model.execute() — no graph capture.

    In the old buffer layout ([ stable inputs | comm buffers ]) the allgather
    stable-input copy wrote to byte 0, overwriting the Signal struct that the
    subsequent scatter barrier reads from — deadlock.

    The fix ([ comm buffers | stable inputs ]) puts stable inputs in the upper
    half so the Signal struct at byte 0 is untouched when scatter runs.
    """
    available_gpus = accelerator_count()
    if available_gpus < 2:
        pytest.skip("Test requires at least 2 GPUs")

    host = CPU()
    devices = [Accelerator(i) for i in range(available_gpus)]
    graph_devices = [DeviceRef.GPU(id=i) for i in range(available_gpus)]
    signals = Signals(devices=graph_devices)

    M, N = 64, 128
    allgather_input_types = [
        TensorType(dtype=DType.float32, shape=[M, N], device=graph_devices[i])
        for i in range(available_gpus)
    ]
    # Single scatter chunk on GPU 0 (root); all GPUs receive it (tp_size=available_gpus).
    scatter_chunk_type = TensorType(
        dtype=DType.float32, shape=[M, N], device=graph_devices[0]
    )

    with Graph(
        "allgather_stable_then_distributed_scatter",
        input_types=[
            *allgather_input_types,
            scatter_chunk_type,
            *signals.input_types(),
        ],
    ) as graph:
        signal_bufs = [inp.buffer for inp in graph.inputs[available_gpus + 1 :]]
        allgather_inputs = [inp.tensor for inp in graph.inputs[:available_gpus]]
        scatter_chunk = graph.inputs[available_gpus].tensor

        # First op: allgather via the stable-address scheme.
        gathered = ops.allgather(allgather_inputs, signal_bufs, axis=0)

        # Second op: distributed_scatter (old input scheme) on the same signal
        # buffers. Deadlocks if the first op's stable copy corrupted byte 0.
        ops.distributed_scatter([scatter_chunk], signal_bufs)

        graph.output(*gathered)

    session = InferenceSession(devices=[host, *devices])
    model = session.load(graph)

    data = np.ones((M, N), dtype=np.float32)
    allgather_tensors = [Buffer.from_numpy(data).to(d) for d in devices]
    scatter_tensor = Buffer.from_numpy(data).to(devices[0])
    signal_buffers = signals.buffers()

    # If the signal buffer layout is wrong this will deadlock.
    model.execute(*allgather_tensors, scatter_tensor, *signal_buffers)


def test_broadcast_stable_followed_by_reducescatter() -> None:
    """Regression: broadcast (stable-address) must not corrupt signal buffers for reducescatter.

    Graph: distributed_broadcast (stable-address scheme) → reducescatter.sum
    (old input scheme), both sharing the same signal buffers. Executed with
    plain model.execute() — no graph capture.

    In the old buffer layout ([ stable inputs | comm buffers ]) the broadcast
    stable-input copy wrote to byte 0, overwriting the Signal struct that the
    subsequent reducescatter barrier reads from — deadlock.

    The fix ([ comm buffers | stable inputs ]) puts stable inputs in the upper
    half so the Signal struct at byte 0 is untouched when reducescatter runs.
    """
    available_gpus = accelerator_count()
    if available_gpus < 2:
        pytest.skip("Test requires at least 2 GPUs")

    host = CPU()
    devices = [Accelerator(i) for i in range(available_gpus)]
    graph_devices = [DeviceRef.GPU(id=i) for i in range(available_gpus)]
    signals = Signals(devices=graph_devices)

    M, N = 64, 128
    # Broadcast root is GPU 0; the single input tensor lives there.
    input_type = TensorType(
        dtype=DType.float32, shape=[M, N], device=graph_devices[0]
    )

    with Graph(
        "broadcast_stable_then_reducescatter",
        input_types=[input_type, *signals.input_types()],
    ) as graph:
        signal_bufs = [inp.buffer for inp in graph.inputs[1:]]

        # First op: broadcast from GPU 0 via the stable-address scheme.
        broadcasted = ops.distributed_broadcast(
            graph.inputs[0].tensor, signal_bufs
        )

        # Second op: reducescatter (old input scheme) on the same signal
        # buffers. Deadlocks if the first op's stable copy corrupted byte 0.
        scattered = ops.reducescatter.sum(broadcasted, signal_bufs, axis=0)

        graph.output(*scattered)

    session = InferenceSession(devices=[host, *devices])
    model = session.load(graph)

    input_data = np.ones((M, N), dtype=np.float32)
    input_tensor = Buffer.from_numpy(input_data).to(devices[0])
    signal_buffers = signals.buffers()

    # If the signal buffer layout is wrong this will deadlock.
    model.execute(input_tensor, *signal_buffers)
