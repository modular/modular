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
"""Integration test for distributed scatter on multi-GPU."""

from __future__ import annotations

import math

import numpy as np
import pytest
from max.driver import CPU, Accelerator, Buffer, Device, accelerator_count
from max.dtype import DType
from max.engine import InferenceSession
from max.graph import DeviceRef, Graph, TensorType, ops
from max.nn import Signals


def scatter_graph(
    signals: Signals,
    chunk_sizes: list[int],
) -> Graph:
    """Build a graph that scatters dp_size chunks to ngpus devices.

    Args:
        signals: Signal buffers for all devices.
        chunk_sizes: Size of each DP replica's chunk.
    """
    devices = signals.devices
    dp_size = len(chunk_sizes)

    # All input chunks live on device 0.
    input_types = [
        TensorType(dtype=DType.uint32, shape=[size], device=devices[0])
        for size in chunk_sizes
    ]
    all_input_types = input_types + list(signals.input_types())

    with Graph(
        "scatter",
        input_types=all_input_types,
    ) as graph:
        input_chunks = [graph.inputs[i].tensor for i in range(dp_size)]
        signal_buffers = [inp.buffer for inp in graph.inputs[dp_size:]]

        scatter_outputs = ops.distributed_scatter(input_chunks, signal_buffers)
        graph.output(*scatter_outputs)
        return graph


@pytest.mark.parametrize(
    "num_gpus,dp_size",
    [
        (2, 1),  # DP=1, TP=2: all GPUs get same chunk (like broadcast)
        (2, 2),  # DP=2, TP=1: each GPU gets unique chunk
        (4, 2),  # DP=2, TP=2: pairs of GPUs share chunks
        (4, 4),  # DP=4, TP=1: each GPU gets unique chunk
    ],
)
def test_scatter_execution(num_gpus: int, dp_size: int) -> None:
    """Test scatter distributes correct chunks to correct GPUs."""
    if (available := accelerator_count()) < num_gpus:
        pytest.skip(f"Need {num_gpus} GPUs, have {available}")

    tp_size = math.ceil(num_gpus / dp_size)
    devices: list[Device] = [Accelerator(i) for i in range(num_gpus)]
    host = CPU()

    signals = Signals(devices=[DeviceRef.GPU(id=i) for i in range(num_gpus)])

    # Create chunks with distinct values per replica.
    # Chunk i contains [i*100+1, i*100+2, ..., i*100+5].
    chunk_size = 5
    chunks_np = [
        np.arange(i * 100 + 1, i * 100 + chunk_size + 1, dtype=np.uint32)
        for i in range(dp_size)
    ]

    graph = scatter_graph(signals, [chunk_size] * dp_size)

    session = InferenceSession(devices=[host] + devices)
    compiled = session.load(graph)

    # All input chunks on device 0.
    input_buffers = [Buffer.from_numpy(c).to(devices[0]) for c in chunks_np]

    outputs = compiled.execute(*input_buffers, *signals.buffers())

    # Verify each GPU got the right chunk.
    for gpu_idx in range(num_gpus):
        replica = min(gpu_idx // tp_size, dp_size - 1)
        expected = chunks_np[replica]
        result = outputs[gpu_idx]
        assert isinstance(result, Buffer)
        actual = result.to(host).to_numpy()
        np.testing.assert_array_equal(
            actual,
            expected,
            err_msg=(
                f"GPU {gpu_idx} (replica {replica}): "
                f"expected {expected}, got {actual}"
            ),
        )
