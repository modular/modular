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

"""Tests `ops.allgather_rows_slice` against `ops.allgather` plus a row slice."""

from __future__ import annotations

from typing import Any, cast

import numpy as np
import pytest
from max.driver import CPU, Accelerator, Buffer, accelerator_count
from max.dtype import DType
from max.engine import InferenceSession
from max.graph import DeviceRef, Graph, TensorType, Type, ops
from max.nn.legacy import Signals


def _rows_slice_graph(
    signals: Signals, rows: int | str, cols: int, axis: int
) -> Graph:
    devices = signals.devices
    num_devices = len(devices)
    with Graph(
        "allgather_rows_slice",
        input_types=cast(
            list[Type[Any]],
            [
                TensorType(DType.float32, [rows, cols], device=device)
                for device in devices
            ]
            + signals.input_types(),
        ),
    ) as graph:
        shards = [v.tensor for v in graph.inputs[:num_devices]]
        buffers = [v.buffer for v in graph.inputs[num_devices:]]
        sliced = ops.allgather_rows_slice(shards, buffers, axis=axis)
        graph.output(*sliced)
        return graph


@pytest.mark.parametrize("num_gpus", [2, 4, 8])
@pytest.mark.parametrize("rows", [64, "rows"])
def test_allgather_rows_slice_execution(num_gpus: int, rows: int | str) -> None:
    """Each device gets its own row block of the gathered tensor, bit for bit.

    Covers a static and a symbolic row count; the symbolic one gives the
    outputs an algebraic ``rows // num_gpus`` dimension.
    """
    if num_gpus > accelerator_count():
        pytest.skip(f"Not enough GPUs to run with {num_gpus} GPUs.")

    cols = 1024
    signals = Signals(devices=[DeviceRef.GPU(id=i) for i in range(num_gpus)])
    graph = _rows_slice_graph(signals, rows, cols, axis=-1)
    host = CPU()
    devices = [Accelerator(n) for n in range(num_gpus)]
    session = InferenceSession(devices=[host, *devices])
    compiled = session.load(graph)

    num_rows = 64
    block = num_rows // num_gpus
    # Random bit patterns, NaNs and signed zeros included: the op only moves
    # data, so the output must match bit for bit.
    rng = np.random.default_rng(0)
    shards = [
        rng.integers(0, 2**32, size=(num_rows, cols), dtype=np.uint32).view(
            np.float32
        )
        for _ in range(num_gpus)
    ]
    inputs = [
        Buffer.from_numpy(shard).to(device)
        for shard, device in zip(shards, devices, strict=True)
    ]

    outputs = compiled.execute(*inputs, *signals.buffers())

    gathered = np.concatenate(shards, axis=-1)
    for d, output in enumerate(outputs):
        assert isinstance(output, Buffer)
        assert output.device == devices[d]
        expected = gathered[d * block : (d + 1) * block]
        got = output.to(host).to_numpy()
        assert got.shape == expected.shape
        assert np.array_equal(got.view(np.uint32), expected.view(np.uint32))


def test_allgather_rows_slice_rejects_unaligned_static_rows() -> None:
    """A static row count that the device count does not divide is refused."""
    signals = Signals(devices=[DeviceRef.GPU(id=i) for i in range(2)])
    with pytest.raises(ValueError, match="multiple of the number of devices"):
        _rows_slice_graph(signals, rows=5, cols=8, axis=-1)


def test_allgather_rows_slice_rejects_row_axis() -> None:
    """Concatenating along the split dimension is meaningless and refused."""
    signals = Signals(devices=[DeviceRef.GPU(id=i) for i in range(2)])
    with pytest.raises(ValueError, match="axis 0"):
        _rows_slice_graph(signals, rows=4, cols=8, axis=0)


def test_allgather_rows_slice_symbolic_rows_type() -> None:
    """The output row count is the symbolic row count over the device count."""
    signals = Signals(devices=[DeviceRef.GPU(id=i) for i in range(2)])
    graph = _rows_slice_graph(signals, rows="rows", cols=8, axis=-1)
    out = graph.output_types[0]
    assert isinstance(out, TensorType)
    assert str(out.shape[0]) == "rows // 2"
    assert int(out.shape[1]) == 16
