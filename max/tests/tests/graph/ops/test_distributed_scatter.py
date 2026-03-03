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
"""Test the ops.distributed_scatter operation (graph construction only)."""

import pytest
from max.dtype import DType
from max.graph import DeviceRef, Graph, TensorType, ops
from max.nn import Signals


def test_scatter_basic() -> None:
    """Test basic scatter with DP=2, TP=2 (4 GPUs)."""
    devices = [DeviceRef.GPU(id=i) for i in range(4)]
    signals = Signals(devices)

    with Graph(
        "scatter",
        input_types=[
            # 2 input chunks on GPU 0 (one per DP replica)
            TensorType(dtype=DType.uint32, shape=[5], device=devices[0]),
            TensorType(dtype=DType.uint32, shape=[5], device=devices[0]),
            *signals.input_types(),
        ],
    ) as graph:
        input_chunks = [graph.inputs[0].tensor, graph.inputs[1].tensor]
        signal_buffers = [inp.buffer for inp in graph.inputs[2:]]

        scatter_outputs = ops.distributed_scatter(input_chunks, signal_buffers)
        graph.output(*scatter_outputs)

        # 4 outputs, one per GPU
        assert len(scatter_outputs) == 4
        for output, device in zip(scatter_outputs, devices, strict=True):
            assert output.device == device
            assert output.dtype == DType.uint32


def test_scatter_dp4_tp1() -> None:
    """Test scatter with DP=4, TP=1 (4 GPUs, each gets unique chunk)."""
    devices = [DeviceRef.GPU(id=i) for i in range(4)]
    signals = Signals(devices)

    with Graph(
        "scatter_dp4",
        input_types=[
            TensorType(dtype=DType.uint32, shape=[3], device=devices[0]),
            TensorType(dtype=DType.uint32, shape=[3], device=devices[0]),
            TensorType(dtype=DType.uint32, shape=[3], device=devices[0]),
            TensorType(dtype=DType.uint32, shape=[3], device=devices[0]),
            *signals.input_types(),
        ],
    ) as graph:
        input_chunks = [graph.inputs[i].tensor for i in range(4)]
        signal_buffers = [inp.buffer for inp in graph.inputs[4:]]

        scatter_outputs = ops.distributed_scatter(input_chunks, signal_buffers)

        assert len(scatter_outputs) == 4
        for output, device in zip(scatter_outputs, devices, strict=True):
            assert output.device == device
            assert list(output.shape) == [3]


def test_scatter_different_chunk_sizes() -> None:
    """Test scatter where chunks have different sizes."""
    devices = [DeviceRef.GPU(id=i) for i in range(4)]
    signals = Signals(devices)

    with Graph(
        "scatter_varied",
        input_types=[
            TensorType(dtype=DType.uint32, shape=[5], device=devices[0]),
            TensorType(dtype=DType.uint32, shape=[3], device=devices[0]),
            *signals.input_types(),
        ],
    ) as graph:
        input_chunks = [graph.inputs[0].tensor, graph.inputs[1].tensor]
        signal_buffers = [inp.buffer for inp in graph.inputs[2:]]

        scatter_outputs = ops.distributed_scatter(input_chunks, signal_buffers)

        assert len(scatter_outputs) == 4
        # GPUs 0,1 (replica 0) get shape [5]
        assert list(scatter_outputs[0].shape) == [5]
        assert list(scatter_outputs[1].shape) == [5]
        # GPUs 2,3 (replica 1) get shape [3]
        assert list(scatter_outputs[2].shape) == [3]
        assert list(scatter_outputs[3].shape) == [3]


def test_scatter_repeated_device() -> None:
    """Test error handling for duplicate GPU IDs."""
    devices = [
        DeviceRef.GPU(id=0),
        DeviceRef.GPU(id=0),  # Duplicate
    ]
    signals = Signals(devices)

    with pytest.raises(
        ValueError,
        match=r"distributed_scatter requires unique devices",
    ):
        with Graph(
            "scatter",
            input_types=[
                TensorType(dtype=DType.uint32, shape=[5], device=devices[0]),
                *signals.input_types(),
            ],
        ) as graph:
            ops.distributed_scatter(
                [graph.inputs[0].tensor],
                [inp.buffer for inp in graph.inputs[1:]],
            )


def test_scatter_chunks_on_different_devices() -> None:
    """Test error when input chunks are on different devices."""
    devices = [DeviceRef.GPU(id=i) for i in range(4)]
    signals = Signals(devices)

    with pytest.raises(
        ValueError,
        match=r"All input chunks must be on the same device",
    ):
        with Graph(
            "scatter",
            input_types=[
                TensorType(dtype=DType.uint32, shape=[5], device=devices[0]),
                TensorType(dtype=DType.uint32, shape=[5], device=devices[1]),
                *signals.input_types(),
            ],
        ) as graph:
            ops.distributed_scatter(
                [graph.inputs[0].tensor, graph.inputs[1].tensor],
                [inp.buffer for inp in graph.inputs[2:]],
            )


def test_scatter_too_few_signal_buffers() -> None:
    """Test error with fewer than 2 signal buffers."""
    devices = [DeviceRef.GPU(id=0)]
    signals = Signals(devices)

    with pytest.raises(
        ValueError,
        match=r"distributed_scatter requires at least 2 devices",
    ):
        with Graph(
            "scatter",
            input_types=[
                TensorType(dtype=DType.uint32, shape=[5], device=devices[0]),
                *signals.input_types(),
            ],
        ) as graph:
            ops.distributed_scatter(
                [graph.inputs[0].tensor],
                [graph.inputs[1].buffer],
            )
