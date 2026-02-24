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
"""Test the ops.distributed_broadcast operation."""

import pytest
from max.dtype import DType
from max.graph import DeviceRef, Graph, TensorType, ops
from max.nn import Signals


def test_broadcast_basic() -> None:
    """Test basic broadcast from GPU 0."""
    devices = [
        DeviceRef.GPU(id=0),
        DeviceRef.GPU(id=1),
        DeviceRef.GPU(id=2),
        DeviceRef.GPU(id=3),
    ]
    signals = Signals(devices)

    with Graph(
        "broadcast",
        input_types=[
            TensorType(dtype=DType.float32, shape=[6, 5], device=devices[0]),
            *signals.input_types(),
        ],
    ) as graph:
        broadcast_outputs = ops.distributed_broadcast(
            input=graph.inputs[0].tensor,
            signal_buffers=(v.buffer for v in graph.inputs[1:]),
        )
        # graph.output() is not required for these assertions, but included
        # to demonstrate complete graph construction usage.
        graph.output(*broadcast_outputs)

        # Verify outputs have correct device assignment
        for output, device in zip(broadcast_outputs, devices, strict=True):
            assert device == output.device
            # Verify same shape and dtype as input
            assert output.shape == graph.inputs[0].tensor.shape
            assert output.dtype == graph.inputs[0].tensor.dtype


def test_broadcast_nonzero_root() -> None:
    """Test broadcast from non-zero root device (GPU 2)."""
    devices = [
        DeviceRef.GPU(id=0),
        DeviceRef.GPU(id=1),
        DeviceRef.GPU(id=2),
        DeviceRef.GPU(id=3),
    ]
    signals = Signals(devices)

    with Graph(
        "broadcast_root2",
        input_types=[
            # Input tensor on GPU 2 - root is inferred from this
            TensorType(dtype=DType.float32, shape=[6, 5], device=devices[2]),
            *signals.input_types(),
        ],
    ) as graph:
        broadcast_outputs = ops.distributed_broadcast(
            input=graph.inputs[0].tensor,
            signal_buffers=(v.buffer for v in graph.inputs[1:]),
        )

        # Verify outputs have correct device assignment
        for output, device in zip(broadcast_outputs, devices, strict=True):
            assert device == output.device
            assert output.shape == graph.inputs[0].tensor.shape
            assert output.dtype == graph.inputs[0].tensor.dtype


def test_broadcast_repeated_device() -> None:
    """Test error handling for duplicate GPU IDs."""
    devices = [
        DeviceRef.GPU(id=0),
        DeviceRef.GPU(id=0),  # Duplicate
        DeviceRef.GPU(id=2),
        DeviceRef.GPU(id=3),
    ]
    signals = Signals(devices)

    with pytest.raises(
        ValueError,
        match=r"distributed_broadcast requires unique devices",
    ):
        with Graph(
            "broadcast",
            input_types=[
                TensorType(
                    dtype=DType.float32, shape=[6, 5], device=devices[0]
                ),
                *signals.input_types(),
            ],
        ) as graph:
            ops.distributed_broadcast(
                input=graph.inputs[0].tensor,
                signal_buffers=(v.buffer for v in graph.inputs[1:]),
            )


def test_broadcast_device_not_found() -> None:
    """Test error when input tensor device is not in signal buffers."""
    devices = [
        DeviceRef.GPU(id=1),
        DeviceRef.GPU(id=2),
        DeviceRef.GPU(id=3),
    ]
    signals = Signals(devices)

    with pytest.raises(
        ValueError,
        match=r"input tensor device .* not found in signal buffer devices",
    ):
        with Graph(
            "broadcast",
            input_types=[
                # Input tensor on GPU 0, but signal buffers are for GPU 1,2,3
                TensorType(
                    dtype=DType.float32,
                    shape=[6, 5],
                    device=DeviceRef.GPU(id=0),
                ),
                *signals.input_types(),
            ],
        ) as graph:
            ops.distributed_broadcast(
                input=graph.inputs[0].tensor,
                signal_buffers=(v.buffer for v in graph.inputs[1:]),
            )


def test_broadcast_single_gpu() -> None:
    """Test that broadcast with single GPU is a no-op."""
    devices = [DeviceRef.GPU(id=0)]
    signals = Signals(devices)

    with Graph(
        "broadcast_single",
        input_types=[
            TensorType(dtype=DType.float32, shape=[6, 5], device=devices[0]),
            *signals.input_types(),
        ],
    ) as graph:
        broadcast_outputs = ops.distributed_broadcast(
            input=graph.inputs[0].tensor,
            signal_buffers=[graph.inputs[1].buffer],
        )
        # Single GPU broadcast returns the input unchanged
        assert len(broadcast_outputs) == 1
        assert broadcast_outputs[0]._mlir_value == graph.inputs[0]._mlir_value


def test_broadcast_empty() -> None:
    """Test that broadcast with no signal buffers returns empty list."""
    with Graph(
        "broadcast_empty",
        input_types=[
            TensorType(
                dtype=DType.float32, shape=[6, 5], device=DeviceRef.GPU(id=0)
            ),
        ],
    ) as graph:
        broadcast_outputs = ops.distributed_broadcast(
            input=graph.inputs[0].tensor,
            signal_buffers=[],
        )
        assert broadcast_outputs == []


def test_broadcast_different_dtypes() -> None:
    """Test broadcast with different data types."""
    devices = [
        DeviceRef.GPU(id=0),
        DeviceRef.GPU(id=1),
    ]
    signals = Signals(devices)

    for dtype in [DType.float16, DType.bfloat16, DType.float32]:
        with Graph(
            f"broadcast_{dtype}",
            input_types=[
                TensorType(dtype=dtype, shape=[4, 8], device=devices[0]),
                *signals.input_types(),
            ],
        ) as graph:
            broadcast_outputs = ops.distributed_broadcast(
                input=graph.inputs[0].tensor,
                signal_buffers=(v.buffer for v in graph.inputs[1:]),
            )
            for output in broadcast_outputs:
                assert output.dtype == dtype


def test_broadcast_different_shapes() -> None:
    """Test broadcast with different tensor shapes."""
    devices = [
        DeviceRef.GPU(id=0),
        DeviceRef.GPU(id=1),
    ]
    signals = Signals(devices)

    test_shapes = [
        [1024],
        [64, 128],
        [8, 16, 32],
        [2, 4, 8, 16],
    ]

    for shape in test_shapes:
        with Graph(
            f"broadcast_shape_{len(shape)}d",
            input_types=[
                TensorType(dtype=DType.float32, shape=shape, device=devices[0]),
                *signals.input_types(),
            ],
        ) as graph:
            broadcast_outputs = ops.distributed_broadcast(
                input=graph.inputs[0].tensor,
                signal_buffers=(v.buffer for v in graph.inputs[1:]),
            )
            for output in broadcast_outputs:
                assert list(output.shape) == shape
