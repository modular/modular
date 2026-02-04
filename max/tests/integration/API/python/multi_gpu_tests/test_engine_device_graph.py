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

"""Test CUDA device graph capture and replay with multiple GPUs.

NOTE: These tests are configured to run only on NVIDIA GPUs via the
multi-gpu-device-graph-tests BUILD target. CUDA graph capture is currently
only fully supported on NVIDIA hardware. AMD ROCm/HIP has known limitations
with graph capture, particularly when updating input buffers via inplace_copy_from.
"""

from __future__ import annotations

import numpy as np
import pytest
from max.driver import CPU, Accelerator, Buffer, accelerator_count
from max.dtype import DType
from max.engine import InferenceSession
from max.graph import DeviceRef, Graph, TensorType


def test_multi_device_graph_capture_replay() -> None:
    """Test that graph capture and replay work across multiple GPUs."""
    available_gpus = accelerator_count()
    if available_gpus < 2:
        pytest.skip("Test requires at least 2 GPUs")

    # Use 2 GPUs for this test
    host = CPU()
    device0 = Accelerator(0)
    device1 = Accelerator(1)

    # Create a graph that performs operations on both GPUs
    input_type0 = TensorType(
        dtype=DType.float32, shape=[4], device=DeviceRef.GPU(0)
    )
    input_type1 = TensorType(
        dtype=DType.float32, shape=[4], device=DeviceRef.GPU(1)
    )

    with Graph(
        "multi_device_capture", input_types=[input_type0, input_type1]
    ) as graph:
        # Add 1 to each input on their respective devices
        result0 = graph.inputs[0].tensor + 1
        result1 = graph.inputs[1].tensor + 2
        graph.output(result0, result1)

    session = InferenceSession(devices=[host, device0, device1])
    model = session.load(graph)

    # Create input tensors on each device
    input0 = Buffer.from_numpy(np.arange(4, dtype=np.float32)).to(device0)
    input1 = Buffer.from_numpy(np.arange(4, dtype=np.float32) + 10).to(device1)

    # Execute baseline to get expected results
    baseline0, baseline1 = model.execute(input0, input1)
    np.testing.assert_allclose(
        baseline0.to_numpy(), np.arange(4, dtype=np.float32) + 1
    )
    np.testing.assert_allclose(
        baseline1.to_numpy(), np.arange(4, dtype=np.float32) + 12
    )

    # Capture the graph
    captured_output0, captured_output1 = model.capture(input0, input1)

    # Replay and verify outputs match expected values
    model.replay(input0, input1)
    np.testing.assert_allclose(
        captured_output0.to_numpy(), np.arange(4, dtype=np.float32) + 1
    )
    np.testing.assert_allclose(
        captured_output1.to_numpy(), np.arange(4, dtype=np.float32) + 12
    )

    # Replay again to ensure it's stable
    model.replay(input0, input1)
    np.testing.assert_allclose(
        captured_output0.to_numpy(), np.arange(4, dtype=np.float32) + 1
    )
    np.testing.assert_allclose(
        captured_output1.to_numpy(), np.arange(4, dtype=np.float32) + 12
    )


def test_multi_device_graph_capture_with_updated_inputs() -> None:
    """Test that replaying with updated inputs produces correct results."""
    available_gpus = accelerator_count()
    if available_gpus < 2:
        pytest.skip("Test requires at least 2 GPUs")

    host = CPU()
    device0 = Accelerator(0)
    device1 = Accelerator(1)

    # Create a graph with simple operations
    input_type0 = TensorType(
        dtype=DType.float32, shape=[4], device=DeviceRef.GPU(0)
    )
    input_type1 = TensorType(
        dtype=DType.float32, shape=[4], device=DeviceRef.GPU(1)
    )

    with Graph(
        "multi_device_updated_inputs", input_types=[input_type0, input_type1]
    ) as graph:
        result0 = graph.inputs[0].tensor * 2
        result1 = graph.inputs[1].tensor * 3
        graph.output(result0, result1)

    session = InferenceSession(devices=[host, device0, device1])
    model = session.load(graph)

    # Create initial input tensors
    input0 = Buffer.from_numpy(np.ones(4, dtype=np.float32)).to(device0)
    input1 = Buffer.from_numpy(np.ones(4, dtype=np.float32) * 2).to(device1)

    # Capture with initial values
    captured_output0, captured_output1 = model.capture(input0, input1)
    model.replay(input0, input1)

    # Verify initial outputs
    np.testing.assert_allclose(
        captured_output0.to_numpy(), np.ones(4, dtype=np.float32) * 2
    )
    np.testing.assert_allclose(
        captured_output1.to_numpy(), np.ones(4, dtype=np.float32) * 6
    )

    # Update inputs in-place
    updated_values0 = Buffer.from_numpy(np.arange(4, dtype=np.float32) + 5).to(
        device0
    )
    updated_values1 = Buffer.from_numpy(np.arange(4, dtype=np.float32) + 10).to(
        device1
    )

    input0.inplace_copy_from(updated_values0)
    input1.inplace_copy_from(updated_values1)

    # Replay with updated inputs
    model.replay(input0, input1)

    # Verify outputs reflect updated inputs
    np.testing.assert_allclose(
        captured_output0.to_numpy(), (np.arange(4, dtype=np.float32) + 5) * 2
    )
    np.testing.assert_allclose(
        captured_output1.to_numpy(), (np.arange(4, dtype=np.float32) + 10) * 3
    )


def test_multi_device_graph_independent_operations() -> None:
    """Test graph capture with independent operations on multiple GPUs.

    This test verifies that each GPU can have its own captured operations
    without cross-device data movement during the captured execution.
    """
    available_gpus = accelerator_count()
    if available_gpus < 2:
        pytest.skip("Test requires at least 2 GPUs")

    host = CPU()
    device0 = Accelerator(0)
    device1 = Accelerator(1)

    # Create a graph with independent operations on each GPU
    input_type0 = TensorType(
        dtype=DType.float32, shape=[8], device=DeviceRef.GPU(0)
    )
    input_type1 = TensorType(
        dtype=DType.float32, shape=[8], device=DeviceRef.GPU(1)
    )

    with Graph(
        "multi_device_independent", input_types=[input_type0, input_type1]
    ) as graph:
        # Independent operations on each GPU
        result0 = graph.inputs[0].tensor * 2 + 5
        result1 = graph.inputs[1].tensor * 3 + 10
        graph.output(result0, result1)

    session = InferenceSession(devices=[host, device0, device1])
    model = session.load(graph)

    # Create input tensors
    input0 = Buffer.from_numpy(np.arange(8, dtype=np.float32)).to(device0)
    input1 = Buffer.from_numpy(np.arange(8, dtype=np.float32) + 10).to(device1)

    # Execute baseline
    baseline0, baseline1 = model.execute(input0, input1)
    expected0 = np.arange(8, dtype=np.float32) * 2 + 5
    expected1 = (np.arange(8, dtype=np.float32) + 10) * 3 + 10

    np.testing.assert_allclose(baseline0.to_numpy(), expected0)
    np.testing.assert_allclose(baseline1.to_numpy(), expected1)

    # Capture the graph
    captured_output0, captured_output1 = model.capture(input0, input1)

    # Replay and verify
    model.replay(input0, input1)
    np.testing.assert_allclose(captured_output0.to_numpy(), expected0)
    np.testing.assert_allclose(captured_output1.to_numpy(), expected1)

    # Update inputs and replay
    new_input0 = Buffer.from_numpy(np.ones(8, dtype=np.float32) * 7).to(device0)
    new_input1 = Buffer.from_numpy(np.ones(8, dtype=np.float32) * 4).to(device1)
    input0.inplace_copy_from(new_input0)
    input1.inplace_copy_from(new_input1)

    model.replay(input0, input1)
    expected0_updated = np.ones(8, dtype=np.float32) * 7 * 2 + 5
    expected1_updated = np.ones(8, dtype=np.float32) * 4 * 3 + 10

    np.testing.assert_allclose(captured_output0.to_numpy(), expected0_updated)
    np.testing.assert_allclose(captured_output1.to_numpy(), expected1_updated)


def test_multi_device_multiple_replay_cycles() -> None:
    """Test multiple capture and replay cycles on multiple GPUs.

    This verifies that captured graphs can be replayed multiple times
    with different input values and produce correct outputs each time.
    """
    available_gpus = accelerator_count()
    if available_gpus < 2:
        pytest.skip("Test requires at least 2 GPUs")

    host = CPU()
    device0 = Accelerator(0)
    device1 = Accelerator(1)

    # Create a graph that operates on both GPUs
    input_type0 = TensorType(
        dtype=DType.float32, shape=[4], device=DeviceRef.GPU(0)
    )
    input_type1 = TensorType(
        dtype=DType.float32, shape=[4], device=DeviceRef.GPU(1)
    )

    with Graph(
        "multi_device_replay_cycles",
        input_types=[input_type0, input_type1],
    ) as graph:
        result0 = graph.inputs[0].tensor + 10
        result1 = graph.inputs[1].tensor + 20
        graph.output(result0, result1)

    session = InferenceSession(devices=[host, device0, device1])
    model = session.load(graph)

    # Create input buffers
    input0 = Buffer.from_numpy(
        np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    ).to(device0)
    input1 = Buffer.from_numpy(
        np.array([5.0, 6.0, 7.0, 8.0], dtype=np.float32)
    ).to(device1)

    # Capture the graph
    captured0, captured1 = model.capture(input0, input1)

    # First replay cycle
    model.replay(input0, input1)
    np.testing.assert_allclose(
        captured0.to_numpy(),
        np.array([11.0, 12.0, 13.0, 14.0], dtype=np.float32),
    )
    np.testing.assert_allclose(
        captured1.to_numpy(),
        np.array([25.0, 26.0, 27.0, 28.0], dtype=np.float32),
    )

    # Update inputs and replay again
    input0.inplace_copy_from(
        Buffer.from_numpy(
            np.array([10.0, 20.0, 30.0, 40.0], dtype=np.float32)
        ).to(device0)
    )
    input1.inplace_copy_from(
        Buffer.from_numpy(
            np.array([50.0, 60.0, 70.0, 80.0], dtype=np.float32)
        ).to(device1)
    )

    model.replay(input0, input1)
    np.testing.assert_allclose(
        captured0.to_numpy(),
        np.array([20.0, 30.0, 40.0, 50.0], dtype=np.float32),
    )
    np.testing.assert_allclose(
        captured1.to_numpy(),
        np.array([70.0, 80.0, 90.0, 100.0], dtype=np.float32),
    )

    # Third replay cycle with different values
    input0.inplace_copy_from(
        Buffer.from_numpy(np.ones(4, dtype=np.float32) * 5).to(device0)
    )
    input1.inplace_copy_from(
        Buffer.from_numpy(np.ones(4, dtype=np.float32) * 15).to(device1)
    )

    model.replay(input0, input1)
    np.testing.assert_allclose(
        captured0.to_numpy(), np.ones(4, dtype=np.float32) * 15
    )
    np.testing.assert_allclose(
        captured1.to_numpy(), np.ones(4, dtype=np.float32) * 35
    )
