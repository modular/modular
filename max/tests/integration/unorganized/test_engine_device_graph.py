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

from __future__ import annotations

from collections.abc import Generator

import numpy as np
import pytest
import torch
from max.driver import Accelerator, Buffer, accelerator_count
from max.dtype import DType
from max.engine import InferenceSession
from max.graph import DeviceRef, Graph, TensorType
from max.nn.legacy import Linear


@pytest.fixture(autouse=True)
def clean_up_gpus() -> Generator[None, None, None]:
    """Call synchronize after each test on all accelerators.

    GPU failures for a particular device can spill over to later tests,
    incorrectly reporting the source of the error. This fixture synchronizes
    all accelerators after each test, which will propagate any pending errors
    up to the Python level.
    """

    yield

    for i in range(accelerator_count()):
        accelerator = Accelerator(i)
        accelerator.synchronize()


def test_execution_trace_capture_replay() -> None:
    if accelerator_count() == 0:
        pytest.skip("GPU not available")

    accelerator = Accelerator()
    session = InferenceSession(devices=[accelerator])

    with Graph(
        "execution_trace_capture",
        input_types=[TensorType(DType.float32, [4], device=DeviceRef.GPU(0))],
    ) as graph:
        graph.output(graph.inputs[0].tensor + 1)

    model = session.load(graph)
    input_tensor = Buffer.from_numpy(np.arange(4, dtype=np.float32)).to(
        model.input_devices[0]
    )

    (baseline,) = model.execute(input_tensor)
    np.testing.assert_allclose(
        baseline.to_numpy(), np.arange(4, dtype=np.float32) + 1
    )

    (captured_output,) = model.capture(input_tensor)
    model.replay(input_tensor)

    # (captured_output,) = model.execute(input_tensor)
    np.testing.assert_allclose(
        captured_output.to_numpy(), np.arange(4, dtype=np.float32) + 1
    )

    # Replay with original input values and verify output.
    model.replay(input_tensor)
    np.testing.assert_allclose(
        captured_output.to_numpy(), np.arange(4, dtype=np.float32) + 1
    )

    # Update input in-place and replay to verify the graph uses updated values.
    updated_values = Buffer.from_numpy(np.arange(4, dtype=np.float32) + 3).to(
        model.input_devices[0]
    )
    input_tensor.inplace_copy_from(updated_values)

    model.replay(input_tensor)
    np.testing.assert_allclose(
        captured_output.to_numpy(), np.arange(4, dtype=np.float32) + 4
    )


def test_same_shapes_different_input_buffers() -> None:
    """Test that graphs captured with same shapes but different buffers are cached separately.

    Without this fix, replaying with different buffers could incorrectly reuse
    a cached graph that references the wrong memory.
    """
    if accelerator_count() == 0:
        pytest.skip("GPU not available")

    accelerator = Accelerator()
    session = InferenceSession(devices=[accelerator])

    # Create a simple graph that adds 10 to the input
    with Graph(
        "same_shapes_test",
        input_types=[TensorType(DType.float32, [4], device=DeviceRef.GPU(0))],
    ) as graph:
        graph.output(graph.inputs[0].tensor + 10)

    model = session.load(graph)

    # Create two input buffers with the same shape but different values and underlying memory
    input_tensor1 = Buffer.from_numpy(
        np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    ).to(model.input_devices[0])
    input_tensor2 = Buffer.from_numpy(
        np.array([5.0, 6.0, 7.0, 8.0], dtype=np.float32)
    ).to(model.input_devices[0])

    # Capture graphs with each input - these should create separate cached graphs
    # because the buffer addresses are different (per commit a236dd8126)
    (captured_output1,) = model.capture(input_tensor1)
    (captured_output2,) = model.capture(input_tensor2)

    # Replay both graphs to populate output buffers
    model.replay(input_tensor1)
    model.replay(input_tensor2)

    # Verify each replay wrote to its own output buffer
    np.testing.assert_allclose(
        captured_output1.to_numpy(),
        np.array([11.0, 12.0, 13.0, 14.0], dtype=np.float32),
    )
    np.testing.assert_allclose(
        captured_output2.to_numpy(),
        np.array([15.0, 16.0, 17.0, 18.0], dtype=np.float32),
    )

    # Replay input1 again and verify only captured_output1 is updated
    model.replay(input_tensor1)
    np.testing.assert_allclose(
        captured_output1.to_numpy(),
        np.array([11.0, 12.0, 13.0, 14.0], dtype=np.float32),
    )
    # captured_output2 should remain unchanged
    np.testing.assert_allclose(
        captured_output2.to_numpy(),
        np.array([15.0, 16.0, 17.0, 18.0], dtype=np.float32),
    )

    # Update input1 in-place and replay to verify the graph reads from the correct buffer
    updated_values1 = Buffer.from_numpy(
        np.array([10.0, 20.0, 30.0, 40.0], dtype=np.float32)
    ).to(model.input_devices[0])
    input_tensor1.inplace_copy_from(updated_values1)

    model.replay(input_tensor1)
    np.testing.assert_allclose(
        captured_output1.to_numpy(),
        np.array([20.0, 30.0, 40.0, 50.0], dtype=np.float32),
    )
    # captured_output2 should still be unchanged
    np.testing.assert_allclose(
        captured_output2.to_numpy(),
        np.array([15.0, 16.0, 17.0, 18.0], dtype=np.float32),
    )


def test_replay_with_external_allocations() -> None:
    if accelerator_count() == 0:
        pytest.skip("GPU not available")

    accelerator = Accelerator()
    session = InferenceSession(devices=[accelerator])

    # Use Linear layer which internally uses matmul with transpose_b=True.
    # Shape (M=65, N=6144, K=4096) with bfloat16 triggers SM100 dispatch
    # for native Mojo kernels (not vendor BLAS), which is required for
    # CUDA stream capture and launch trace verification.
    # Dimensions mapping: M=sequence_length, K=in_features, N=out_features
    sequence_length = 65  # M=65-81 range has tuning config for this shape
    in_features = 4096
    out_features = 6144  # (N=6144, K=4096) has tuning config in SM100 dispatch

    max_linear = Linear(
        in_dim=in_features,
        out_dim=out_features,
        dtype=DType.bfloat16,
        has_bias=False,
        device=DeviceRef.GPU(),
    )

    # Initialize weights with random bfloat16 values using torch
    weight_tensor = torch.randn(out_features, in_features, dtype=torch.bfloat16)
    max_linear.load_state_dict({"weight": weight_tensor})

    with Graph(
        "buffer_reuse_test",
        input_types=[
            TensorType(
                DType.bfloat16,
                [sequence_length, in_features],
                device=DeviceRef.GPU(),
            )
        ],
    ) as graph:
        graph.output(max_linear(graph.inputs[0].tensor))

    model = session.load(graph, weights_registry=max_linear.state_dict())

    # Create input buffer using torch for bfloat16 support
    input_tensor = torch.randn(
        sequence_length, in_features, dtype=torch.bfloat16, device="cuda"
    )
    input_buf = Buffer.from_dlpack(input_tensor)

    results = model.capture(input_buf)

    external_buffers = []
    for _ in range(10):
        external_buffers.append(
            Buffer(DType.float32, [256, 256], device=model.input_devices[0])
        )
    accelerator.synchronize()

    external_buffers.clear()
    accelerator.synchronize()

    for _ in range(10):
        external_buffers.append(
            Buffer(DType.float32, [256, 256], device=model.input_devices[0])
        )
    accelerator.synchronize()

    del external_buffers
    accelerator.synchronize()

    del results
    accelerator.synchronize()

    del input_tensor
    del input_buf
    accelerator.synchronize()

    del model
    accelerator.synchronize()


def test_debug_verify_replay() -> None:
    """Test that debug_verify_replay validates captured graphs match eager execution."""
    if accelerator_count() == 0:
        pytest.skip("GPU not available")

    accelerator = Accelerator()
    accelerator.synchronize()

    session = InferenceSession(devices=[accelerator])

    # Create a simple graph that adds 2 to the input
    with Graph(
        "debug_verify_test",
        input_types=[TensorType(DType.float32, [4], device=DeviceRef.GPU(0))],
    ) as graph:
        graph.output(graph.inputs[0].tensor + 2)

    model = session.load(graph)
    input_tensor = Buffer.from_numpy(np.arange(4, dtype=np.float32)).to(
        model.input_devices[0]
    )

    # Capture the graph
    model.capture(input_tensor)

    # Verify that the captured graph matches eager execution
    # This should succeed since the execution is identical
    model.debug_verify_replay(input_tensor)

    # Verify the captured output is still correct after verification
    model.replay(input_tensor)
    (output,) = model.execute(input_tensor)
    np.testing.assert_allclose(
        output.to_numpy(), np.arange(4, dtype=np.float32) + 2
    )
