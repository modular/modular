# ===----------------------------------------------------------------------=== #
# Copyright (c) 2025, Modular Inc. All rights reserved.
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

import numpy as np
import pytest
import torch
from max.driver import Accelerator, Buffer, accelerator_count
from max.dtype import DType
from max.engine import InferenceSession
from max.graph import DeviceRef, Graph, TensorType
from max.nn.legacy import Linear


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

    model.capture(input_tensor)
    (captured_output,) = model.execute(input_tensor)
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

    model.capture(input_buf)

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
