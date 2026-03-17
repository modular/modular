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
"""Test that runtime GPU errors include Python stack traces when debug mode is enabled."""

import os
from pathlib import Path

import numpy as np
import pytest
from max.driver import Accelerator, Buffer, accelerator_count
from max.dtype import DType
from max.engine import InferenceSession
from max.graph import DeviceRef, Graph, TensorType, ops


@pytest.fixture
def kernel_verification_ops_path() -> Path:
    return Path(os.environ["MODULAR_KERNEL_VERIFICATION_OPS_PATH"])


@pytest.fixture(scope="module")
def session() -> InferenceSession:
    return InferenceSession(devices=[Accelerator()])


@pytest.mark.skipif(accelerator_count() == 0, reason="Requires GPU")
def test_runtime_error_includes_python_stack_trace(
    session: InferenceSession,
    kernel_verification_ops_path: Path,
) -> None:
    """Verify that runtime CUDA errors include Python stack trace from graph construction.

    This test validates that when:
    - MODULAR_MAX_DEBUG=True (at build time)
    - MODULAR_DEVICE_CONTEXT_SYNC_MODE=1 (at runtime)

    Runtime GPU errors include "Source Traceback:" with the Python call stack
    showing where the failing operation was created in the graph.
    """

    def build_graph_with_crash() -> Graph:
        """Build a graph containing an op that will crash at runtime."""
        with Graph(
            "crash_test",
            input_types=[
                TensorType(DType.float32, [10], device=DeviceRef.GPU())
            ],
        ) as graph:
            graph._import_kernels([kernel_verification_ops_path])
            x = graph.inputs[0].tensor
            # This custom op launches a GPU kernel that traps
            crashed = ops.custom(
                name="intentional_gpu_crash",
                device=DeviceRef.GPU(),
                values=[x],
                out_types=[
                    TensorType(DType.float32, [10], device=DeviceRef.GPU())
                ],
            )[0].tensor
            graph.output(crashed)
        return graph

    graph = build_graph_with_crash()
    model = session.load(graph)

    # Create input and execute - this should crash
    input_data = np.zeros(10, dtype=np.float32)
    input_buffer = Buffer.from_numpy(input_data).to(Accelerator())

    with pytest.raises(Exception) as exc_info:
        model(input_buffer)

    error_message = str(exc_info.value)

    # Verify the error contains Python stack trace
    assert "Source Traceback:" in error_message, (
        f"Expected 'Source Traceback:' in error message.\n"
        f"Got: {error_message}\n\n"
        f"Hint: Ensure MODULAR_MAX_DEBUG=True was set at build time."
    )

    # Verify stack trace includes the function where the op was created
    assert "build_graph_with_crash" in error_message, (
        f"Expected 'build_graph_with_crash' function name in stack trace.\n"
        f"Got: {error_message}"
    )

    # Verify stack trace includes the test file name
    assert "test_runtime_source_notes.py" in error_message, (
        f"Expected 'test_runtime_source_notes.py' in stack trace.\n"
        f"Got: {error_message}"
    )
