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
"""Tests `.disable_auto_sync()` and `.mark_as_sync()` Buffer options."""

import numpy as np
import pytest
from max.driver import CPU, Accelerator, Buffer, Device, accelerator_count
from max.dtype import DType
from max.engine import InferenceSession, Model
from max.graph import DeviceRef, Graph, TensorType, ops

# Skip all tests in this module if no GPU is available
pytestmark = pytest.mark.skipif(
    accelerator_count() == 0, reason="Disable auto-sync tests require GPU"
)


@pytest.fixture(scope="module")
def gpu() -> Device:
    """Fixture providing GPU device."""
    return Accelerator()


@pytest.fixture(scope="module")
def cpu() -> Device:
    """Fixture providing CPU device."""
    return CPU()


# Size of the buffers
WIDTH = 1024 * 1024 * 10


@pytest.fixture(scope="module")
def add_graph(gpu: Device) -> Graph:
    """Fixture providing a simple vector addition graph."""
    with Graph(
        "add",
        input_types=[
            TensorType(DType.int32, [WIDTH], device=DeviceRef.from_device(gpu)),
            TensorType(DType.int32, [WIDTH], device=DeviceRef.from_device(gpu)),
        ],
    ) as graph:
        a, b = graph.inputs
        r = ops.add(a, b)
        graph.output(r)
    return graph


@pytest.fixture(scope="module")
def model(gpu: Device, add_graph: Graph) -> Model:
    """Fixture providing loaded model."""
    session = InferenceSession(devices=[gpu])
    model = session.load(add_graph)
    return model


def test_manual_sync_with_concurrent_execution(
    gpu: Device, model: Model
) -> None:
    """Test that manual sync allows overlapping GPU computation and CPU processing.

    This test exercises using `disable_auto_sync()` and `mark_as_ready()` to
    manually control synchronization. It only checks for correctness, but
    running it under nsys you should be able to observe the synchronization
    behavior overlapping m2 execution with m1 results checking.
    """

    # Setup pinned buffers
    a_pinned = Buffer(DType.int32, [WIDTH], device=gpu, pinned=True)
    a_pinned.to_numpy().fill(1)

    b_pinned = Buffer(DType.int32, [WIDTH], device=gpu, pinned=True)
    b_pinned.to_numpy().fill(1)

    c_pinned = Buffer(DType.int32, [WIDTH], device=gpu, pinned=True)
    c_pinned.to_numpy().fill(1)

    out_pinned_1 = Buffer(DType.int32, [WIDTH], device=gpu, pinned=True)
    # Disable auto-sync for our first buffer, which means it will require
    # mark_as_ready for proper synchronization
    out_pinned_1.disable_auto_sync()

    out_pinned_2 = Buffer(DType.int32, [WIDTH], device=gpu, pinned=True)

    # Send input buffers to the GPU
    a = a_pinned.to(gpu)
    b = b_pinned.to(gpu)

    # Start the first model execution: a + b
    m1 = model.execute(a, b)

    # Copy results back to a pinned buffer
    out_pinned_1.inplace_copy_from(m1[0])

    # Mark the first output buffer as ready - this sets a synchronization point
    out_pinned_1.mark_as_ready()

    # Send buffers to the GPU
    c = c_pinned.to(gpu)

    # Run the second model execution: a + c
    m2 = model.execute(a, c)
    # Copy back the data for the second model
    out_pinned_2.inplace_copy_from(m2[0])

    # Process the results of m1. Since out_pinned_1 is manually synced, this will
    # only wait to the point where mark_as_ready was called instead of waiting on
    # everything, effectively allowing model 2 to keep running on the GPU while
    # we're processing the output of model 1
    out_pinned_1_np = out_pinned_1.to_numpy()

    # Verify the result: 1 + 1 = 2
    # Don't check the entire range so the overlap is more visible in the trace
    sample_indices = range(WIDTH // 1024)
    assert np.all(out_pinned_1_np[sample_indices] == 2), (
        f"Expected all values to be 2, but found: {out_pinned_1_np[sample_indices]}"
    )

    # Process the result of m2. Since out_pinned_2 is NOT marked as manually synced
    # this will wait on all previously enqueued operations.
    out_pinned_2_np = out_pinned_2.to_numpy()

    # Verify the result: 1 + 1 = 2
    assert np.all(out_pinned_2_np[sample_indices] == 2), (
        f"Expected all values to be 2, but found: {out_pinned_2_np[sample_indices]}"
    )


def test_disable_auto_sync(gpu: Device) -> None:
    """Test that disable_auto_sync correctly disables automatic synchronization."""
    buffer = Buffer(DType.int32, [WIDTH], device=gpu, pinned=True)

    # Verify that disable_auto_sync doesn't raise an error
    buffer.disable_auto_sync()

    # Verify that mark_as_ready can be called after disable_auto_sync
    buffer.mark_as_ready()


@pytest.mark.xfail(
    strict=True,
    reason="Invalid use of markAsReady on buffer not with auto-sync enabled",
)
def test_mark_as_ready_error(gpu: Device) -> None:
    """Test that mark_as_ready fails if disable_auto_sync wasn't called."""
    buffer = Buffer(DType.int32, [WIDTH], device=gpu, pinned=True)
    buffer.mark_as_ready()


@pytest.mark.xfail(
    strict=True, reason="Buffer synchronization already disabled"
)
def test_disable_auto_sync_error(gpu: Device) -> None:
    """Test that disable_auto_sync fails if called twice on the same buffer."""
    buffer = Buffer(DType.int32, [WIDTH], device=gpu, pinned=True)
    buffer.disable_auto_sync()
    buffer.disable_auto_sync()
