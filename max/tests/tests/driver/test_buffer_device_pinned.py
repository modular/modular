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
"""Test DevicePinnedBuffer class."""

import numpy as np
import pytest
from max.driver import (
    CPU,
    Accelerator,
    Buffer,
    DevicePinnedBuffer,
    accelerator_count,
)
from max.dtype import DType


def test_device_pinned_buffer_type_cpu() -> None:
    """Test that DevicePinnedBuffer raises error on CPU."""
    cpu = CPU()

    # DevicePinnedBuffer should raise ValueError on CPU devices
    with pytest.raises(
        ValueError, match="DevicePinnedBuffer requires a non-host device"
    ):
        DevicePinnedBuffer(dtype=DType.float32, shape=[10], device=cpu)


@pytest.mark.skipif(
    accelerator_count() == 0,
    reason="DevicePinnedBuffer GPU tests require GPU",
)
def test_device_pinned_buffer_type_gpu() -> None:
    """Test that DevicePinnedBuffer creates pinned memory on GPU."""
    gpu = Accelerator()
    buffer = DevicePinnedBuffer(dtype=DType.float32, shape=[10], device=gpu)

    # Should be an instance of DevicePinnedBuffer
    assert isinstance(buffer, DevicePinnedBuffer)
    assert isinstance(buffer, Buffer)

    # On GPU, should be pinned
    assert buffer.pinned
    # Pinned buffers are associated with GPU device context
    assert not buffer.device.is_host


@pytest.mark.skipif(
    accelerator_count() == 0,
    reason="DevicePinnedBuffer GPU tests require GPU",
)
def test_device_pinned_buffer_zeros() -> None:
    """Test DevicePinnedBuffer.zeros() static method."""
    gpu = Accelerator()
    buffer = DevicePinnedBuffer.zeros(
        shape=[5, 3], dtype=DType.int32, device=gpu
    )

    # Should be an instance of DevicePinnedBuffer
    assert isinstance(buffer, DevicePinnedBuffer)
    assert isinstance(buffer, Buffer)

    # Should be pinned
    assert buffer.pinned

    # Should be initialized to zeros
    np_array = buffer.to_numpy()
    assert np_array.shape == (5, 3)
    assert np.all(np_array == 0)


@pytest.mark.skipif(
    accelerator_count() == 0,
    reason="DevicePinnedBuffer GPU tests require GPU",
)
def test_device_pinned_buffer_data_transfer() -> None:
    """Test that DevicePinnedBuffer can transfer data to/from GPU."""
    gpu = Accelerator()

    # Create a pinned buffer and fill it with data
    host_buffer = DevicePinnedBuffer(
        dtype=DType.float32, shape=[100], device=gpu
    )
    host_np = host_buffer.to_numpy()
    host_np[:] = np.arange(100, dtype=np.float32)

    # Transfer to GPU
    gpu_buffer = host_buffer.to(gpu)

    # Transfer back to a new pinned buffer
    result_buffer = DevicePinnedBuffer(
        dtype=DType.float32, shape=[100], device=gpu
    )
    result_buffer.inplace_copy_from(gpu_buffer)

    # Verify data is correct
    result_np = result_buffer.to_numpy()
    expected = np.arange(100, dtype=np.float32)
    assert np.allclose(result_np, expected)


def test_device_pinned_buffer_cpu_zeros() -> None:
    """Test DevicePinnedBuffer.zeros() raises error on CPU."""
    cpu = CPU()

    # DevicePinnedBuffer.zeros() should raise ValueError on CPU devices
    with pytest.raises(
        ValueError, match="DevicePinnedBuffer requires a non-host device"
    ):
        DevicePinnedBuffer.zeros(shape=[4, 2], dtype=DType.float64, device=cpu)


@pytest.mark.skipif(
    accelerator_count() == 0,
    reason="DevicePinnedBuffer with events requires GPU",
)
def test_device_pinned_buffer_with_events() -> None:
    """Test DevicePinnedBuffer with DeviceEvent for explicit synchronization."""
    gpu = Accelerator()
    stream = gpu.default_stream

    # Create a pinned buffer for efficient host-device transfers
    host_buffer = DevicePinnedBuffer(
        dtype=DType.float32, shape=[1000], device=gpu
    )
    host_np = host_buffer.to_numpy()
    host_np[:] = np.arange(1000, dtype=np.float32)

    # Transfer to GPU and record an event after the transfer
    gpu_buffer = host_buffer.to(gpu)
    event = stream.record_event()

    # Event should eventually be ready
    event.synchronize()
    assert event.is_ready()

    # Transfer back using pinned buffer
    result_buffer = DevicePinnedBuffer(
        dtype=DType.float32, shape=[1000], device=gpu
    )
    result_buffer.inplace_copy_from(gpu_buffer)

    # Record another event after the copy
    copy_event = stream.record_event()
    copy_event.synchronize()

    # Verify data is correct
    result_np = result_buffer.to_numpy()
    expected = np.arange(1000, dtype=np.float32)
    assert np.allclose(result_np, expected)
