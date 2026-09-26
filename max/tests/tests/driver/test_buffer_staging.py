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
"""Tests staging buffers on an accelerator, tracked and untracked."""

import threading

import numpy as np
import pytest
from max.driver import (
    Accelerator,
    Buffer,
    CompletionFlag,
    Usage,
    accelerator_count,
)
from max.dtype import DType

pytestmark = pytest.mark.skipif(
    accelerator_count() == 0, reason="staging Buffer GPU tests require GPU"
)

UNTRACKED_STAGING = Usage.STAGING | Usage.UNTRACKED


def test_staging_buffer_data_transfer() -> None:
    gpu = Accelerator()

    host_buffer = Buffer(
        dtype=DType.float32, shape=[100], device=gpu, usage=Usage.STAGING
    )
    host_buffer.to_numpy()[:] = np.arange(100, dtype=np.float32)

    gpu_buffer = host_buffer.to(gpu)

    result_buffer = Buffer(
        dtype=DType.float32, shape=[100], device=gpu, usage=Usage.STAGING
    )
    result_buffer.inplace_copy_from(gpu_buffer)

    np.testing.assert_array_equal(
        result_buffer.to_numpy(), np.arange(100, dtype=np.float32)
    )


def test_staging_buffer_with_events() -> None:
    gpu = Accelerator()
    stream = gpu.default_queue

    host_buffer = Buffer(
        dtype=DType.float32, shape=[1000], device=gpu, usage=Usage.STAGING
    )
    host_buffer.to_numpy()[:] = np.arange(1000, dtype=np.float32)

    gpu_buffer = host_buffer.to(gpu)
    event = stream.record_event()
    event.synchronize()
    assert event.is_ready()

    result_buffer = Buffer(
        dtype=DType.float32, shape=[1000], device=gpu, usage=Usage.STAGING
    )
    result_buffer.inplace_copy_from(gpu_buffer)
    stream.record_event().synchronize()

    np.testing.assert_array_equal(
        result_buffer.to_numpy(), np.arange(1000, dtype=np.float32)
    )


def test_untracked_staging_to_numpy_does_not_synchronize() -> None:
    """``to_numpy`` on untracked staging memory returns without synchronizing.

    Contract 4a (DRIV-311): the read proceeds while the default stream is
    gated. A full ``device.synchronize()`` is the positive control: it must
    stay blocked while the gate is held, proving the gate is engaged and the
    staging result meaningful.
    """
    gpu = Accelerator()
    if gpu.api not in ("cuda", "hip"):
        pytest.skip("stream host-value gating requires CUDA/HIP")

    sentinel = np.arange(1, 5, dtype=np.float32)
    staging = Buffer(
        dtype=DType.float32, shape=[4], device=gpu, usage=UNTRACKED_STAGING
    )
    staging.to_numpy()[:] = sentinel

    flag = CompletionFlag(gpu)
    gate_stream = gpu.default_queue
    staging_done = threading.Event()
    control_done = threading.Event()
    staging_box: dict[str, object] = {}

    def read_staging() -> None:
        try:
            staging_box["value"] = staging.to_numpy()
        finally:
            staging_done.set()

    def sync_device() -> None:
        try:
            gpu.synchronize()
        finally:
            control_done.set()

    staging_thread = threading.Thread(target=read_staging, daemon=True)
    control_thread = threading.Thread(target=sync_device, daemon=True)

    gate_stream.wait_for_host_value(flag, 1)
    try:
        staging_thread.start()
        control_thread.start()
        assert staging_done.wait(timeout=10.0), (
            "untracked staging to_numpy blocked on the gated stream; it"
            " synchronized"
        )
        assert not control_done.wait(timeout=2.0), (
            "device.synchronize() returned while gated; the gate is not"
            " engaged so the staging result is not meaningful"
        )
    finally:
        flag.signal(1)
        staging_thread.join(timeout=60.0)
        control_thread.join(timeout=60.0)

    assert not staging_thread.is_alive() and not control_thread.is_alive()
    np.testing.assert_array_equal(staging_box["value"], sentinel)


def test_staging_to_numpy_is_zero_copy_view() -> None:
    """``to_numpy`` on staging memory aliases it (DRIV-311 contract 4b).

    Because the export is zero-copy, the numpy array does not own its data and
    a later host write to the buffer is visible through the array returned
    earlier.
    """
    gpu = Accelerator()
    staging = Buffer(
        dtype=DType.int32, shape=[8], device=gpu, usage=Usage.STAGING
    )

    view = staging.to_numpy()
    assert not view.flags.owndata

    written = np.arange(10, 18, dtype=np.int32)
    staging.to_numpy()[:] = written
    np.testing.assert_array_equal(view, written)


def test_untracked_is_the_only_difference() -> None:
    """Both staging intents allocate the same memory; one flag differs.

    ``UNTRACKED`` hands ordering to the caller, which is why the hazard layer
    skips it and a plain staging buffer is tracked.
    """
    gpu = Accelerator()
    tracked = Buffer(
        dtype=DType.float32, shape=[10], device=gpu, usage=Usage.STAGING
    )
    untracked = Buffer(
        dtype=DType.float32, shape=[10], device=gpu, usage=UNTRACKED_STAGING
    )

    assert tracked.pinned == untracked.pinned == True  # noqa: E712
    assert not tracked.device.is_host

    assert tracked.usage == Usage.STAGING
    assert untracked.usage == UNTRACKED_STAGING
    assert tracked._host_hazard_tracked
    assert not untracked._host_hazard_tracked


@pytest.mark.parametrize("usage", [Usage.STAGING, UNTRACKED_STAGING])
def test_staging_usage_survives_slice_and_view(usage: Usage) -> None:
    """Slices/views share storage, so they keep the parent's usage.

    An untracked slice that decayed to tracked would make reads like
    ``to_numpy()`` wait where the caller expects them not to (DRIV-7).
    """
    gpu = Accelerator()
    buf = Buffer(dtype=DType.float32, shape=[4, 4], device=gpu, usage=usage)
    # Buffer.get requires as many indices as the rank.
    sliced = buf[0, :]
    assert sliced.usage == usage
    assert sliced.pinned
    assert sliced._host_hazard_tracked == buf._host_hazard_tracked
    assert buf[1:3, :].usage == usage
    assert buf.view(DType.int32, [4, 4]).usage == usage


def test_staging_zeros() -> None:
    gpu = Accelerator()
    buf = Buffer.zeros(
        shape=[5, 3], dtype=DType.int32, device=gpu, usage=Usage.STAGING
    )
    assert buf.usage == Usage.STAGING
    assert buf.pinned
    # zeros() enqueues an async memset.
    gpu.synchronize()
    np.testing.assert_array_equal(
        buf.to_numpy(), np.zeros((5, 3), dtype=np.int32)
    )


@pytest.mark.parametrize("usage", [Usage.STAGING, UNTRACKED_STAGING])
def test_staging_dlpack_does_not_synchronize(usage: Usage) -> None:
    """Neither staging export drains the device; only the tracked one waits.

    Both allocate the same pinned memory, so what separates them is the hazard
    layer: ``STAGING`` is tracked and waits for the copy this test enqueues
    behind the gate, while ``STAGING | UNTRACKED`` is the caller-owns-ordering
    escape hatch and returns with the copy still pending.
    """
    gpu = Accelerator()
    if gpu.api not in ("cuda", "hip"):
        pytest.skip("stream host-value gating requires CUDA/HIP")

    host_buf = Buffer(dtype=DType.int32, shape=[4], device=gpu, usage=usage)
    src = Buffer.from_numpy(np.arange(1, 5, dtype=np.int32)).to(gpu)

    flag = CompletionFlag(gpu)
    done = threading.Event()

    def worker() -> None:
        host_buf.to_numpy()
        done.set()

    worker_thread = threading.Thread(target=worker, daemon=True)
    gpu.default_queue.wait_for_host_value(flag, 1)
    try:
        host_buf.inplace_copy_from(src)
        worker_thread.start()
        completed_while_gated = done.wait(timeout=2.0)
    finally:
        flag.signal(1)
        worker_thread.join(timeout=60.0)

    assert not worker_thread.is_alive()
    if usage == Usage.STAGING:
        assert not completed_while_gated, (
            "a tracked staging read returned while the copy filling it was"
            " still gated; the hazard layer did not wait on its producer"
        )
    else:
        assert completed_while_gated, (
            "to_numpy()/__dlpack__ on untracked staging memory blocked on"
            " gated device work; nothing may be waited on there"
        )
