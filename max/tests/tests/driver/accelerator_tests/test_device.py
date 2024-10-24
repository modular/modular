# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
from max.driver import CPU, CUDA


def test_cuda_device():
    # We should be able to create a CUDA device.
    cuda = CUDA()
    assert "cuda" in str(cuda)
    assert not cuda.is_host


def scoped_device():
    _ = CUDA(0)  # NOTE: device ID is intentionally explicit.


def test_stress_cuda_device():
    # We should be able to call CUDA() many times, and get cached outputs.
    devices = [CUDA() for _ in range(64)]
    assert len({id(cuda._device) for cuda in devices}) == 1

    # TODO(MSDK-1220): move this before the above assert when the context no
    # longer leaks. Until then, this should still test that the default device
    # ID and explicit 0 ID share a device cache entry.
    for _ in range(64):
        scoped_device()


def test_equality():
    # We should be able to test the equality of devices.
    cpu = CPU()
    cuda = CUDA()

    assert cpu != cuda


def test_stats():
    # We should be able to query utilization stats for the device.
    cuda = CUDA()
    stats = cuda.stats
    assert "timestamp" in stats
    assert "free_memory" in stats
    assert "total_memory" in stats
