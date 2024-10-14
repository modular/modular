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


def test_stress_cuda_device():
    # We should be able to call CUDA() many times, and get cached outputs.
    devices = [CUDA() for _ in range(64)]
    assert len({id(cuda._device) for cuda in devices}) == 1


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
