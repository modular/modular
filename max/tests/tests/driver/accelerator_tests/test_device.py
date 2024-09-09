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


def test_equality():
    # We should be able to test the equality of devices.
    cpu = CPU()
    cuda = CUDA()

    assert cpu != cuda
