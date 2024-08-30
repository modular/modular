# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
import numpy as np
from max.driver import CPU, CUDA, Tensor
from max.dtype import DType


def test_from_numpy_cuda():
    # A user should be able to create a GPU tensor from a numpy array.
    arr = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int32)
    tensor = Tensor.from_numpy(arr, device=CUDA())
    assert tensor.shape == (2, 3)
    assert tensor.dtype == DType.int32


def test_is_host_cuda():
    # CUDA tensors should be marked as not being on-host.
    assert not Tensor((1, 1), DType.int32, device=CUDA()).is_host


def test_host_device_copy():
    # We should be able to freely copy tensors between host and device.
    cpu_device = CPU()
    cuda_device = CUDA()

    host_tensor = Tensor.from_numpy(
        np.array([1, 2, 3], dtype=np.int32), device=cpu_device
    )
    device_tensor = host_tensor.copy_to(cuda_device)
    tensor = device_tensor.copy_to(cpu_device)

    assert tensor.shape == host_tensor.shape
    assert tensor.dtype == DType.int32
    assert tensor[0].item() == 1
    assert tensor[1].item() == 2
    assert tensor[2].item() == 3


def test_device_device_copy():
    # We should be able to freely copy tensors between device and device.
    cpu_device = CPU()
    cuda_device = CUDA()

    host_tensor = Tensor.from_numpy(
        np.array([1, 2, 3], dtype=np.int32), device=cuda_device
    )
    device_tensor = host_tensor.copy_to(cuda_device)
    tensor = device_tensor.copy_to(cpu_device)

    assert tensor.shape == host_tensor.shape
    assert tensor.dtype == DType.int32
    assert tensor[0].item() == 1
    assert tensor[1].item() == 2
    assert tensor[2].item() == 3
