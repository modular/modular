# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

import numpy as np
import torch
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


def test_torch_tensor_conversion():
    # Our tensors should be convertible to and from Torch tensors. We have to a
    # bunch of juggling between host and device because we don't have a
    # CUDA-compatible version of torch available yet.
    torch_tensor = torch.reshape(torch.arange(1, 11, dtype=torch.int32), (2, 5))
    copied_tensor = Tensor.from_dlpack(torch_tensor)
    gpu_tensor = copied_tensor.copy_to(CUDA())
    assert gpu_tensor.shape == (2, 5)
    assert gpu_tensor.dtype == DType.int32
    host_tensor = gpu_tensor.copy_to(CPU())
    torch_tensor_copy = torch.from_dlpack(host_tensor)
    assert torch.all(torch.eq(torch_tensor, torch_tensor_copy))


def test_device():
    cpu = CPU()
    cuda = CUDA()

    host_tensor = Tensor((3, 3), dtype=DType.int32, device=cpu)
    gpu_tensor = host_tensor.copy_to(cuda)

    assert cpu == host_tensor.device
    assert cuda == gpu_tensor.device

    assert cuda != host_tensor.device
    assert cpu != gpu_tensor.device


def test_zeros():
    # We should be able to initialize an all-zero tensor.
    tensor = Tensor.zeros((3, 3), DType.int32, device=CUDA())
    host_tensor = tensor.copy_to(CPU())
    assert np.array_equal(
        host_tensor.to_numpy(), np.zeros((3, 3), dtype=np.int32)
    )


def test_dlpack_device():
    tensor = Tensor((3, 3), DType.int32, device=CUDA())
    device_tuple = tensor.__dlpack_device__()
    assert len(device_tuple) == 2
    assert isinstance(device_tuple[0], int)
    assert device_tuple[0] == 2  # 2 is the value of DLDeviceType::kDLCUDA
    assert isinstance(device_tuple[1], int)
    assert device_tuple[1] == 0  # should be the default device
