# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #


import numpy as np
import torch
from max.driver import CPU, Accelerator, Tensor, accelerator_api
from max.dtype import DType


def test_from_numpy_accelerator():
    # A user should be able to create an accelerator tensor from a numpy array.
    arr = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int32)
    tensor = Tensor.from_numpy(arr).to(Accelerator())
    assert tensor.shape == (2, 3)
    assert tensor.dtype == DType.int32


def test_is_host_accelerator():
    # Accelerator tensors should be marked as not being on-host.
    assert not Tensor(DType.int32, (1, 1), device=Accelerator()).is_host


def test_host_device_copy():
    # We should be able to freely copy tensors between host and device.
    host_tensor = Tensor.from_numpy(np.array([1, 2, 3], dtype=np.int32))
    device_tensor = host_tensor.copy(Accelerator())
    tensor = device_tensor.copy(CPU())

    assert tensor.shape == host_tensor.shape
    assert tensor.dtype == host_tensor.dtype
    assert tensor[0].item() == 1
    assert tensor[1].item() == 2
    assert tensor[2].item() == 3


def test_device_device_copy():
    # We should be able to freely copy tensors between device and device.
    acc = Accelerator()

    device_tensor1 = Tensor.from_numpy(np.array([1, 2, 3], dtype=np.int32)).to(
        acc
    )
    device_tensor2 = device_tensor1.copy(acc)
    tensor = device_tensor2.copy(CPU())

    assert tensor.shape == device_tensor1.shape
    assert tensor.dtype == DType.int32
    assert tensor[0].item() == 1
    assert tensor[1].item() == 2
    assert tensor[2].item() == 3


def test_torch_tensor_conversion():
    # Our tensors should be convertible to and from Torch tensors. We have to a
    # bunch of juggling between host and device because we don't have a
    # Accelerator-compatible version of torch available yet.
    torch_tensor = torch.reshape(torch.arange(1, 11, dtype=torch.int32), (2, 5))
    host_tensor = Tensor.from_dlpack(torch_tensor)
    acc_tensor = host_tensor.to(Accelerator())
    assert acc_tensor.shape == (2, 5)
    assert acc_tensor.dtype == DType.int32
    host_tensor = acc_tensor.to(CPU())
    torch_tensor_copy = torch.from_dlpack(host_tensor)
    assert torch.all(torch.eq(torch_tensor, torch_tensor_copy))


def test_to_device():
    cpu = CPU()
    acc = Accelerator()

    host_tensor = Tensor(dtype=DType.int32, shape=(3, 3), device=cpu)
    acc_tensor = host_tensor.to(acc)

    assert cpu == host_tensor.device
    assert acc == acc_tensor.device

    assert acc != host_tensor.device
    assert cpu != acc_tensor.device


def test_zeros():
    # We should be able to initialize an all-zero tensor.
    tensor = Tensor.zeros((3, 3), DType.int32, device=Accelerator())
    host_tensor = tensor.to(CPU())
    assert np.array_equal(
        host_tensor.to_numpy(), np.zeros((3, 3), dtype=np.int32)
    )


DLPACK_DTYPES = {
    DType.int8: torch.int8,
    DType.int16: torch.int16,
    DType.int32: torch.int32,
    DType.int64: torch.int64,
    DType.uint8: torch.uint8,
    DType.uint16: torch.uint16,
    DType.uint32: torch.uint32,
    DType.uint64: torch.uint64,
    DType.float16: torch.float16,
    DType.float32: torch.float32,
    DType.float64: torch.float64,
}


def test_dlpack_accelerator():
    # TODO(MSDK-897): improve test coverage with different shapes and strides.
    for dtype, torch_dtype in DLPACK_DTYPES.items():
        tensor = Tensor(dtype, (1, 4))
        for j in range(4):
            tensor[0, j] = j
        acc_tensor = tensor.to(Accelerator())

        torch_tensor = torch.from_dlpack(acc_tensor)
        assert torch_tensor.dtype == torch_dtype
        assert acc_tensor.shape == torch_tensor.shape

        torch_tensor[0, 0] = 7
        assert acc_tensor[0, 0].to(CPU()).item() == 7


def test_from_dlpack():
    # TODO(MSDK-897): improve test coverage with different shapes and strides.
    for dtype, torch_dtype in DLPACK_DTYPES.items():
        torch_tensor = torch.tensor([0, 1, 2, 3], dtype=torch_dtype).cuda()
        acc_tensor = Tensor.from_dlpack(torch_tensor)
        assert acc_tensor.dtype == dtype
        assert acc_tensor.shape == torch_tensor.shape

        torch_tensor[0] = 7
        assert acc_tensor[0].to(CPU()).item() == 7


def test_dlpack_device():
    tensor = Tensor(DType.int32, (3, 3), device=Accelerator())
    device_tuple = tensor.__dlpack_device__()
    assert len(device_tuple) == 2
    assert isinstance(device_tuple[0], int)
    if accelerator_api() == "hip":
        # 10 is the value of DLDeviceType::kDLROCM
        assert device_tuple[0] == 10
    else:
        # 2 is the value of DLDeviceType::kDLCUDA
        assert device_tuple[0] == 2
    assert isinstance(device_tuple[1], int)
    assert device_tuple[1] == 0  # should be the default device


def test_scalar():
    # We should be able to create scalar values on accelerators.
    acc = Accelerator()
    scalar = Tensor.scalar(5, DType.int32, device=acc)
    assert scalar.device == acc

    host_scalar = scalar.to(CPU())
    assert host_scalar.item() == 5


def test_accelerator_to_numpy():
    acc = Accelerator()
    tensor = Tensor.zeros((3, 3), DType.int32, device=acc)

    assert np.array_equal(tensor.to_numpy(), np.zeros((3, 3), dtype=np.int32))


def test_d2h_inplace_copy_from_tensor_view():
    enumerated = np.zeros((5, 2, 3), dtype=np.int32)
    for i, j, k in np.ndindex(enumerated.shape):
        enumerated[i, j, k] = 100 * i + 10 * j + k

    all_nines = np.full((5, 2, 3), 999, dtype=np.int32)

    gpu_tensor = Tensor.from_numpy(enumerated).to(Accelerator())
    host_tensor = Tensor.from_numpy(all_nines)

    # Copy 3rd row of gpu_tensor into 1st row of host_tensor.
    host_tensor[1, :, :].inplace_copy_from(gpu_tensor[3, :, :])

    expected = np.array(
        [
            [[999, 999, 999], [999, 999, 999]],
            [[300, 301, 302], [310, 311, 312]],
            [[999, 999, 999], [999, 999, 999]],
            [[999, 999, 999], [999, 999, 999]],
            [[999, 999, 999], [999, 999, 999]],
        ]
    )
    assert np.array_equal(host_tensor.to_numpy(), expected)
