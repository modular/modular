# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
"""Test max.driver Tensors."""
from itertools import product

import numpy as np
import pytest
import torch
from max.driver import CPU, Tensor
from max.dtype import DType


def test_tensor():
    # Validate that metadata shows up correctly
    tensor = Tensor((3, 4, 5), DType.float32)
    assert DType.float32 == tensor.dtype
    assert "DType.float32" == str(tensor.dtype)
    assert (3, 4, 5) == tensor.shape
    assert 3 == tensor.rank


def test_get_and_set():
    tensor = Tensor((3, 4, 5), DType.int32)
    tensor[0, 1, 3] = 68
    # Get should return zero-d tensor
    elt = tensor[0, 1, 3]
    assert 0 == elt.rank
    assert () == elt.shape
    assert DType.int32 == elt.dtype
    assert 68 == elt.item()

    # Setting negative indices
    tensor[-1, -1, -1] = 72
    assert 72 == tensor[2, 3, 4].item()

    # Ensure we're not writing to the same memory location with each index
    assert 68 == tensor[0, 1, 3].item()

    # Cannot do out-of-bounds indexing
    with pytest.raises(IndexError):
        tensor[5, 2, 2] = 23

    # Cannot do out-of-bounds with negative indices
    with pytest.raises(IndexError):
        tensor[-4, -3, -3] = 42

    # Indexes need to be equal to the tensor rank
    with pytest.raises(ValueError):
        tensor[2, 2] = 2
    with pytest.raises(ValueError):
        tensor[2, 2, 2, 2] = 2

    # Cannot call item (without arguments) on a non-zero-rank tensor
    with pytest.raises(ValueError):
        tensor.item()


def test_slice():
    # Tensor slices should have the desired shape and should preserve
    # reference semantics.
    tensor = Tensor((3, 3, 3), DType.int32)
    subtensor = tensor[:2, :2, :2]
    assert subtensor.shape == (2, 2, 2)
    subtensor[0, 0, 0] = 25
    assert tensor[0, 0, 0].item() == 25

    # We can take arbitrary slices of slices and preserve reference
    # semantics to the original tensor and any derived slices.
    subsubtensor = subtensor[:1, :1, :1]
    assert subsubtensor[0, 0, 0].item() == 25
    subsubtensor[0, 0, 0] = 37
    assert tensor[0, 0, 0].item() == 37
    assert subtensor[0, 0, 0].item() == 37

    # Users should be able to specify step sizes and get tensors of
    # an expected size and beginning at the expected offset.
    strided_subtensor = tensor[1::2, 1::2, 1::2]
    assert strided_subtensor.shape == (1, 1, 1)
    strided_subtensor[0, 0, 0] = 256
    assert tensor[1, 1, 1].item() == 256

    # Invalid slice semantics should throw an exception
    with pytest.raises(ValueError):
        tensor[::0, ::0, ::0]


def test_drop_dimensions():
    tensor = Tensor((5, 5, 5), DType.int32)
    # When indexing into a tensor with a mixture of slices and integral
    # indices, the slice should drop any dimensions that correspond to
    # integral indices.
    droptensor = tensor[:, 2, :]
    assert droptensor.rank == 2
    assert droptensor.shape == (5, 5)

    for i in range(4):
        droptensor[i, i] = i
    droptensor[-1, -1] = 4
    for i in range(5):
        assert tensor[i, 2, i].item() == i


def test_negative_step():
    tensor = Tensor((3, 3), DType.int32)
    tensor[0, 0] = 1
    tensor[0, 1] = 2
    tensor[0, 2] = 3
    tensor[1, 0] = 4
    tensor[1, 1] = 5
    tensor[1, 2] = 6
    tensor[2, 0] = 7
    tensor[2, 1] = 8
    tensor[2, 2] = 9
    # Tensors should support slices with negative steps.
    revtensor = tensor[::-1, ::-1]
    assert revtensor[0, 0].item() == 9
    assert revtensor[0, 1].item() == 8
    assert revtensor[0, 2].item() == 7
    assert revtensor[1, 0].item() == 6
    assert revtensor[1, 1].item() == 5
    assert revtensor[1, 2].item() == 4
    assert revtensor[2, 0].item() == 3
    assert revtensor[2, 1].item() == 2
    assert revtensor[2, 2].item() == 1


def test_out_of_bounds_slices():
    tensor = Tensor((3, 3, 3), DType.int32)

    # Out of bounds indexes are allowed in slices.
    assert tensor[4:, :2, 8:10:-1].shape == (0, 2, 0)

    # Out of bounds indexes are not allowed in integral indexing.
    with pytest.raises(IndexError):
        tensor[4:, :2, 4]


def test_one_dimensional_tensor():
    tensor = Tensor((10,), DType.int32)
    for i in range(10):
        tensor[i] = i

    for i in range(i):
        assert tensor[i].item() == i


def test_contiguous_tensor():
    # Initialized tensors should be contiguous, and tensor slices should not be.
    tensor = Tensor((3, 3), DType.int32)
    assert tensor.is_contiguous
    val = 1
    for x, y in product(range(3), range(3)):
        tensor[x, y] = val
        val += 1

    subtensor = tensor[:2, :2]
    assert not subtensor.is_contiguous

    # There's a special case where reversed slices (which are "technically"
    # contiguous) should not be considered as such.
    assert not tensor[::-1, ::-1].is_contiguous

    subsubtensor = tensor[:2, :2]
    assert subsubtensor.shape == (2, 2)
    cont_tensor = subsubtensor.contiguous()
    assert cont_tensor.shape == (2, 2)
    assert cont_tensor.is_contiguous
    assert cont_tensor[0, 0].item() == 1
    assert cont_tensor[0, 1].item() == 2
    assert cont_tensor[1, 0].item() == 4
    assert cont_tensor[1, 1].item() == 5


def test_modify_contiguous_tensor():
    # Modifications made to the original tensor should not be reflected
    # on the contiguous copy, and vice-versa.
    tensor = Tensor((3, 3), DType.int32)
    for x, y in product(range(3), range(3)):
        tensor[x, y] = 1

    cont_tensor = tensor.contiguous()

    cont_tensor[1, 1] = 22
    assert tensor[1, 1].item() == 1

    tensor[2, 2] = 25
    assert cont_tensor[2, 2].item() == 1


def test_contiguous_slice():
    # A contiguous slice of a tensor should be considered contiguous. An
    # example of this is taking a single row from a 2-d array.
    singlerow = Tensor.from_numpy(
        np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int32)
    )[0, :]
    assert singlerow.shape == (3,)
    assert singlerow.is_contiguous

    # This should also work in cases where we take a couple of adjacent rows
    # from a multi-dimensional array.
    multirow = Tensor.from_numpy(np.ones((5, 5), dtype=np.int32))[2:4, :]
    assert multirow.shape == (2, 5)
    assert multirow.is_contiguous

    # We also need this work in cases where we're just taking subarrays of 1-d
    # arrays.
    subarray = Tensor.from_numpy(np.ones((10,), dtype=np.int32))[2:5]
    assert subarray.shape == (3,)
    assert subarray.is_contiguous


def test_from_numpy():
    # A user should be able to create a tensor from a numpy array.
    arr = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int32)
    tensor = Tensor.from_numpy(arr)
    assert tensor.shape == (2, 3)
    assert tensor.dtype == DType.int32
    assert tensor[0, 0].item() == 1
    assert tensor[0, 1].item() == 2
    assert tensor[0, 2].item() == 3
    assert tensor[1, 0].item() == 4
    assert tensor[1, 1].item() == 5
    assert tensor[1, 2].item() == 6


def test_is_host():
    # CPU tensors should be marked as being on-host.
    assert Tensor((1, 1), DType.int32, device=CPU()).is_host


def test_host_host_copy():
    # We should be able to freely copy tensors between host and host.
    cpu_device = CPU()

    host_tensor = Tensor.from_numpy(np.array([1, 2, 3], dtype=np.int32))
    tensor = host_tensor.copy_to(cpu_device)

    assert tensor.shape == host_tensor.shape
    assert tensor.dtype == DType.int32
    assert tensor[0].item() == 1
    assert tensor[1].item() == 2
    assert tensor[2].item() == 3


DLPACK_DTYPES = {
    np.int8: DType.int8,
    np.int16: DType.int16,
    np.int32: DType.int32,
    np.int64: DType.int64,
    np.uint8: DType.uint8,
    np.uint16: DType.uint16,
    np.uint32: DType.uint32,
    np.uint64: DType.uint64,
    # np.float16  # TODO(MSDK-893): enable float16.
    np.float32: DType.float32,
    np.float64: DType.float64,
}


def test_from_dlpack():
    # TODO(MSDK-897): improve test coverage with different shapes and strides.
    for np_dtype, our_dtype in DLPACK_DTYPES.items():
        array = np.array([0, 1, 2, 3], np_dtype)
        tensor = Tensor.from_dlpack(array)
        assert tensor.dtype == our_dtype
        assert tensor.shape == array.shape

        tensor[0] = np_dtype(7)
        assert array[0] == np_dtype(7)


def test_dlpack_device():
    tensor = Tensor((3, 3), DType.int32)
    device_tuple = tensor.__dlpack_device__()
    assert len(device_tuple) == 2
    assert isinstance(device_tuple[0], int)
    assert device_tuple[0] == 1  # 1 is the value of DLDeviceType::kDLCPU
    assert isinstance(device_tuple[1], int)
    assert device_tuple[1] == 0  # should be the default device


def test_dlpack():
    # TODO(MSDK-897): improve test coverage with different shapes and strides.
    for np_dtype, our_dtype in DLPACK_DTYPES.items():
        tensor = Tensor((1, 4), our_dtype)
        for j in range(4):
            tensor[0, j] = j

        array = np.from_dlpack(tensor)
        assert array.dtype == np_dtype
        assert tensor.shape == array.shape

        # Numpy creates a read-only array, so we modify ours.
        tensor[0, 0] = np_dtype(7)
        assert array[0, 0] == np_dtype(7)


def test_torch_tensor_conversion():
    # Our tensors should be convertible to and from Torch tensors.
    torch_tensor = torch.reshape(torch.arange(1, 11, dtype=torch.int32), (2, 5))
    driver_tensor = Tensor.from_dlpack(torch_tensor)
    assert driver_tensor.shape == (2, 5)
    assert driver_tensor.dtype == DType.int32
    for x, y in product(range(2), range(5)):
        assert torch_tensor[x, y].item() == driver_tensor[x, y].item()

    converted_tensor = torch.from_dlpack(driver_tensor)
    assert torch.all(torch.eq(torch_tensor, converted_tensor))

    # We should also be able to get this running for boolean tensors.
    bool_tensor = torch.tensor([False, True, False, True])
    converted_bool = Tensor.from_dlpack(bool_tensor)
    assert converted_bool.shape == (4,)
    assert converted_bool.dtype == DType.bool
    for x in range(4):
        assert bool_tensor[x].item() == converted_bool[x].item()

    reconverted_bool = torch.from_dlpack(converted_bool)
    assert torch.all(torch.eq(bool_tensor, reconverted_bool))


def test_device():
    # We should be able to set and query the device that a tensor is resident on.
    cpu = CPU()
    tensor = Tensor((3, 3), dtype=DType.int32, device=cpu)
    assert cpu == tensor.device
