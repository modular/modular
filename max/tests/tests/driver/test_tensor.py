# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
"""Test max.driver Tensors."""

import math
import tempfile
from itertools import product
from pathlib import Path

import numpy as np
import pytest
import torch
from hypothesis import given
from hypothesis import strategies as st
from max.driver import CPU, MemMapTensor, Tensor
from max.dtype import DType


def test_tensor() -> None:
    # Validate that metadata shows up correctly
    tensor = Tensor((3, 4, 5), DType.float32)
    assert DType.float32 == tensor.dtype
    assert "DType.float32" == str(tensor.dtype)
    assert (3, 4, 5) == tensor.shape
    assert 3 == tensor.rank

    # Validate that shape can be specified as a list and we copy the dims.
    shape = [2, 3]
    tensor2 = Tensor(shape, DType.float32)
    shape[0] = 1
    assert (2, 3) == tensor2.shape


def test_get_and_set() -> None:
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

    # Validate that passing the indices as a sequence object works.
    assert 72 == tensor[(2, 3, 4)].item()

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


def test_slice() -> None:
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


def test_drop_dimensions() -> None:
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


def test_negative_step() -> None:
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


def test_out_of_bounds_slices() -> None:
    tensor = Tensor((3, 3, 3), DType.int32)

    # Out of bounds indexes are allowed in slices.
    assert tensor[4:, :2, 8:10:-1].shape == (0, 2, 0)

    # Out of bounds indexes are not allowed in integral indexing.
    with pytest.raises(IndexError):
        tensor[4:, :2, 4]


def test_one_dimensional_tensor() -> None:
    tensor = Tensor((10,), DType.int32)
    for i in range(10):
        tensor[i] = i

    for i in range(i):
        assert tensor[i].item() == i


def test_contiguous_tensor() -> None:
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


def test_modify_contiguous_tensor() -> None:
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


def test_contiguous_slice() -> None:
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


def test_from_numpy() -> None:
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


def test_from_numpy_scalar() -> None:
    # Also test that scalar numpy arrays remain scalar.
    arr = np.array(1.0, dtype=np.float32)
    tensor = Tensor.from_numpy(arr)

    assert tensor.dtype == DType.float32
    assert tensor.shape == arr.shape


def test_is_host() -> None:
    # CPU tensors should be marked as being on-host.
    assert Tensor((1, 1), DType.int32, device=CPU()).is_host


def test_host_host_copy() -> None:
    # We should be able to freely copy tensors between host and host.
    host_tensor = Tensor.from_numpy(np.array([1, 2, 3], dtype=np.int32))
    tensor = host_tensor.copy(CPU())

    assert tensor.shape == host_tensor.shape
    assert tensor.dtype == host_tensor.dtype
    host_tensor[0] = 0  # Ensure we got a deep copy.
    assert tensor[0].item() == 1
    assert tensor[1].item() == 2
    assert tensor[2].item() == 3

    tensor2 = host_tensor.copy(CPU())
    assert tensor2[0].item() == 0
    assert tensor2[1].item() == 2
    assert tensor2[2].item() == 3


DLPACK_DTYPES = [
    DType.bool,
    DType.int8,
    DType.int16,
    DType.int32,
    DType.int64,
    DType.uint8,
    DType.uint16,
    DType.uint32,
    DType.uint64,
    DType.float16,
    DType.float32,
    DType.float64,
]


def test_from_dlpack() -> None:
    # TODO(MSDK-897): improve test coverage with different shapes and strides.
    for dtype in DLPACK_DTYPES:
        np_dtype = dtype.to_numpy()
        array = np.array([0, 1, 2, 3], np_dtype)
        tensor = Tensor.from_dlpack(array)
        assert tensor.dtype == dtype
        assert tensor.shape == array.shape

        if dtype is dtype.bool:
            assert array[0] == False
            tensor[0] = True
            assert array[0] == True
        else:
            tensor[0] = np_dtype.type(7)
            assert array[0] == np_dtype.type(7)


def test_from_dlpack_short_circuit() -> None:
    tensor = Tensor((4,), DType.int8)
    for i in range(4):
        tensor[i] = i

    # Test short circuiting.
    same_tensor = Tensor.from_dlpack(tensor)
    assert tensor is same_tensor
    copy_tensor = Tensor.from_dlpack(tensor, copy=True)
    assert tensor is not copy_tensor
    assert tensor.dtype == copy_tensor.dtype
    assert tensor.shape == copy_tensor.shape


def test_from_dlpack_copy() -> None:
    tensor = Tensor((4,), DType.int8)
    for i in range(4):
        tensor[i] = i

    arr = np.from_dlpack(tensor)
    tensor_copy = Tensor.from_dlpack(arr)  # Should be implicitly copied.
    tensor_copy[0] = np.int8(7)
    assert arr[0] != np.int8(7)

    with pytest.raises(BufferError):
        Tensor.from_dlpack(arr, copy=False)


def test_dlpack_device() -> None:
    tensor = Tensor((3, 3), DType.int32)
    device_tuple = tensor.__dlpack_device__()
    assert len(device_tuple) == 2
    assert isinstance(device_tuple[0], int)
    assert device_tuple[0] == 1  # 1 is the value of DLDeviceType::kDLCPU
    assert isinstance(device_tuple[1], int)
    assert device_tuple[1] == 0  # should be the default device


def test_dlpack() -> None:
    # TODO(MSDK-897): improve test coverage with different shapes and strides.
    for dtype in DLPACK_DTYPES:
        tensor = Tensor((1, 4), dtype)
        for j in range(4):
            tensor[0, j] = j

        # Numpy's dlpack implementation cannot handle its own bool types.
        if dtype is dtype.bool:
            continue

        np_dtype = dtype.to_numpy()
        array = np.from_dlpack(tensor)
        assert array.dtype == np_dtype
        assert tensor.shape == array.shape

        # Numpy creates a read-only array, so we modify ours.
        tensor[0, 0] = np_dtype.type(7)
        assert array[0, 0] == np_dtype.type(7)


def test_torch_tensor_conversion() -> None:
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


@given(st.floats())
def test_setitem_bfloat16(value: float) -> None:
    tensor = Tensor((1,), DType.bfloat16)
    tensor[0] = value
    expected = torch.tensor([value]).type(torch.bfloat16)
    # Torch rounds values up, whereas we currently truncate.
    # In particular this is an issue near infinity, as there's certain values
    # that torch will represent as inf, while we will instead represent them
    # as bfloat16_max.
    result = torch.from_dlpack(tensor)
    bf16info = torch.finfo(torch.bfloat16)
    if value > bf16info.max and math.isfinite(result.item()):
        assert result.item() == bf16info.max
    elif value < bf16info.min and math.isfinite(result.item()):
        assert result.item() == bf16info.min
    else:
        torch.testing.assert_close(expected, result, equal_nan=True)


@given(st.floats())
def test_getitem_bfloat16(value: float) -> None:
    torch_value = torch.tensor([value]).type(torch.bfloat16)
    tensor = Tensor.from_dlpack(torch_value)
    assert tensor.dtype == DType.bfloat16
    result = tensor[0].item()
    torch.testing.assert_close(torch_value.item(), result, equal_nan=True)


def test_device() -> None:
    # We should be able to set and query the device that a tensor is resident on.
    cpu = CPU()
    tensor = Tensor((3, 3), dtype=DType.int32, device=cpu)
    assert cpu == tensor.device


def test_to_numpy() -> None:
    # We should be able to convert a tensor to a numpy array.
    base_arr = np.arange(1, 6, dtype=np.int32)
    tensor = Tensor.from_numpy(base_arr)
    new_arr = tensor.to_numpy()
    assert np.array_equal(base_arr, new_arr)


def test_zeros() -> None:
    # We should be able to initialize an all-zero tensor.
    tensor = Tensor.zeros((3, 3), DType.int32)
    assert np.array_equal(tensor.to_numpy(), np.zeros((3, 3), dtype=np.int32))


def test_scalar() -> None:
    # We should be able to create scalar values.
    scalar = Tensor.scalar(5, DType.int32)
    assert scalar.item() == 5

    # We allow some ability to mutate scalars.
    scalar[0] = 8
    assert scalar.item() == 8


# NOTE: This is kept at function scope intentionally to avoid issues if tests
# mutate the stored data.
@pytest.fixture(scope="function")
def memmap_example_file():
    with tempfile.NamedTemporaryFile(mode="w+b") as f:
        f.write(b"\x00\x01\x02\x03\x04\x05\x06\x07")
        f.flush()
        yield Path(f.name)


def test_memmap(memmap_example_file: Path) -> None:
    tensor = MemMapTensor(memmap_example_file, dtype=DType.int8, shape=(2, 4))
    assert tensor.shape == (2, 4)
    assert tensor.dtype == DType.int8
    for i, j in product(range(2), range(4)):
        assert tensor[i, j].item() == i * 4 + j

    # Test that offsets work.
    offset_tensor = MemMapTensor(
        memmap_example_file, dtype=DType.int8, shape=(2, 3), offset=2, mode="r"
    )
    assert offset_tensor.shape == (2, 3)
    assert offset_tensor.dtype == DType.int8
    for i, j in product(range(2), range(3)):
        assert offset_tensor[i, j].item() == i * 3 + j + 2

    # Test that read-only arrays cannot be modified.
    with pytest.raises(ValueError):
        offset_tensor[0, 0] = 0

    # Test that a different type works and we can modify the array.
    tensor_16 = MemMapTensor(
        memmap_example_file, dtype=DType.int16, shape=(2,), offset=2, mode="r+"
    )
    tensor_16[0] = 0  # Intentional to avoid endianness issues.

    assert tensor[0, 1].item() == 1
    assert tensor[0, 2].item() == 0
    assert tensor[0, 3].item() == 0
    assert tensor[1, 0].item() == 4

    assert offset_tensor[0, 0].item() == 0
    assert offset_tensor[0, 1].item() == 0
    assert offset_tensor[0, 2].item() == 4


def test_dlpack_memmap(memmap_example_file: Path) -> None:
    tensor = MemMapTensor(memmap_example_file, dtype=DType.int8, shape=(2, 4))
    array = np.from_dlpack(tensor)
    assert array.dtype == np.int8
    assert tensor.shape == array.shape

    # Numpy creates a read-only array, so we modify ours.
    tensor[0, 0] = np.int8(8)
    assert array[0, 0] == np.int8(8)


def test_dlpack_memmap_view(memmap_example_file: Path) -> None:
    tensor = MemMapTensor(memmap_example_file, dtype=DType.int8, shape=(2, 4))
    tensor_view = tensor.view(DType.uint8)
    assert isinstance(tensor_view, MemMapTensor)

    array = np.from_dlpack(tensor_view)
    assert array.dtype == np.uint8
    assert tensor.shape == array.shape

    # Numpy creates a read-only array, so we modify ours.
    tensor[0, 0] = np.uint8(8)
    assert array[0, 0] == np.uint8(8)


def test_from_dlpack_memmap(memmap_example_file: Path) -> None:
    # We test that we can call from_dlpack on a read-only numpy memmap array.
    # TODO(MSDK-976): remove this test when we upgraded numpy to 2.1.
    array = np.memmap(memmap_example_file, dtype=np.int8, mode="r")
    assert not array.flags.writeable

    tensor = Tensor.from_dlpack(array)
    assert isinstance(tensor, MemMapTensor)
    assert array.dtype == np.int8
    assert tensor.shape == array.shape

    # Test that read-onlyness propagates.
    with pytest.raises(ValueError):
        tensor[0] = 0


def test_num_elements() -> None:
    tensor1 = Tensor((2, 4, 3), DType.int8)
    assert tensor1.num_elements == 24

    tensor2 = Tensor((1, 4), DType.int8)
    assert tensor2.num_elements == 4

    tensor3 = Tensor((), DType.int8)
    assert tensor3.num_elements == 1

    tensor4 = Tensor((1, 1, 1, 1, 1), DType.int8)
    assert tensor4.num_elements == 1


def test_element_size() -> None:
    for dtype in DLPACK_DTYPES:
        tensor = Tensor((), dtype)
        assert tensor.element_size == np.dtype(dtype.to_numpy()).itemsize


def test_view() -> None:
    tensor = Tensor((2, 6), DType.int8)
    for i in range(tensor.shape[0]):
        for j in range(tensor.shape[1]):
            tensor[i, j] = i * 10 + j
    assert tensor[0, 0].item() == 0
    assert tensor[1, 0].item() == 10
    assert tensor[1, 3].item() == 13
    assert tensor[1, 5].item() == 15
    # Check that the new shape is properly backed by the original
    tensor_view = tensor.view(DType.int8, (6, 2))
    assert tensor_view[0, 0].item() == 0
    assert tensor_view[1, 0].item() == 2
    assert tensor_view[3, 1].item() == 11
    assert tensor_view[5, 1].item() == 15

    tensor8 = Tensor((2, 4), DType.int8)
    for i, j in product(range(2), range(4)):
        tensor8[i, j] = 1

    # Check that we correctly deduce the shape if not given
    tensor16 = tensor8.view(DType.int16)
    assert tensor16.dtype is DType.int16
    assert tensor16.shape == (2, 2)
    assert tensor16[0, 0].item() == 2**8 + 1
    assert tensor16[0, 1].item() == 2**8 + 1

    # Check that it works with explicit shape.
    tensor32 = tensor8.view(DType.int32, (2,))
    assert tensor32.dtype is DType.int32
    assert tensor32.shape == (2,)
    assert tensor32[0].item() == 2**24 + 2**16 + 2**8 + 1

    # Check that this is not a copy.
    tensor16[0, 0] = 0
    assert tensor8[0, 0].item() == 0
    assert tensor8[0, 1].item() == 0

    # Check that shape deduction fails if the last axis is the wrong size.
    with pytest.raises(ValueError):
        _ = tensor8.view(DType.int64)


def test_from_dlpack_noncontiguous() -> None:
    array = np.arange(4).reshape(2, 2).transpose(1, 0)
    assert not array.flags.c_contiguous

    with pytest.raises(
        ValueError,
        match=r"from_dlpack only accepts contiguous arrays. First call np.ascontiguousarray",
    ):
        tensor = Tensor.from_dlpack(array)


def test_item_success() -> None:
    """Test successful item() calls for valid single-element tensors."""
    # Zero-rank case
    scalar = Tensor.scalar(8, DType.int32)
    assert scalar.item() == 8

    # Single-element tensors of various ranks
    for shape in [(), (1,), (1, 1), (1, 1, 1)]:
        tensor = Tensor(shape, DType.float32)
        tensor[tuple(0 for _ in shape)] = 3.14
        assert math.isclose(tensor.item(), 3.14, rel_tol=1e-6)


def test_item_multiple_elements() -> None:
    """Test item() fails when tensor contains multiple elements"""
    tensor = Tensor((2,), DType.int32)
    with pytest.raises(
        ValueError,
        match="calling `item` on a tensor with 2 items but expected only 1",
    ):
        tensor.item()


def test_aligned() -> None:
    tensor = Tensor((5,), DType.int32)
    assert tensor._aligned()
    assert tensor._aligned(DType.int32.align)

    tensor_uint8 = tensor.view(DType.uint8)
    assert tensor_uint8[1]._aligned()
    assert not tensor_uint8[1]._aligned(DType.int32.align)
