# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
"""Test max.driver Tensors."""
import max.driver as md
import pytest


def test_tensor():
    # Validate that metadata shows up correctly
    tensor = md.Tensor((3, 4, 5), md.DType.float32)
    assert md.DType.float32 == tensor.dtype
    assert "DType.float32" == str(tensor.dtype)
    assert (3, 4, 5) == tensor.shape
    assert 3 == tensor.rank


def test_get_and_set():
    tensor = md.Tensor((3, 4, 5), md.DType.int32)
    tensor[0, 1, 3] = 68
    # Get should return zero-d tensor
    elt = tensor[0, 1, 3]
    assert 0 == elt.rank
    assert () == elt.shape
    assert md.DType.int32 == elt.dtype
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
    tensor = md.Tensor((3, 3, 3), md.DType.int32)
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
    tensor = md.Tensor((5, 5, 5), md.DType.int32)
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
    tensor = md.Tensor((3, 3), md.DType.int32)
    elt = 0
    for x in range(3):
        for y in range(3):
            tensor[x, y] = elt
            elt += 1
    elt -= 1
    # Tensors should support slices with negative steps.
    revtensor = tensor[::-1, ::-1]
    for x in range(3):
        for y in range(3):
            assert revtensor[x, y].item() == elt
            elt -= 1


def test_out_of_bounds_slices():
    tensor = md.Tensor((3, 3, 3), md.DType.int32)

    # Out of bounds indexes are allowed in slices.
    assert tensor[4:, :2, 8:10:-1].shape == (0, 2, 0)

    # Out of bounds indexes are not allowed in integral indexing.
    with pytest.raises(IndexError):
        tensor[4:, :2, 4]


def test_one_dimensional_tensor():
    tensor = md.Tensor((10,), md.DType.int32)
    for i in range(10):
        tensor[i] = i

    for i in range(i):
        assert tensor[i].item() == i
