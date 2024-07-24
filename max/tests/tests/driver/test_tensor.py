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
    tensor = md.Tensor(md.float32, (3, 4, 5))
    assert md.float32 == tensor.dtype
    assert "max.float32" == str(tensor.dtype)
    assert (3, 4, 5) == tensor.shape
    assert 3 == tensor.rank


def test_get_and_set():
    tensor = md.Tensor(md.int32, (3, 4, 5))
    tensor[0, 1, 3] = 68
    # Get should return zero-d tensor
    elt = tensor[0, 1, 3]
    assert 0 == elt.rank
    assert () == elt.shape
    assert md.int32 == elt.dtype
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
