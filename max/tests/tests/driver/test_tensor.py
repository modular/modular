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
    assert 68 == tensor[0, 1, 3]

    # Setting negative indices
    tensor[-1, -1, -1] = 72
    assert 72 == tensor[2, 3, 4]

    # Ensure we're not writing to the same memory location with each index
    assert 68 == tensor[0, 1, 3]

    # Cannot use non-integers when indexing
    with pytest.raises(TypeError):
        tensor[4.2, 2, 2] = 23

    # Cannot do out-of-bounds indexing
    with pytest.raises(IndexError):
        tensor[5, 2, 2] = 23

    # Cannot do out-of-bounds with negative indices
    with pytest.raises(IndexError):
        tensor[-4, -3, -3] = 42
