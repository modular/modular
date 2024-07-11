# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
"""Test max.driver Tensors."""
import max.driver as md


def test_tensor():
    tensor = md.Tensor(md.dtype.float32, (3, 4, 5))
    assert md.dtype.float32 == tensor.dtype
    assert (3, 4, 5) == tensor.shape
    assert 3 == tensor.rank
