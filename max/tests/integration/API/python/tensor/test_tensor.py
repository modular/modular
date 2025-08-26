# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
"""Smoke tests for methods on `max.experimental.tensor.Tensor`.

These tests exercise each expected op at least once with real data and kernels.
They don't otherwise make any attempt at coverage, edge cases, or correctness.
"""

from __future__ import annotations

from conftest import assert_all_close
from max.driver import CPU, Accelerator, accelerator_count
from max.dtype import DType
from max.experimental.tensor import Tensor, default_dtype

DEVICE = Accelerator() if accelerator_count() else CPU()


def test_ones_defaults() -> None:
    with default_dtype(DType.float32):
        t = Tensor.ones([10])
        assert_all_close(list([1] * 10), t)


def test_abs():
    tensor = Tensor.ones([4, 6], dtype=DType.float32, device=DEVICE)
    result = abs(tensor)
    result._sync_realize()
    assert result.real


def test_argmax():
    tensor = Tensor.ones([4, 6], dtype=DType.float32, device=DEVICE)
    result = tensor.argmax()
    result._sync_realize()
    assert result.real


def test_max():
    tensor = Tensor.ones([4, 6], dtype=DType.float32, device=DEVICE)
    result = tensor.max()
    result._sync_realize()
    assert result.real


def test_reshape():
    tensor = Tensor.ones([4, 6], dtype=DType.float32, device=DEVICE)
    result = tensor.reshape([6, 4])
    result._sync_realize()
    assert result.real
    assert list(result.driver_tensor.shape) == [6, 4]


def test_cast():
    tensor = Tensor.ones([4, 6], dtype=DType.float32, device=DEVICE)
    result = tensor.cast(DType.int64)
    result._sync_realize()
    assert result.real
    assert result.dtype == DType.int64


def test_permute():
    tensor = Tensor.ones([2, 3, 4], dtype=DType.float32, device=DEVICE)
    result = tensor.permute([2, 0, 1])
    result._sync_realize()
    assert result.real
    assert list(result.driver_tensor.shape) == [4, 2, 3]


def test_transpose():
    tensor = Tensor.ones([4, 6], dtype=DType.float32, device=DEVICE)
    result = tensor.transpose(0, 1)
    result._sync_realize()
    assert result.real
    assert list(result.driver_tensor.shape) == [6, 4]


def test_T_property():
    tensor = Tensor.ones([4, 6], dtype=DType.float32, device=DEVICE)
    result = tensor.T
    result._sync_realize()
    assert result.real
    assert list(result.driver_tensor.shape) == [6, 4]


def test_getitem():
    tensor = Tensor.ones([4, 6], dtype=DType.float32, device=DEVICE)
    result = tensor[0:2, 1:4]
    result._sync_realize()
    assert result.real


def test_neg():
    tensor = Tensor.ones([4, 6], dtype=DType.float32, device=DEVICE)
    result = -tensor
    result._sync_realize()
    assert result.real


def test_eq():
    a = Tensor.ones([4, 6], dtype=DType.float32, device=DEVICE)
    b = Tensor.ones([4, 6], dtype=DType.float32, device=DEVICE)
    result = a == b
    result._sync_realize()
    assert result.real


def test_ne():
    a = Tensor.ones([4, 6], dtype=DType.float32, device=DEVICE)
    b = Tensor.zeros([4, 6], dtype=DType.float32, device=DEVICE)
    result = a != b
    result._sync_realize()
    assert result.real


def test_ge():
    a = Tensor.ones([4, 6], dtype=DType.float32, device=DEVICE)
    b = Tensor.zeros([4, 6], dtype=DType.float32, device=DEVICE)
    result = a >= b
    result._sync_realize()
    assert result.real


def test_gt():
    a = Tensor.ones([4, 6], dtype=DType.float32, device=DEVICE)
    b = Tensor.zeros([4, 6], dtype=DType.float32, device=DEVICE)
    result = a > b
    result._sync_realize()
    assert result.real


def test_le():
    a = Tensor.zeros([4, 6], dtype=DType.float32, device=DEVICE)
    b = Tensor.ones([4, 6], dtype=DType.float32, device=DEVICE)
    result = a <= b
    result._sync_realize()
    assert result.real


def test_lt():
    a = Tensor.zeros([4, 6], dtype=DType.float32, device=DEVICE)
    b = Tensor.ones([4, 6], dtype=DType.float32, device=DEVICE)
    result = a < b
    result._sync_realize()
    assert result.real


def test_add():
    a = Tensor.ones([4, 6], dtype=DType.float32, device=DEVICE)
    b = Tensor.ones([4, 6], dtype=DType.float32, device=DEVICE)
    result = a + b
    result._sync_realize()
    assert result.real


def test_radd():
    a = Tensor.ones([4, 6], dtype=DType.float32, device=DEVICE)
    result = 2.0 + a  # triggers __radd__
    result._sync_realize()
    assert result.real


def test_sub():
    a = Tensor.ones([4, 6], dtype=DType.float32, device=DEVICE)
    b = Tensor.ones([4, 6], dtype=DType.float32, device=DEVICE)
    result = a - b
    result._sync_realize()
    assert result.real


def test_rsub():
    a = Tensor.ones([4, 6], dtype=DType.float32, device=DEVICE)
    result = 2.0 - a  # triggers __rsub__
    result._sync_realize()
    assert result.real


def test_mul():
    a = Tensor.ones([4, 6], dtype=DType.float32, device=DEVICE)
    b = Tensor.ones([4, 6], dtype=DType.float32, device=DEVICE)
    result = a * b
    result._sync_realize()
    assert result.real


def test_rmul():
    a = Tensor.ones([4, 6], dtype=DType.float32, device=DEVICE)
    result = 2.0 * a  # triggers __rmul__
    result._sync_realize()
    assert result.real


def test_truediv():
    a = Tensor.ones([4, 6], dtype=DType.float32, device=DEVICE)
    b = Tensor.ones([4, 6], dtype=DType.float32, device=DEVICE)
    result = a / b
    result._sync_realize()
    assert result.real


def test_rtruediv():
    a = Tensor.ones([4, 6], dtype=DType.float32, device=DEVICE)
    result = 2.0 / a  # triggers __rtruediv__
    result._sync_realize()
    assert result.real


def test_floordiv():
    a = Tensor.ones([4, 6], dtype=DType.float32, device=DEVICE)
    b = Tensor.ones([4, 6], dtype=DType.float32, device=DEVICE)
    result = a // b
    result._sync_realize()
    assert result.real


def test_rfloordiv():
    a = Tensor.ones([4, 6], dtype=DType.float32, device=DEVICE)
    result = 2.0 // a  # triggers __rfloordiv__
    result._sync_realize()
    assert result.real


def test_mod():
    a = Tensor.ones([4, 6], dtype=DType.float32, device=DEVICE)
    b = Tensor.ones([4, 6], dtype=DType.float32, device=DEVICE)
    result = a % b
    result._sync_realize()
    assert result.real


def test_rmod():
    a = Tensor.ones([4, 6], dtype=DType.float32, device=DEVICE)
    result = 2.0 % a  # triggers __rmod__
    result._sync_realize()
    assert result.real


def test_divmod():
    a = Tensor.ones([4, 6], dtype=DType.float32, device=DEVICE)
    b = Tensor.ones([4, 6], dtype=DType.float32, device=DEVICE)
    quotient, remainder = divmod(a, b)
    quotient._sync_realize()
    remainder._sync_realize()
    assert quotient.real
    assert remainder.real


def test_rdivmod():
    a = Tensor.ones([4, 6], dtype=DType.float32, device=DEVICE)
    # Call __rdivmod__ explicitly, divmod(2.0, a) is typed improperly
    quotient, remainder = a.__rdivmod__(2.0)
    quotient._sync_realize()
    remainder._sync_realize()
    assert quotient.real
    assert remainder.real


def test_matmul():
    a = Tensor.ones([4, 3], dtype=DType.float32, device=DEVICE)
    b = Tensor.ones([3, 6], dtype=DType.float32, device=DEVICE)
    result = a @ b
    result._sync_realize()
    assert result.real
    assert list(result.driver_tensor.shape) == [4, 6]


def test_rmatmul():
    a = Tensor.ones([4, 3], dtype=DType.float32, device=DEVICE)
    b = Tensor.ones([3, 6], dtype=DType.float32, device=DEVICE)
    # a @ b would call __matmul__, so call __rmatmal__ explicitly
    result = b.__rmatmul__(a)
    result._sync_realize()
    assert result.real


def test_pow():
    a = Tensor.ones([4, 6], dtype=DType.float32, device=DEVICE)
    b = Tensor.ones([4, 6], dtype=DType.float32, device=DEVICE)
    result = a**b
    result._sync_realize()
    assert result.real


def test_rpow():
    a = Tensor.ones([4, 6], dtype=DType.float32, device=DEVICE)
    result = 2.0**a  # triggers __rpow__
    result._sync_realize()
    assert result.real


def test_and():
    a = Tensor.full([4, 6], True, dtype=DType.bool, device=DEVICE)
    b = Tensor.full([4, 6], False, dtype=DType.bool, device=DEVICE)
    result = a & b
    result._sync_realize()
    assert result.real


def test_rand():
    a = Tensor.full([4, 6], True, dtype=DType.bool, device=DEVICE)
    result = True & a  # triggers __rand__
    result._sync_realize()
    assert result.real


def test_or():
    a = Tensor.full([4, 6], True, dtype=DType.bool, device=DEVICE)
    b = Tensor.full([4, 6], False, dtype=DType.bool, device=DEVICE)
    result = a | b
    result._sync_realize()
    assert result.real


def test_ror():
    a = Tensor.full([4, 6], True, dtype=DType.bool, device=DEVICE)
    result = False | a  # triggers __ror__
    result._sync_realize()
    assert result.real


def test_xor():
    a = Tensor.full([4, 6], True, dtype=DType.bool, device=DEVICE)
    b = Tensor.full([4, 6], False, dtype=DType.bool, device=DEVICE)
    result = a ^ b
    result._sync_realize()
    assert result.real


def test_rxor():
    a = Tensor.full([4, 6], True, dtype=DType.bool, device=DEVICE)
    result = False ^ a  # triggers __rxor__
    result._sync_realize()
    assert result.real


def test_invert():
    a = Tensor.full([4, 6], True, dtype=DType.bool, device=DEVICE)
    result = ~a
    result._sync_realize()
    assert result.real
