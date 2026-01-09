# ===----------------------------------------------------------------------=== #
# Copyright (c) 2025, Modular Inc. All rights reserved.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions:
# https://llvm.org/LICENSE.txt
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ===----------------------------------------------------------------------=== #
"""Smoke tests for methods on `max.experimental.tensor.Tensor`.

These tests exercise each expected op at least once with real data and kernels.
They don't otherwise make any attempt at coverage, edge cases, or correctness.
"""

from __future__ import annotations

import weakref

from conftest import assert_all_close
from max.driver import CPU, Accelerator, accelerator_count
from max.dtype import DType
from max.experimental import random
from max.experimental.tensor import Tensor, default_dtype

# Do not do this because this test is parallelized at pytest's level. Each test should create its own accelerator.
ACCELERATOR = Accelerator() if accelerator_count() else CPU()


def test_ones_defaults() -> None:
    with default_dtype(DType.float32):
        t = Tensor.ones([10])
        assert_all_close([1] * 10, t)


def test_zeros_like() -> None:
    ref = Tensor.ones(
        [4, 6],
        dtype=DType.float32,
        device=Accelerator() if accelerator_count() else CPU(),
    )
    result = Tensor.zeros_like(ref)
    result._sync_realize()
    assert result.real
    assert list(result.driver_tensor.shape) == [4, 6]
    assert result.dtype == DType.float32


def test_ones_like() -> None:
    ref = Tensor.zeros(
        [4, 6],
        dtype=DType.float32,
        device=Accelerator() if accelerator_count() else CPU(),
    )
    result = Tensor.ones_like(ref)
    result._sync_realize()
    assert result.real
    assert list(result.driver_tensor.shape) == [4, 6]
    assert result.dtype == DType.float32


def test_full_like() -> None:
    ref = Tensor.zeros(
        [4, 6],
        dtype=DType.float32,
        device=Accelerator() if accelerator_count() else CPU(),
    )
    result = Tensor.full_like(ref, value=42.0)
    result._sync_realize()
    assert result.real
    assert list(result.driver_tensor.shape) == [4, 6]
    assert result.dtype == DType.float32


def test_abs() -> None:
    tensor = Tensor.ones(
        [4, 6],
        dtype=DType.float32,
        device=Accelerator() if accelerator_count() else CPU(),
    )
    result = abs(tensor)
    result._sync_realize()
    assert result.real


def test_argmax() -> None:
    tensor = Tensor.ones(
        [4, 6],
        dtype=DType.float32,
        device=Accelerator() if accelerator_count() else CPU(),
    )
    result = tensor.argmax()
    result._sync_realize()
    assert result.real


def test_max() -> None:
    tensor = Tensor.ones(
        [4, 6],
        dtype=DType.float32,
        device=Accelerator() if accelerator_count() else CPU(),
    )
    result = tensor.max()
    result._sync_realize()
    assert result.real


def test_mean() -> None:
    tensor = Tensor.ones(
        [4, 6],
        dtype=DType.float32,
        device=Accelerator() if accelerator_count() else CPU(),
    )
    result = tensor.mean()
    result._sync_realize()
    assert result.real


def test_sum() -> None:
    tensor = Tensor.constant(
        [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
        dtype=DType.float32,
        device=Accelerator() if accelerator_count() else CPU(),
    )
    # Sum along last axis (rows)
    row_sum = tensor.sum(axis=-1)
    row_sum._sync_realize()
    assert row_sum.real
    assert list(row_sum.shape) == [2, 1]
    # Values should be [6.0, 15.0]
    values = list(row_sum._values())
    assert abs(values[0] - 6.0) < 1e-5
    assert abs(values[1] - 15.0) < 1e-5


def test_clip() -> None:
    x = random.normal([20])
    assert all((x.clip(max=0.0) <= 0.0)._values())
    assert all((x.clip(min=0.0) >= 0.0)._values())
    assert all(-0.5 <= v <= 0.5 for v in x.clip(min=-0.5, max=0.5)._values())


def test_squeeze() -> None:
    tensor = Tensor.ones(
        [4, 1, 6],
        dtype=DType.float32,
        device=Accelerator() if accelerator_count() else CPU(),
    )
    result = tensor.squeeze(axis=1)
    result._sync_realize()
    assert result.real
    assert list(result.driver_tensor.shape) == [4, 6]


def test_unsqueeze() -> None:
    tensor = Tensor.ones(
        [4, 6],
        dtype=DType.float32,
        device=Accelerator() if accelerator_count() else CPU(),
    )
    # Unsqueeze at the end
    result = tensor.unsqueeze(axis=-1)
    result._sync_realize()
    assert result.real
    assert list(result.driver_tensor.shape) == [4, 6, 1]

    # Unsqueeze at the beginning
    result2 = tensor.unsqueeze(axis=0)
    result2._sync_realize()
    assert result2.real
    assert list(result2.driver_tensor.shape) == [1, 4, 6]


def test_split_with_int() -> None:
    """Test split with int split_size (PyTorch-style)."""
    t = Tensor.ones(
        [10, 4],
        dtype=DType.float32,
        device=Accelerator() if accelerator_count() else CPU(),
    )
    # Split into chunks of size 3 (last chunk will be size 1)
    chunks = t.split(3, axis=0)
    assert len(chunks) == 4
    for chunk in chunks:
        chunk._sync_realize()
        assert chunk.real
    assert list(chunks[0].driver_tensor.shape) == [3, 4]
    assert list(chunks[1].driver_tensor.shape) == [3, 4]
    assert list(chunks[2].driver_tensor.shape) == [3, 4]
    assert list(chunks[3].driver_tensor.shape) == [1, 4]


def test_split_with_list() -> None:
    """Test split with list of sizes."""
    t = Tensor.ones(
        [10, 4],
        dtype=DType.float32,
        device=Accelerator() if accelerator_count() else CPU(),
    )
    # Split into exact sizes
    chunks = t.split([2, 3, 5], axis=0)
    assert len(chunks) == 3
    for chunk in chunks:
        chunk._sync_realize()
        assert chunk.real
    assert list(chunks[0].driver_tensor.shape) == [2, 4]
    assert list(chunks[1].driver_tensor.shape) == [3, 4]
    assert list(chunks[2].driver_tensor.shape) == [5, 4]


def test_reshape() -> None:
    tensor = Tensor.ones(
        [4, 6],
        dtype=DType.float32,
        device=Accelerator() if accelerator_count() else CPU(),
    )
    result = tensor.reshape([6, 4])
    result._sync_realize()
    assert result.real
    assert list(result.driver_tensor.shape) == [6, 4]


def test_cast() -> None:
    tensor = Tensor.ones(
        [4, 6],
        dtype=DType.float32,
        device=Accelerator() if accelerator_count() else CPU(),
    )
    result = tensor.cast(DType.int64)
    result._sync_realize()
    assert result.real
    assert result.dtype == DType.int64


def test_permute() -> None:
    tensor = Tensor.ones(
        [2, 3, 4],
        dtype=DType.float32,
        device=Accelerator() if accelerator_count() else CPU(),
    )
    result = tensor.permute([2, 0, 1])
    result._sync_realize()
    assert result.real
    assert list(result.driver_tensor.shape) == [4, 2, 3]


def test_transpose() -> None:
    tensor = Tensor.ones(
        [4, 6],
        dtype=DType.float32,
        device=Accelerator() if accelerator_count() else CPU(),
    )
    result = tensor.transpose(0, 1)
    result._sync_realize()
    assert result.real
    assert list(result.driver_tensor.shape) == [6, 4]


def test_T_property() -> None:
    tensor = Tensor.ones(
        [4, 6],
        dtype=DType.float32,
        device=Accelerator() if accelerator_count() else CPU(),
    )
    result = tensor.T
    result._sync_realize()
    assert result.real
    assert list(result.driver_tensor.shape) == [6, 4]


def test_getitem() -> None:
    tensor = Tensor.ones(
        [4, 6],
        dtype=DType.float32,
        device=Accelerator() if accelerator_count() else CPU(),
    )
    result = tensor[0:2, 1:4]
    result._sync_realize()
    assert result.real


def test_neg() -> None:
    tensor = Tensor.ones(
        [4, 6],
        dtype=DType.float32,
        device=Accelerator() if accelerator_count() else CPU(),
    )
    result = -tensor
    result._sync_realize()
    assert result.real


def test_eq() -> None:
    a = Tensor.ones(
        [4, 6],
        dtype=DType.float32,
        device=Accelerator() if accelerator_count() else CPU(),
    )
    b = Tensor.ones(
        [4, 6],
        dtype=DType.float32,
        device=Accelerator() if accelerator_count() else CPU(),
    )
    result = a == b
    result._sync_realize()
    assert result.real


def test_ne() -> None:
    a = Tensor.ones(
        [4, 6],
        dtype=DType.float32,
        device=Accelerator() if accelerator_count() else CPU(),
    )
    b = Tensor.zeros(
        [4, 6],
        dtype=DType.float32,
        device=Accelerator() if accelerator_count() else CPU(),
    )
    result = a != b
    result._sync_realize()
    assert result.real


def test_ge() -> None:
    a = Tensor.ones(
        [4, 6],
        dtype=DType.float32,
        device=Accelerator() if accelerator_count() else CPU(),
    )
    b = Tensor.zeros(
        [4, 6],
        dtype=DType.float32,
        device=Accelerator() if accelerator_count() else CPU(),
    )
    result = a >= b
    result._sync_realize()
    assert result.real


def test_gt() -> None:
    a = Tensor.ones(
        [4, 6],
        dtype=DType.float32,
        device=Accelerator() if accelerator_count() else CPU(),
    )
    b = Tensor.zeros(
        [4, 6],
        dtype=DType.float32,
        device=Accelerator() if accelerator_count() else CPU(),
    )
    result = a > b
    result._sync_realize()
    assert result.real


def test_le() -> None:
    a = Tensor.zeros(
        [4, 6],
        dtype=DType.float32,
        device=Accelerator() if accelerator_count() else CPU(),
    )
    b = Tensor.ones(
        [4, 6],
        dtype=DType.float32,
        device=Accelerator() if accelerator_count() else CPU(),
    )
    result = a <= b
    result._sync_realize()
    assert result.real


def test_lt() -> None:
    a = Tensor.zeros(
        [4, 6],
        dtype=DType.float32,
        device=Accelerator() if accelerator_count() else CPU(),
    )
    b = Tensor.ones(
        [4, 6],
        dtype=DType.float32,
        device=Accelerator() if accelerator_count() else CPU(),
    )
    result = a < b
    result._sync_realize()
    assert result.real


def test_add() -> None:
    a = Tensor.ones(
        [4, 6],
        dtype=DType.float32,
        device=Accelerator() if accelerator_count() else CPU(),
    )
    b = Tensor.ones(
        [4, 6],
        dtype=DType.float32,
        device=Accelerator() if accelerator_count() else CPU(),
    )
    result = a + b
    result._sync_realize()
    assert result.real


def test_radd() -> None:
    a = Tensor.ones(
        [4, 6],
        dtype=DType.float32,
        device=Accelerator() if accelerator_count() else CPU(),
    )
    result = 2.0 + a  # triggers __radd__
    result._sync_realize()
    assert result.real


def test_sub() -> None:
    a = Tensor.ones(
        [4, 6],
        dtype=DType.float32,
        device=Accelerator() if accelerator_count() else CPU(),
    )
    b = Tensor.ones(
        [4, 6],
        dtype=DType.float32,
        device=Accelerator() if accelerator_count() else CPU(),
    )
    result = a - b
    result._sync_realize()
    assert result.real


def test_rsub() -> None:
    a = Tensor.ones(
        [4, 6],
        dtype=DType.float32,
        device=Accelerator() if accelerator_count() else CPU(),
    )
    result = 2.0 - a  # triggers __rsub__
    result._sync_realize()
    assert result.real


def test_mul() -> None:
    a = Tensor.ones(
        [4, 6],
        dtype=DType.float32,
        device=Accelerator() if accelerator_count() else CPU(),
    )
    b = Tensor.ones(
        [4, 6],
        dtype=DType.float32,
        device=Accelerator() if accelerator_count() else CPU(),
    )
    result = a * b
    result._sync_realize()
    assert result.real


def test_rmul() -> None:
    a = Tensor.ones(
        [4, 6],
        dtype=DType.float32,
        device=Accelerator() if accelerator_count() else CPU(),
    )
    result = 2.0 * a  # triggers __rmul__
    result._sync_realize()
    assert result.real


def test_truediv() -> None:
    a = Tensor.ones(
        [4, 6],
        dtype=DType.float32,
        device=Accelerator() if accelerator_count() else CPU(),
    )
    b = Tensor.ones(
        [4, 6],
        dtype=DType.float32,
        device=Accelerator() if accelerator_count() else CPU(),
    )
    result = a / b
    result._sync_realize()
    assert result.real


def test_rtruediv() -> None:
    a = Tensor.ones(
        [4, 6],
        dtype=DType.float32,
        device=Accelerator() if accelerator_count() else CPU(),
    )
    result = 2.0 / a  # triggers __rtruediv__
    result._sync_realize()
    assert result.real


def test_floordiv() -> None:
    a = Tensor.ones(
        [4, 6],
        dtype=DType.float32,
        device=Accelerator() if accelerator_count() else CPU(),
    )
    b = Tensor.ones(
        [4, 6],
        dtype=DType.float32,
        device=Accelerator() if accelerator_count() else CPU(),
    )
    result = a // b
    result._sync_realize()
    assert result.real


def test_rfloordiv() -> None:
    a = Tensor.ones(
        [4, 6],
        dtype=DType.float32,
        device=Accelerator() if accelerator_count() else CPU(),
    )
    result = 2.0 // a  # triggers __rfloordiv__
    result._sync_realize()
    assert result.real


def test_mod() -> None:
    a = Tensor.ones(
        [4, 6],
        dtype=DType.float32,
        device=Accelerator() if accelerator_count() else CPU(),
    )
    b = Tensor.ones(
        [4, 6],
        dtype=DType.float32,
        device=Accelerator() if accelerator_count() else CPU(),
    )
    result = a % b
    result._sync_realize()
    assert result.real


def test_rmod() -> None:
    a = Tensor.ones(
        [4, 6],
        dtype=DType.float32,
        device=Accelerator() if accelerator_count() else CPU(),
    )
    result = 2.0 % a  # triggers __rmod__
    result._sync_realize()
    assert result.real


def test_divmod() -> None:
    a = Tensor.ones(
        [4, 6],
        dtype=DType.float32,
        device=Accelerator() if accelerator_count() else CPU(),
    )
    b = Tensor.ones(
        [4, 6],
        dtype=DType.float32,
        device=Accelerator() if accelerator_count() else CPU(),
    )
    quotient, remainder = divmod(a, b)
    quotient._sync_realize()
    remainder._sync_realize()
    assert quotient.real
    assert remainder.real


def test_rdivmod() -> None:
    a = Tensor.ones(
        [4, 6],
        dtype=DType.float32,
        device=Accelerator() if accelerator_count() else CPU(),
    )
    # Call __rdivmod__ explicitly, divmod(2.0, a) is typed improperly
    quotient, remainder = a.__rdivmod__(2.0)
    quotient._sync_realize()
    remainder._sync_realize()
    assert quotient.real
    assert remainder.real


def test_matmul() -> None:
    a = Tensor.ones(
        [4, 3],
        dtype=DType.float32,
        device=Accelerator() if accelerator_count() else CPU(),
    )
    b = Tensor.ones(
        [3, 6],
        dtype=DType.float32,
        device=Accelerator() if accelerator_count() else CPU(),
    )
    result = a @ b
    result._sync_realize()
    assert result.real
    assert list(result.driver_tensor.shape) == [4, 6]


def test_rmatmul() -> None:
    a = Tensor.ones(
        [4, 3],
        dtype=DType.float32,
        device=Accelerator() if accelerator_count() else CPU(),
    )
    b = Tensor.ones(
        [3, 6],
        dtype=DType.float32,
        device=Accelerator() if accelerator_count() else CPU(),
    )
    # a @ b would call __matmul__, so call __rmatmal__ explicitly
    result = b.__rmatmul__(a)
    result._sync_realize()
    assert result.real


def test_pow() -> None:
    a = Tensor.ones(
        [4, 6],
        dtype=DType.float32,
        device=Accelerator() if accelerator_count() else CPU(),
    )
    b = Tensor.ones(
        [4, 6],
        dtype=DType.float32,
        device=Accelerator() if accelerator_count() else CPU(),
    )
    result = a**b
    result._sync_realize()
    assert result.real


def test_rpow() -> None:
    a = Tensor.ones(
        [4, 6],
        dtype=DType.float32,
        device=Accelerator() if accelerator_count() else CPU(),
    )
    result = 2.0**a  # triggers __rpow__
    result._sync_realize()
    assert result.real


def test_and() -> None:
    a = Tensor.full(
        [4, 6],
        True,
        dtype=DType.bool,
        device=Accelerator() if accelerator_count() else CPU(),
    )
    b = Tensor.full(
        [4, 6],
        False,
        dtype=DType.bool,
        device=Accelerator() if accelerator_count() else CPU(),
    )
    result = a & b
    result._sync_realize()
    assert result.real


def test_rand() -> None:
    a = Tensor.full(
        [4, 6],
        True,
        dtype=DType.bool,
        device=Accelerator() if accelerator_count() else CPU(),
    )
    result = True & a  # triggers __rand__
    result._sync_realize()
    assert result.real


def test_or() -> None:
    a = Tensor.full(
        [4, 6],
        True,
        dtype=DType.bool,
        device=Accelerator() if accelerator_count() else CPU(),
    )
    b = Tensor.full(
        [4, 6],
        False,
        dtype=DType.bool,
        device=Accelerator() if accelerator_count() else CPU(),
    )
    result = a | b
    result._sync_realize()
    assert result.real


def test_ror() -> None:
    a = Tensor.full(
        [4, 6],
        True,
        dtype=DType.bool,
        device=Accelerator() if accelerator_count() else CPU(),
    )
    result = False | a  # triggers __ror__
    result._sync_realize()
    assert result.real


def test_xor() -> None:
    a = Tensor.full(
        [4, 6],
        True,
        dtype=DType.bool,
        device=Accelerator() if accelerator_count() else CPU(),
    )
    b = Tensor.full(
        [4, 6],
        False,
        dtype=DType.bool,
        device=Accelerator() if accelerator_count() else CPU(),
    )
    result = a ^ b
    result._sync_realize()
    assert result.real


def test_rxor() -> None:
    a = Tensor.full(
        [4, 6],
        True,
        dtype=DType.bool,
        device=Accelerator() if accelerator_count() else CPU(),
    )
    result = False ^ a  # triggers __rxor__
    result._sync_realize()
    assert result.real


def test_invert() -> None:
    a = Tensor.full(
        [4, 6],
        True,
        dtype=DType.bool,
        device=Accelerator() if accelerator_count() else CPU(),
    )
    result = ~a
    result._sync_realize()
    assert result.real


def test_dead_sources_freed() -> None:
    a = Tensor.ones([1])
    a._sync_realize()
    weak_a = weakref.ref(a)
    a *= 2
    a._sync_realize()
    assert a.item() == 2
    assert weak_a() is None
