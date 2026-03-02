# ===----------------------------------------------------------------------=== #
# Copyright (c) 2026, Modular Inc. All rights reserved.
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

from max.driver import CPU, Accelerator, accelerator_count
from max.dtype import DType
from max.experimental import random
from max.experimental.tensor import Tensor


def test_abs() -> None:
    tensor = Tensor.ones(
        [4, 6],
        dtype=DType.float32,
        device=Accelerator() if accelerator_count() else CPU(),
    )
    result = abs(tensor)
    assert result.real


def test_argmax() -> None:
    tensor = Tensor.ones(
        [4, 6],
        dtype=DType.float32,
        device=Accelerator() if accelerator_count() else CPU(),
    )
    result = tensor.argmax()
    assert result.real


def test_max() -> None:
    tensor = Tensor.ones(
        [4, 6],
        dtype=DType.float32,
        device=Accelerator() if accelerator_count() else CPU(),
    )
    result = tensor.max()
    assert result.real


def test_min() -> None:
    tensor = Tensor.ones(
        [4, 6],
        dtype=DType.float32,
        device=Accelerator() if accelerator_count() else CPU(),
    )
    result = tensor.min()
    assert result.real


def test_mean() -> None:
    tensor = Tensor.ones(
        [4, 6],
        dtype=DType.float32,
        device=Accelerator() if accelerator_count() else CPU(),
    )
    result = tensor.mean()
    assert result.real


def test_sum() -> None:
    tensor = Tensor.constant(
        [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
        dtype=DType.float32,
        device=Accelerator() if accelerator_count() else CPU(),
    )
    # Sum along last axis (rows)
    row_sum = tensor.sum(axis=-1)
    assert row_sum.real
    assert list(row_sum.shape) == [2, 1]
    # Values should be [6.0, 15.0]
    values = list(row_sum._values())
    assert abs(values[0] - 6.0) < 1e-5
    assert abs(values[1] - 15.0) < 1e-5


def test_prod() -> None:
    tensor = Tensor.constant(
        [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
        dtype=DType.float32,
        device=Accelerator() if accelerator_count() else CPU(),
    )
    # Product along last axis (rows)
    row_prod = tensor.prod(axis=-1)
    assert row_prod.real
    assert list(row_prod.shape) == [2, 1]
    # Values should be [6.0, 120.0]
    values = list(row_prod._values())
    assert abs(values[0] - 6.0) < 1e-5
    assert abs(values[1] - 120.0) < 1e-5


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
    assert result.real
    assert list(result.driver_tensor.shape) == [4, 6, 1]

    # Unsqueeze at the beginning
    result2 = tensor.unsqueeze(axis=0)
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
    assert result.real
    assert list(result.driver_tensor.shape) == [6, 4]


def test_cast() -> None:
    tensor = Tensor.ones(
        [4, 6],
        dtype=DType.float32,
        device=Accelerator() if accelerator_count() else CPU(),
    )
    result = tensor.cast(DType.int64)
    assert result.real
    assert result.dtype == DType.int64


def test_permute() -> None:
    tensor = Tensor.ones(
        [2, 3, 4],
        dtype=DType.float32,
        device=Accelerator() if accelerator_count() else CPU(),
    )
    result = tensor.permute([2, 0, 1])
    assert result.real
    assert list(result.driver_tensor.shape) == [4, 2, 3]


def test_transpose() -> None:
    tensor = Tensor.ones(
        [4, 6],
        dtype=DType.float32,
        device=Accelerator() if accelerator_count() else CPU(),
    )
    result = tensor.transpose(0, 1)
    assert result.real
    assert list(result.driver_tensor.shape) == [6, 4]


def test_T_property() -> None:
    tensor = Tensor.ones(
        [4, 6],
        dtype=DType.float32,
        device=Accelerator() if accelerator_count() else CPU(),
    )
    result = tensor.T
    assert result.real
    assert list(result.driver_tensor.shape) == [6, 4]


def test_getitem() -> None:
    tensor = Tensor.ones(
        [4, 6],
        dtype=DType.float32,
        device=Accelerator() if accelerator_count() else CPU(),
    )
    result = tensor[0:2, 1:4]
    assert result.real


def test_neg() -> None:
    tensor = Tensor.ones(
        [4, 6],
        dtype=DType.float32,
        device=Accelerator() if accelerator_count() else CPU(),
    )
    result = -tensor
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
    assert result.real


def test_radd() -> None:
    a = Tensor.ones(
        [4, 6],
        dtype=DType.float32,
        device=Accelerator() if accelerator_count() else CPU(),
    )
    result = 2.0 + a  # triggers __radd__
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
    assert result.real


def test_rsub() -> None:
    a = Tensor.ones(
        [4, 6],
        dtype=DType.float32,
        device=Accelerator() if accelerator_count() else CPU(),
    )
    result = 2.0 - a  # triggers __rsub__
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
    assert result.real


def test_rmul() -> None:
    a = Tensor.ones(
        [4, 6],
        dtype=DType.float32,
        device=Accelerator() if accelerator_count() else CPU(),
    )
    result = 2.0 * a  # triggers __rmul__
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
    assert result.real


def test_rtruediv() -> None:
    a = Tensor.ones(
        [4, 6],
        dtype=DType.float32,
        device=Accelerator() if accelerator_count() else CPU(),
    )
    result = 2.0 / a  # triggers __rtruediv__
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
    assert result.real


def test_rfloordiv() -> None:
    a = Tensor.ones(
        [4, 6],
        dtype=DType.float32,
        device=Accelerator() if accelerator_count() else CPU(),
    )
    result = 2.0 // a  # triggers __rfloordiv__
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
    assert result.real


def test_rmod() -> None:
    a = Tensor.ones(
        [4, 6],
        dtype=DType.float32,
        device=Accelerator() if accelerator_count() else CPU(),
    )
    result = 2.0 % a  # triggers __rmod__
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
    assert quotient.real
    assert remainder.real


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
    assert result.real


def test_rpow() -> None:
    a = Tensor.ones(
        [4, 6],
        dtype=DType.float32,
        device=Accelerator() if accelerator_count() else CPU(),
    )
    result = 2.0**a  # triggers __rpow__
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
    assert result.real


def test_rand() -> None:
    a = Tensor.full(
        [4, 6],
        True,
        dtype=DType.bool,
        device=Accelerator() if accelerator_count() else CPU(),
    )
    result = True & a  # triggers __rand__
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
    assert result.real


def test_ror() -> None:
    a = Tensor.full(
        [4, 6],
        True,
        dtype=DType.bool,
        device=Accelerator() if accelerator_count() else CPU(),
    )
    result = False | a  # triggers __ror__
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
    assert result.real


def test_rxor() -> None:
    a = Tensor.full(
        [4, 6],
        True,
        dtype=DType.bool,
        device=Accelerator() if accelerator_count() else CPU(),
    )
    result = False ^ a  # triggers __rxor__
    assert result.real


def test_invert() -> None:
    a = Tensor.full(
        [4, 6],
        True,
        dtype=DType.bool,
        device=Accelerator() if accelerator_count() else CPU(),
    )
    result = ~a
    assert result.real


def test_max_axis_none() -> None:
    """Test that tensor.max with axis=None reduces over all dimensions."""
    data = [[1.2, 3.5, 2.1], [2.3, 1.9, 4.2]]
    tensor = Tensor.constant(
        data,
        dtype=DType.float32,
        device=Accelerator() if accelerator_count() else CPU(),
    )
    result = tensor.max(axis=None)
    result._sync_realize()
    assert result.shape == [1]
    expected_max = 4.2
    result_value = result.item()
    assert abs(result_value - expected_max) < 1e-5


def test_min_axis_none() -> None:
    """Test that tensor.min with axis=None reduces over all dimensions."""
    data = [[1.2, 3.5, 2.1], [2.3, 1.9, 4.2]]
    tensor = Tensor.constant(
        data,
        dtype=DType.float32,
        device=Accelerator() if accelerator_count() else CPU(),
    )
    result = tensor.min(axis=None)
    result._sync_realize()
    assert result.shape == [1]
    expected_min = 1.2
    result_value = result.item()
    assert abs(result_value - expected_min) < 1e-5


def test_mean_axis_none() -> None:
    """Test that tensor.mean with axis=None reduces over all dimensions."""
    data = [[2.0, 4.0, 6.0], [8.0, 10.0, 12.0]]
    tensor = Tensor.constant(
        data,
        dtype=DType.float32,
        device=Accelerator() if accelerator_count() else CPU(),
    )
    result = tensor.mean(axis=None)
    result._sync_realize()
    assert result.shape == [1]
    # Mean of [2, 4, 6, 8, 10, 12] = 42/6 = 7.0
    expected_mean = 7.0
    result_value = result.item()
    assert abs(result_value - expected_mean) < 1e-5


def test_argmax_axis_none() -> None:
    """Test that tensor.argmax with axis=None returns flattened index."""
    data = [[1.2, 3.5, 2.1], [2.3, 1.9, 4.2]]
    tensor = Tensor.constant(
        data,
        dtype=DType.float32,
        device=Accelerator() if accelerator_count() else CPU(),
    )
    result = tensor.argmax(axis=None)
    result._sync_realize()
    assert result.shape == [1]
    # The maximum value 4.2 is at position [1, 2]
    # Flattened index = 1*3 + 2 = 5
    expected_index = 5
    result_value = result.item()
    assert result_value == expected_index
