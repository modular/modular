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
"""Tests elementwise operations with DTensor."""

from __future__ import annotations

import pytest
from conftest import mesh_1d
from max.dtype import DType
from max.experimental.distributed import DTensor, Replicated, Sharded

NOT_IMPLEMENTED = r"does not support .* yet"


def _replicated_dt() -> DTensor:
    return DTensor.distributed_ones(
        [4, 6], mesh_1d(2), [Replicated()], dtype=DType.float32
    )


def _sharded_dt() -> DTensor:
    return DTensor.distributed_ones(
        [4, 6], mesh_1d(2), [Sharded(axis=0)], dtype=DType.float32
    )


# -- Arithmetic ops --


class TestArithmeticOps:
    def test_add(self) -> None:
        a, b = _replicated_dt(), _replicated_dt()
        with pytest.raises(NotImplementedError, match=NOT_IMPLEMENTED):
            a + b

    def test_radd(self) -> None:
        a = _replicated_dt()
        with pytest.raises(NotImplementedError, match=NOT_IMPLEMENTED):
            2.0 + a

    def test_sub(self) -> None:
        a, b = _replicated_dt(), _replicated_dt()
        with pytest.raises(NotImplementedError, match=NOT_IMPLEMENTED):
            a - b

    def test_mul(self) -> None:
        a, b = _replicated_dt(), _replicated_dt()
        with pytest.raises(NotImplementedError, match=NOT_IMPLEMENTED):
            a * b

    def test_truediv(self) -> None:
        a, b = _replicated_dt(), _replicated_dt()
        with pytest.raises(NotImplementedError, match=NOT_IMPLEMENTED):
            a / b

    def test_floordiv(self) -> None:
        a, b = _replicated_dt(), _replicated_dt()
        with pytest.raises(NotImplementedError, match=NOT_IMPLEMENTED):
            a // b

    def test_mod(self) -> None:
        a, b = _replicated_dt(), _replicated_dt()
        with pytest.raises(NotImplementedError, match=NOT_IMPLEMENTED):
            a % b

    def test_pow(self) -> None:
        a, b = _replicated_dt(), _replicated_dt()
        with pytest.raises(NotImplementedError, match=NOT_IMPLEMENTED):
            a**b

    def test_neg(self) -> None:
        a = _replicated_dt()
        with pytest.raises(NotImplementedError, match=NOT_IMPLEMENTED):
            _ = -a

    def test_abs(self) -> None:
        a = _replicated_dt()
        with pytest.raises(NotImplementedError, match=NOT_IMPLEMENTED):
            _ = abs(a)

    def test_matmul(self) -> None:
        a, b = _replicated_dt(), _replicated_dt()
        with pytest.raises(NotImplementedError, match=NOT_IMPLEMENTED):
            _ = a @ b


# -- Comparison ops --


class TestComparisonOps:
    def test_eq(self) -> None:
        a, b = _replicated_dt(), _replicated_dt()
        with pytest.raises(NotImplementedError, match=NOT_IMPLEMENTED):
            _ = a == b

    def test_ne(self) -> None:
        a, b = _replicated_dt(), _replicated_dt()
        with pytest.raises(NotImplementedError, match=NOT_IMPLEMENTED):
            _ = a != b

    def test_gt(self) -> None:
        a, b = _replicated_dt(), _replicated_dt()
        with pytest.raises(NotImplementedError, match=NOT_IMPLEMENTED):
            _ = a > b

    def test_ge(self) -> None:
        a, b = _replicated_dt(), _replicated_dt()
        with pytest.raises(NotImplementedError, match=NOT_IMPLEMENTED):
            _ = a >= b

    def test_lt(self) -> None:
        a, b = _replicated_dt(), _replicated_dt()
        with pytest.raises(NotImplementedError, match=NOT_IMPLEMENTED):
            _ = a < b

    def test_le(self) -> None:
        a, b = _replicated_dt(), _replicated_dt()
        with pytest.raises(NotImplementedError, match=NOT_IMPLEMENTED):
            _ = a <= b


# -- Reduction ops --


class TestReductionOps:
    def test_argmax(self) -> None:
        a = _replicated_dt()
        with pytest.raises(NotImplementedError, match=NOT_IMPLEMENTED):
            a.argmax()

    def test_max(self) -> None:
        a = _replicated_dt()
        with pytest.raises(NotImplementedError, match=NOT_IMPLEMENTED):
            a.max()

    def test_min(self) -> None:
        a = _replicated_dt()
        with pytest.raises(NotImplementedError, match=NOT_IMPLEMENTED):
            a.min()

    def test_mean(self) -> None:
        a = _replicated_dt()
        with pytest.raises(NotImplementedError, match=NOT_IMPLEMENTED):
            a.mean()

    def test_sum(self) -> None:
        a = _replicated_dt()
        with pytest.raises(NotImplementedError, match=NOT_IMPLEMENTED):
            a.sum()

    def test_prod(self) -> None:
        a = _replicated_dt()
        with pytest.raises(NotImplementedError, match=NOT_IMPLEMENTED):
            a.prod()


# -- Shape ops --


class TestShapeOps:
    def test_reshape(self) -> None:
        a = _replicated_dt()
        with pytest.raises(NotImplementedError, match=NOT_IMPLEMENTED):
            a.reshape([6, 4])

    def test_squeeze(self) -> None:
        dt = DTensor.distributed_ones(
            [4, 1, 6], mesh_1d(2), [Replicated()], dtype=DType.float32
        )
        with pytest.raises(NotImplementedError, match=NOT_IMPLEMENTED):
            dt.squeeze(axis=1)

    def test_unsqueeze(self) -> None:
        a = _replicated_dt()
        with pytest.raises(NotImplementedError, match=NOT_IMPLEMENTED):
            a.unsqueeze(axis=0)

    def test_permute(self) -> None:
        a = _replicated_dt()
        with pytest.raises(NotImplementedError, match=NOT_IMPLEMENTED):
            a.permute([1, 0])

    def test_transpose(self) -> None:
        a = _replicated_dt()
        with pytest.raises(NotImplementedError, match=NOT_IMPLEMENTED):
            a.transpose(0, 1)

    def test_T(self) -> None:
        a = _replicated_dt()
        with pytest.raises(NotImplementedError, match=NOT_IMPLEMENTED):
            _ = a.T

    def test_cast(self) -> None:
        a = _replicated_dt()
        with pytest.raises(NotImplementedError, match=NOT_IMPLEMENTED):
            a.cast(DType.int64)

    def test_split(self) -> None:
        a = _replicated_dt()
        with pytest.raises(NotImplementedError, match=NOT_IMPLEMENTED):
            a.split(2, axis=0)

    def test_broadcast_to(self) -> None:
        a = _replicated_dt()
        with pytest.raises(NotImplementedError, match=NOT_IMPLEMENTED):
            a.broadcast_to([4, 6])


# -- Sharded DTensor raises the same errors --


class TestShardedDTensorOps:
    def test_add(self) -> None:
        a, b = _sharded_dt(), _sharded_dt()
        with pytest.raises(NotImplementedError, match=NOT_IMPLEMENTED):
            a + b

    def test_argmax(self) -> None:
        a = _sharded_dt()
        with pytest.raises(NotImplementedError, match=NOT_IMPLEMENTED):
            a.argmax()

    def test_reshape(self) -> None:
        a = _sharded_dt()
        with pytest.raises(NotImplementedError, match=NOT_IMPLEMENTED):
            a.reshape([6, 4])
