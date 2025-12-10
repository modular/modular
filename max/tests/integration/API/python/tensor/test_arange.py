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

from __future__ import annotations

import pytest
from conftest import assert_all_close
from max.driver import CPU
from max.dtype import DType
from max.experimental.tensor import Tensor, default_dtype


def test_range_like() -> None:
    t = Tensor.ones([3, 4, 5], dtype=DType.float32, device=CPU())
    t2 = Tensor.range_like(t.type)
    assert t.type == t2.type
    assert_all_close(range(5), t2[0, 0, :])
    assert_all_close(range(5), t2[1, 2, :])


def test_arange() -> None:
    t = Tensor.arange(10, dtype=DType.float32, device=CPU())
    assert_all_close(range(10), t)


def test_arange_defaults() -> None:
    with default_dtype(DType.float32):
        t = Tensor.arange(10)
        assert_all_close(range(10), t)


def test_invalid() -> None:
    t = Tensor.arange(10, dtype=DType.float32, device=CPU())
    with pytest.raises(
        AssertionError, match=r"atol: tensors not close at index 0, 2.0 > 1e-06"
    ):
        assert_all_close(range(2, 12), t)
