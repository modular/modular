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
from max.experimental import random


def test_normal() -> None:
    t1 = random.normal([20], dtype=DType.float32, device=CPU())
    t2 = random.normal([20], dtype=DType.float32, device=CPU())

    with pytest.raises(AssertionError):
        assert_all_close(t1, t2)


def test_normal_defaults() -> None:
    t1 = random.normal([20])
    t2 = random.normal([20])

    with pytest.raises(AssertionError):
        assert_all_close(t1, t2)


def test_normal_different_dtype() -> None:
    t1 = random.normal([20], dtype=DType.float32)
    t2 = random.normal([20], dtype=DType.float32)

    with pytest.raises(AssertionError):
        assert_all_close(t1, t2)


def test_normal_integral() -> None:
    t = random.normal([20], dtype=DType.uint8)
    _ = repr(t)


def test_uniform() -> None:
    t1 = random.uniform([20], dtype=DType.float32, device=CPU())
    t2 = random.uniform([20], dtype=DType.float32, device=CPU())

    with pytest.raises(AssertionError):
        assert_all_close(t1, t2)


def test_uniform_defaults() -> None:
    t1 = random.uniform([20])
    t2 = random.uniform([20])

    with pytest.raises(AssertionError):
        assert_all_close(t1, t2)
