# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

from __future__ import annotations

import pytest
from conftest import assert_all_close
from max.driver import CPU
from max.dtype import DType
from max.experimental import random
from max.experimental.tensor import Tensor


def test_normal():
    t = Tensor.arange(10, dtype=DType.float32, device=CPU())
    assert_all_close(list(range(10)), t)


def test_uniform():
    t1 = random.uniform([20], dtype=DType.float32, device=CPU())
    t2 = random.uniform([20], dtype=DType.float32, device=CPU())

    with pytest.raises(AssertionError):
        assert_all_close(t1, t2)
