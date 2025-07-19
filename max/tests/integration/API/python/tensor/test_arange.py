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
from max.experimental.tensor import Tensor


def test_arange() -> None:
    t = Tensor.arange(10, dtype=DType.float32, device=CPU())
    assert_all_close(list(range(10)), t)


def test_invalid() -> None:
    t = Tensor.arange(10, dtype=DType.float32, device=CPU())
    with pytest.raises(
        AssertionError, match="atol: tensors not close at index 0, 2.0 > 1e-06"
    ):
        assert_all_close(list(range(2, 12)), t)
