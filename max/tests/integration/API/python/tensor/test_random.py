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
from max.experimental.tensor import default_device


@pytest.mark.skip("GEX-2512: these return the same value :facepalm:")
def test_normal() -> None:
    t1 = random.normal([20], dtype=DType.float32, device=CPU())
    t2 = random.normal([20], dtype=DType.float32, device=CPU())

    with pytest.raises(AssertionError):
        assert_all_close(t1, t2)


@pytest.mark.skip("GEX-2512: these return the same value :facepalm:")
def test_normal_defaults() -> None:
    # `normal` not implemented on GPU yet
    with default_device(CPU()):
        t1 = random.normal([20])
        t2 = random.normal([20])

    with pytest.raises(AssertionError):
        assert_all_close(t1, t2)


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
