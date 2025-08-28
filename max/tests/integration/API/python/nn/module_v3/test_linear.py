# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
"""Tests for max.nn.module_v3.Linear."""

from __future__ import annotations

from max.experimental.tensor import Tensor
from max.nn.module_v3 import Linear


def test_repr():
    assert repr(Linear(2, 2)) == "Linear(in_dim=Dim(2), out_dim=Dim(2))"
    assert (
        repr(Linear(1, 3, bias=False))
        == "Linear(in_dim=Dim(1), out_dim=Dim(3), bias=False)"
    )


def test_in_dim():
    assert Linear(2, 3).in_dim == 2


def test_out_dim():
    assert Linear(2, 3).out_dim == 3


def test_parameters():
    linear = Linear(2, 3)
    assert dict(linear.parameters) == {
        "weight": linear.weight,
        "bias": linear.bias,
    }


def test_call():
    linear = Linear(2, 3)
    result = linear(Tensor.ones([2]))
    assert result.shape == [3]
