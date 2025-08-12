# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
"""Smoke tests for ops in `max.experimental.functional`.

These tests exercise each expected op at least once with real data and kernels.
They don't otherwise make any attempt at coverage, edge cases, or correctness.
"""

import pytest
from max.driver import CPU, Accelerator, accelerator_count
from max.dtype import DType
from max.experimental import functional as F
from max.experimental.tensor import Tensor

DEVICE = Accelerator() if accelerator_count() else CPU()

UNARY = [
    F.abs,
    F.argsort,
    F.atanh,
    F.cos,
    F.cumsum,
    F.erf,
    F.exp,
    F.floor,
    F.gelu,
    F.is_inf,
    F.is_nan,
    F.log,
    F.log1p,
    # TODO(KERNELS-1976): Implement on GPU
    # F.logsoftmax,
    F.negate,
    F.relu,
    F.round,
    F.rsqrt,
    F.sigmoid,
    F.silu,
    F.sin,
    F.softmax,
    F.sqrt,
    F.tanh,
    F.trunc,
]

LOGICAL_UNARY = [
    F.logical_not,
]


@pytest.mark.parametrize("op", UNARY)
def test_unary(op):  # noqa: ANN001
    tensor = Tensor.zeros([10], dtype=DType.float32, device=DEVICE)
    result = op(tensor)
    result._sync_realize()
    assert result.real
    assert list(result.driver_tensor.shape) == tensor.shape


@pytest.mark.parametrize("op", LOGICAL_UNARY)
def test_logical_unary(op):  # noqa: ANN001
    tensor = Tensor.full([10], False, dtype=DType.bool, device=DEVICE)
    result = op(tensor)
    result._sync_realize()
    assert result.real
    assert list(result.driver_tensor.shape) == tensor.shape
