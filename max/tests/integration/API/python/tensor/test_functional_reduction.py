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

REDUCTION = [
    F.argmax,
    F.argmin,
    F.mean,
    F.sum,
]


@pytest.mark.parametrize("op", REDUCTION)
def test_reduction(op):  # noqa: ANN001
    tensor = Tensor.zeros([10, 10], dtype=DType.float32, device=DEVICE)
    result = op(tensor, axis=-1)
    result._sync_realize()
    assert result.real
    assert list(result.driver_tensor.shape) == [10, 1]
