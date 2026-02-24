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

BINARY = [
    F.add,
    F.div,
    F.equal,
    F.greater,
    F.greater_equal,
    F.max,
    F.min,
    F.mod,
    F.mul,
    F.not_equal,
    F.pow,
    F.sub,
]

LOGICAL_BINARY = [
    F.logical_and,
    F.logical_or,
    F.logical_xor,
]


@pytest.mark.parametrize("op", BINARY)
def test_binary(op) -> None:  # noqa: ANN001
    a = Tensor.zeros([10, 1], dtype=DType.float32, device=DEVICE)
    b = Tensor.zeros([10], dtype=DType.float32, device=DEVICE)
    result = op(a, b)
    assert result.real
    assert list(result.driver_tensor.shape) == [10, 10]


@pytest.mark.parametrize("op", LOGICAL_BINARY)
def test_logical_binary(op) -> None:  # noqa: ANN001
    a = Tensor.full([10, 1], False, dtype=DType.bool, device=DEVICE)
    b = Tensor.full([10], False, dtype=DType.bool, device=DEVICE)
    result = op(a, b)
    assert result.real
    assert list(result.driver_tensor.shape) == [10, 10]
