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
"""Smoke tests for methods on `max.experimental.tensor.Tensor`.

These tests exercise each expected op at least once with real data and kernels.
They don't otherwise make any attempt at coverage, edge cases, or correctness.
"""

from __future__ import annotations

from max.driver import CPU, Accelerator, accelerator_count
from max.dtype import DType
from max.experimental.tensor import Tensor


def test_matmul() -> None:
    a = Tensor.ones(
        [4, 3],
        dtype=DType.float32,
        device=Accelerator() if accelerator_count() else CPU(),
    )
    b = Tensor.ones(
        [3, 6],
        dtype=DType.float32,
        device=Accelerator() if accelerator_count() else CPU(),
    )
    result = a @ b
    assert result.real
    assert list(result.driver_tensor.shape) == [4, 6]


def test_rmatmul() -> None:
    a = Tensor.ones(
        [4, 3],
        dtype=DType.float32,
        device=Accelerator() if accelerator_count() else CPU(),
    )
    b = Tensor.ones(
        [3, 6],
        dtype=DType.float32,
        device=Accelerator() if accelerator_count() else CPU(),
    )
    # a @ b would call __matmul__, so call __rmatmal__ explicitly
    result = b.__rmatmul__(a)
    assert result.real
