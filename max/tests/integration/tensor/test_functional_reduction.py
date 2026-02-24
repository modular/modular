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

REDUCTION = [
    F.argmax,
    F.argmin,
    F.mean,
    F.prod,
    F.sum,
]


@pytest.mark.parametrize("op", REDUCTION)
def test_reduction(op) -> None:  # noqa: ANN001
    tensor = Tensor.zeros([10, 10], dtype=DType.float32, device=DEVICE)
    result = op(tensor, axis=-1)
    assert result.real
    assert list(result.driver_tensor.shape) == [10, 1]
