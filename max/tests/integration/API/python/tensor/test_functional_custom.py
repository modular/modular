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
"""Smoke tests for ops in `max.experimental.functional`.

These tests exercise each expected op at least once with real data and kernels.
They don't otherwise make any attempt at coverage, edge cases, or correctness.
"""

import pytest
from max.driver import CPU, Accelerator, accelerator_count
from max.dtype import DType
from max.experimental import functional as F
from max.experimental.tensor import Tensor
from max.nn import kernels

DEVICE = Accelerator() if accelerator_count() else CPU()

moe_create_indices = F.functional(kernels.moe_create_indices)
scatter_set_constant = F.functional(kernels.scatter_set_constant)


@pytest.mark.skipif(
    DEVICE.is_host, reason="moe_create_indices only supports GPU devices"
)
def test_custom() -> None:
    indices = Tensor.ones([4], dtype=DType.int32, device=DEVICE)
    token_expert_order, *_rest = moe_create_indices(indices, 8)
    assert token_expert_order.real


@pytest.mark.skipif(
    DEVICE.is_host, reason="scatter_set_constant only supports GPU devices"
)
def test_inplace_custom() -> None:
    values = Tensor.zeros([2, 2])
    indices = Tensor.ones([1, 1], dtype=DType.int32)
    scatter_set_constant(values, indices, 5.0)
    assert values[1, 0].item() == 5.0
    assert values.real
    scatter_set_constant(values, indices, 4.0)
    assert not values.real
    assert values[1, 0].item() == 4.0
    assert values.real
