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
"""Tests for max.experimental.nn.norm.group_norm."""

from __future__ import annotations

import pytest
from max.driver import Accelerator, accelerator_count
from max.experimental.nn.norm import GroupNorm
from max.experimental.nn.norm.group_norm import group_norm
from max.experimental.tensor import Tensor


def test_repr() -> None:
    assert "num_groups=4" in repr(GroupNorm(4, 8))
    assert "num_channels=8" in repr(GroupNorm(4, 8))


def test_invalid_init() -> None:
    with pytest.raises(ValueError, match="divisible"):
        GroupNorm(5, 11)


def test_parameters_affine() -> None:
    norm = GroupNorm(4, 8, affine=True)
    params = dict(norm.parameters)
    assert "weight" in params
    assert "bias" in params


def test_parameters_no_affine() -> None:
    norm = GroupNorm(4, 8, affine=False)
    params = dict(norm.parameters)
    assert len(params) == 0


@pytest.mark.skipif(not accelerator_count(), reason="requires accelerator")
@pytest.mark.parametrize(
    "num_channels,num_groups,affine",
    [
        (4, 2, True),
        (8, 4, True),
        (6, 3, False),
    ],
)
def test_group_norm_call(
    num_channels: int, num_groups: int, affine: bool
) -> None:
    """Verify the experimental GroupNorm compiles and runs on GPU."""
    device = Accelerator()
    norm = GroupNorm(num_groups, num_channels, affine=affine).to(device)
    x = Tensor.ones([2, num_channels, 4], device=device)
    result = norm(x)
    assert result.shape == [2, num_channels, 4]


@pytest.mark.skipif(not accelerator_count(), reason="requires accelerator")
def test_group_norm_functional() -> None:
    """Verify the functional group_norm API works end-to-end."""
    device = Accelerator()
    x = Tensor.ones([1, 4, 8], device=device)
    weight = Tensor.ones([4], device=device)
    bias = Tensor.zeros([4], device=device)
    result = group_norm(x, weight, bias, num_groups=2, epsilon=1e-5)
    assert result.shape == [1, 4, 8]
