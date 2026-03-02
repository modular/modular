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
"""Tests for max.nn.Linear."""

from __future__ import annotations

from max.experimental.tensor import Tensor
from max.nn.module_v3 import Linear


def test_repr() -> None:
    assert repr(Linear(2, 2)) == "Linear(in_dim=Dim(2), out_dim=Dim(2))"
    assert (
        repr(Linear(1, 3, bias=False))
        == "Linear(in_dim=Dim(1), out_dim=Dim(3), bias=False)"
    )


def test_in_dim() -> None:
    assert Linear(2, 3).in_dim == 2


def test_out_dim() -> None:
    assert Linear(2, 3).out_dim == 3


def test_parameters() -> None:
    linear = Linear(2, 3)
    assert dict(linear.parameters) == {
        "weight": linear.weight,
        "bias": linear.bias,
    }


def test_call() -> None:
    linear = Linear(2, 3)
    result = linear(Tensor.ones([2]))
    assert result.shape == [3]
