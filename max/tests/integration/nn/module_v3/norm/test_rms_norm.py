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
"""Tests for max.nn.norm.rms_norm."""

from __future__ import annotations

import pytest
from max.experimental.tensor import Tensor
from max.nn.module_v3.norm import GemmaRMSNorm, RMSNorm


def test_repr() -> None:
    assert repr(RMSNorm(2)) == "RMSNorm(dim=Dim(2))"
    assert repr(RMSNorm(2, 1e-7) == "RMSNorm(dim=Dim(2), eps=1e-7)")


def test_dim() -> None:
    assert RMSNorm(2).dim == 2


def test_parameters() -> None:
    norm = RMSNorm(2)
    assert dict(norm.parameters) == {"weight": norm.weight}


def test_call() -> None:
    norm = RMSNorm(2)

    assert norm(Tensor.ones([2])).shape == [2]
    assert norm(Tensor.ones([10, 2])).shape == [10, 2]

    with pytest.raises(ValueError):
        norm(Tensor.ones([1]))

    with pytest.raises(ValueError):
        norm(Tensor.ones([10, 1]))


def test_gemma_repr() -> None:
    assert repr(GemmaRMSNorm(2)) == "GemmaRMSNorm(dim=Dim(2))"
    assert repr(GemmaRMSNorm(2, 1e-7) == "GemmaRMSNorm(dim=Dim(2), eps=1e-7)")


def test_gemma_dim() -> None:
    assert GemmaRMSNorm(2).dim == 2


def test_gemma_parameters() -> None:
    norm = GemmaRMSNorm(2)
    assert dict(norm.parameters) == {"weight": norm.weight}


def test_gemma_call() -> None:
    norm = GemmaRMSNorm(2)

    assert norm(Tensor.ones([2])).shape == [2]
    assert norm(Tensor.ones([10, 2])).shape == [10, 2]

    with pytest.raises(ValueError):
        norm(Tensor.ones([1]))

    with pytest.raises(ValueError):
        norm(Tensor.ones([10, 1]))
