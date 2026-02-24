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
"""Tests for common_layers.activation."""

from __future__ import annotations

from functools import partial

import pytest
from max.experimental import functional as F
from max.pipelines.architectures.common_layers.activation import (
    activation_function_from_name,
)


@pytest.mark.parametrize(
    "name,expected",
    [
        ("silu", F.silu),
        ("swish", F.silu),
        ("gelu", F.gelu),
        ("relu", F.relu),
        ("tanh", F.tanh),
        ("sigmoid", F.sigmoid),
    ],
)
def test_returns_expected_function(name: str, expected: object) -> None:
    assert activation_function_from_name(name) is expected


@pytest.mark.parametrize(
    "name,approximate",
    [
        ("gelu_tanh", "tanh"),
        ("gelu_quick", "quick"),
        ("quick_gelu", "quick"),
    ],
)
def test_gelu_variants(name: str, approximate: str) -> None:
    result = activation_function_from_name(name)
    assert isinstance(result, partial)
    assert result.func is F.gelu
    assert result.keywords == {"approximate": approximate}


def test_unrecognized_raises() -> None:
    with pytest.raises(
        ValueError, match="Unrecognized activation function name"
    ):
        activation_function_from_name("nonexistent")
