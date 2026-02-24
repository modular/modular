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
"""Tests for max.nn.activation."""

from __future__ import annotations

from functools import partial

import pytest
from max.graph import ops
from max.nn.activation import activation_function_from_name


@pytest.mark.parametrize(
    "name,expected",
    [
        ("silu", ops.silu),
        ("gelu", ops.gelu),
        ("relu", ops.relu),
        ("tanh", ops.tanh),
        ("sigmoid", ops.sigmoid),
    ],
)
def test_returns_expected_function(name: str, expected: object) -> None:
    assert activation_function_from_name(name) is expected


def test_gelu_tanh() -> None:
    result = activation_function_from_name("gelu_tanh")
    assert isinstance(result, partial)
    assert result.func is ops.gelu
    assert result.keywords == {"approximate": "tanh"}


def test_unrecognized_raises() -> None:
    with pytest.raises(
        ValueError, match="Unrecognized activation function name"
    ):
        activation_function_from_name("nonexistent")
