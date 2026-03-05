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
"""Tests for MAXBaseModel equality behavior."""

from __future__ import annotations

import pytest
from max.config.base_model import MAXBaseModel
from pydantic import computed_field


class _ModelWithRaisingComputedField(MAXBaseModel):
    """A model whose computed field always raises, to test __eq__ safety."""

    value: int

    @computed_field  # type: ignore[prop-decorator]
    @property
    def always_explodes(self) -> str:
        raise RuntimeError("computed field evaluated — __eq__ must not do this")


def test_eq_does_not_raise_with_exploding_computed_field() -> None:
    """__eq__ must not evaluate computed fields, even if they raise."""
    a = _ModelWithRaisingComputedField(value=42)
    b = _ModelWithRaisingComputedField(value=42)

    # Sanity-check: accessing the computed field really does raise.
    with pytest.raises(RuntimeError, match="computed field evaluated"):
        _ = a.always_explodes

    # Equality must NOT raise.
    assert a == b


def test_ne_does_not_raise_with_exploding_computed_field() -> None:
    """__ne__ must not evaluate computed fields either."""
    a = _ModelWithRaisingComputedField(value=1)
    b = _ModelWithRaisingComputedField(value=2)

    assert a != b


def test_eq_compares_regular_fields() -> None:
    """Equality is based on non-computed fields."""
    a = _ModelWithRaisingComputedField(value=7)
    b = _ModelWithRaisingComputedField(value=7)
    c = _ModelWithRaisingComputedField(value=8)

    assert a == b
    assert a != c
