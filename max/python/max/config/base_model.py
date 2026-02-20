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
"""Shared base model utilities for MAX pydantic models."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict


class MAXBaseModel(BaseModel):
    """Base class for MAX pydantic models.

    We intentionally define structural equality in terms of declared public model
    fields (via `model_dump`) so objects compare equal across config
    roundtrips/pickling and are not sensitive to pydantic internal bookkeeping.
    """

    model_config = ConfigDict(extra="forbid", strict=True)

    def __eq__(self, other: object) -> bool:
        """Structural equality based on public model fields."""
        if not isinstance(other, type(self)):
            return NotImplemented
        return self.model_dump(mode="python") == other.model_dump(mode="python")

    def __ne__(self, other: object) -> bool:
        """Negation of `__eq__` with consistent NotImplemented behavior."""
        eq_result = self.__eq__(other)
        if eq_result is NotImplemented:
            return NotImplemented
        return not eq_result
