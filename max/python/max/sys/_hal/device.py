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
"""``Device`` — Python projection of HAL ``Device``."""

from __future__ import annotations

from typing import Any

from .context import Context


class Device:
    """A device retrieved from a ``Driver``.

    Not constructed directly; obtain via ``driver.get_device(id)``.
    """

    _inner: Any

    __slots__ = ("_inner",)

    def __init__(self) -> None:
        raise TypeError(
            "Device is not directly constructible; use Driver.get_device(id)"
        )

    @classmethod
    def _wrap(cls, inner: object) -> Device:
        obj = cls.__new__(cls)
        obj._inner = inner
        return obj

    @property
    def id(self) -> int:
        return self._inner.get_id()

    def get_context(self) -> Context:
        return Context._wrap(self._inner.get_context())

    def __repr__(self) -> str:
        return f"Device(id={self.id})"

    __str__ = __repr__
