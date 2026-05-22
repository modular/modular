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
"""``Event`` — Python projection of HAL ``Event``."""

from __future__ import annotations

from typing import Any


class Event:
    """A synchronization event recorded on a ``Queue`` or ``Stream``.

    Always created with CPU-visible flags so ``synchronize()`` and
    ``is_ready()`` are callable from the host.
    """

    _inner: Any

    __slots__ = ("_inner",)

    def __init__(self) -> None:
        raise TypeError(
            "Event is not directly constructible; use Queue.record_event() "
            "or Stream.record_event()"
        )

    @classmethod
    def _wrap(cls, inner: object) -> Event:
        obj = cls.__new__(cls)
        obj._inner = inner
        return obj

    def synchronize(self) -> None:
        self._inner.synchronize()

    def is_ready(self) -> bool:
        return self._inner.is_ready()

    def __repr__(self) -> str:
        return "Event()"

    __str__ = __repr__
