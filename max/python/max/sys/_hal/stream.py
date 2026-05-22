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
"""``Stream`` — Python projection of HAL ``Stream``."""

from __future__ import annotations

from typing import Any

from .event import Event


class Stream:
    """An in-order command stream bound to a ``Context``.

    Operations submitted to a Stream complete in submission order. Each op
    implicitly waits for the previous one to finish.

    Not constructed directly; obtain via ``context.create_stream()``.
    """

    _inner: Any

    __slots__ = ("_inner",)

    def __init__(self) -> None:
        raise TypeError(
            "Stream is not directly constructible; use Context.create_stream()"
        )

    @classmethod
    def _wrap(cls, inner: object) -> Stream:
        obj = cls.__new__(cls)
        obj._inner = inner
        return obj

    def synchronize(self) -> None:
        self._inner.synchronize()

    def record_event(self) -> Event:
        return Event._wrap(self._inner.record_event())

    def __repr__(self) -> str:
        return "Stream()"

    __str__ = __repr__
