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
"""``Buffer`` — Python projection of HAL ``Buffer``."""

from __future__ import annotations

from typing import Any


class Buffer:
    """A device (or host-pinned) memory allocation.

    Owns the underlying allocation: dropping the Python ``Buffer``
    frees the device memory via the HAL's ``free_sync`` /
    ``free_host_pinned`` path. The parent ``Context`` is held alive
    internally for the buffer's lifetime, so it is safe to drop the
    Python ``Context`` handle while buffers obtained from it are still
    in use.

    Not constructed directly; obtain via ``context.alloc_sync(n)`` or
    ``context.alloc_host_pinned(n)``.
    """

    _inner: Any

    __slots__ = ("_inner",)

    def __init__(self) -> None:
        raise TypeError(
            "Buffer is not directly constructible; use "
            "Context.alloc_sync(byte_size) or "
            "Context.alloc_host_pinned(byte_size)"
        )

    @classmethod
    def _wrap(cls, inner: object) -> Buffer:
        obj = cls.__new__(cls)
        obj._inner = inner
        return obj

    @property
    def byte_size(self) -> int:
        return self._inner.get_byte_size()

    def __repr__(self) -> str:
        return f"Buffer(byte_size={self.byte_size})"

    __str__ = __repr__
