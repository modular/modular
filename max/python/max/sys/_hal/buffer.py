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
"""``Buffer`` and ``BufferView`` — Python projections of the HAL types."""

from __future__ import annotations

from typing import Any


class BufferView:
    """A non-owning view over a (sub-range of a) device allocation.

    Carries a byte offset and size into a parent ``Buffer``. It does
    **not** keep that buffer alive: the caller is responsible for
    ensuring the source ``Buffer`` outlives the view and any operation
    that uses it.

    Not constructed directly; obtain via ``buffer.view()`` (the whole
    allocation) or ``buffer.view(byte_offset=..., byte_size=...)``
    (a sub-range).
    """

    _inner: Any

    __slots__ = ("_inner",)

    def __init__(self) -> None:
        raise TypeError(
            "BufferView is not directly constructible; use buffer.view()"
        )

    @classmethod
    def _wrap(cls, inner: object) -> BufferView:
        obj = cls.__new__(cls)
        obj._inner = inner
        return obj

    @property
    def byte_offset(self) -> int:
        return self._inner.get_byte_offset()

    @property
    def byte_size(self) -> int:
        return self._inner.get_byte_size()

    def __repr__(self) -> str:
        return (
            f"BufferView(byte_offset={self.byte_offset}, "
            f"byte_size={self.byte_size})"
        )

    __str__ = __repr__


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

    def view(
        self, *, byte_offset: int = 0, byte_size: int | None = None
    ) -> BufferView:
        """Returns a non-owning view over this buffer or a sub-range of it.

        With no arguments, the view spans the whole allocation. The
        returned :class:`BufferView` does not keep this ``Buffer`` alive;
        the caller must ensure the buffer outlives the view and any
        operation that uses it.

        Args:
            byte_offset: Byte offset into the allocation where the view
                begins.
            byte_size: Size of the view in bytes. Defaults to the
                remainder of the allocation after ``byte_offset``.

        Returns:
            A view over the requested range.

        Raises:
            ValueError: If the requested range falls outside the buffer.
        """
        size = self.byte_size if byte_size is None else byte_size
        if byte_offset < 0 or size < 0 or byte_offset + size > self.byte_size:
            raise ValueError("BufferView range exceeds Buffer bounds")
        return BufferView._wrap(self._inner.view(byte_offset, size))

    def __repr__(self) -> str:
        return f"Buffer(byte_size={self.byte_size})"

    __str__ = __repr__
