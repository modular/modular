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

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

import numpy.typing as npt
from max.dtype import DType

from .array import Array
from .buffer import Buffer, BufferView
from .event import Event
from .function import Function

if TYPE_CHECKING:
    from .context import Context


class Stream:
    """An in-order command stream bound to a ``Context``.

    Operations submitted to a Stream complete in submission order. Each op
    implicitly waits for the previous one to finish.

    Not constructed directly; obtain via ``context.create_stream()``.
    """

    _inner: Any
    _context: Context

    __slots__ = ("_context", "_inner")

    def __init__(self) -> None:
        raise TypeError(
            "Stream is not directly constructible; use Context.create_stream()"
        )

    @classmethod
    def _wrap(cls, inner: object, context: Context) -> Stream:
        obj = cls.__new__(cls)
        obj._inner = inner
        obj._context = context
        return obj

    def synchronize(self) -> None:
        self._inner.synchronize()

    @property
    def native_handle(self) -> int | None:
        """Backend stream handle as an integer, or ``None`` if the underlying
        queue has none (a device with no OS-level stream object).

        Suitable to lend a DLPack producer via ``__dlpack__(stream=...)``.
        """
        return self._inner.native_handle()

    def record_event(self) -> Event:
        return Event._wrap(self._inner.record_event())

    def copy(self, *, dst: Buffer, src: Buffer) -> None:
        """Enqueues a copy of `src` into the front of `dst`, after all previous
        stream ops.

        Dispatches by operand residency (device-to-device, or a pinned buffer
        to/from device); a pinned-to-pinned copy has no queue transport and
        raises — use the module-level `copy` for that.
        """
        self._inner.copy(dst._inner, src._inner)

    def copy_to_device(self, dst: BufferView, src_address: int) -> None:
        self._inner.copy_to_device(dst._inner, src_address)

    def copy_from_device(self, dst_address: int, src: BufferView) -> None:
        self._inner.copy_from_device(dst_address, src._inner)

    def copy_intra_device(self, dst: BufferView, src: BufferView) -> None:
        self._inner.copy_intra_device(dst._inner, src._inner)

    def set_memory(self, dst: BufferView, value: int) -> None:
        self._inner.set_memory(dst._inner, value)

    def fill(self, dst: BufferView, value: int, value_size: int) -> None:
        self._inner.fill(dst._inner, value, value_size)

    def wait_for_events(self, *events: Event) -> None:
        self._inner.wait_for_events(tuple(e._inner for e in events))

    def execute(
        self,
        func: Function,
        grid: tuple[int, int, int],
        block: tuple[int, int, int],
        args: Sequence[int],
        arg_sizes: Sequence[int],
        shared_mem_bytes: int = 0,
    ) -> None:
        if len(grid) != 3:
            raise ValueError("grid must be a 3-tuple")
        if len(block) != 3:
            raise ValueError("block must be a 3-tuple")
        if len(args) != len(arg_sizes):
            raise ValueError("args and arg_sizes must have the same length")
        self._inner.execute(
            func._inner,
            grid,
            block,
            args,
            arg_sizes,
            shared_mem_bytes,
        )

    # ------------------------------------------------------------------
    # Async Array operations
    # ------------------------------------------------------------------

    def array_full(
        self,
        dtype: DType,
        shape: Sequence[int] = (),
        value: float | int = 0,
    ) -> Array:
        """Allocates a device ``Array`` and enqueues a fill on this stream.

        Returns before the fill completes; order it before reading (see the
        section note above).
        """
        arr = Array.empty(self._context, dtype, shape)
        # Nothing to enqueue for an empty (zero-element) array.
        if all(arr._shape):
            arr._enqueue_fill(self, value)
        return arr

    def array_fill(self, arr: Array, value: float | int) -> None:
        """Enqueues a fill of ``arr`` with ``value`` on this stream.

        Returns before the fill completes; order it before reading.
        """
        arr._enqueue_fill(self, value)

    def array_copy(self, src: Array, dst: Array) -> None:
        """Enqueues a same-device copy of ``src`` into ``dst`` on this stream.

        Arguments are source-first, matching the blocking :meth:`Array.copy`.
        The source must be contiguous — a strided source needs an internal temp
        this fire-and-forget path cannot keep alive; use the blocking
        :meth:`Array.copy` for that. Returns before the copy completes; order
        it, and keep ``src`` alive, until then.
        """
        if not src.is_contiguous:
            raise ValueError(
                "array_copy requires a contiguous source; use the blocking "
                "Array.copy for a strided source"
            )
        Array._enqueue_copy(src, dst, self)

    def array_from_numpy(self, np_array: npt.NDArray[Any]) -> Array:
        """Allocates a device ``Array`` and enqueues an H2D copy of
        ``np_array`` on this stream.

        Returns before the copy completes; keep ``np_array`` alive and order
        the copy before reading. ``np_array`` must be C-contiguous — a
        non-contiguous source needs an internal temp this fire-and-forget path
        cannot keep alive; use the blocking :meth:`Array.from_numpy` for that.
        """
        if not np_array.flags.c_contiguous:
            raise ValueError(
                "array_from_numpy requires a C-contiguous array; use the "
                "blocking Array.from_numpy for a non-contiguous source"
            )
        dtype = DType.from_numpy(np_array.dtype)
        arr = Array.empty(self._context, dtype, np_array.shape)
        if np_array.nbytes:
            self.copy_to_device(arr._buffer.view(), np_array.ctypes.data)
        return arr

    def array_from_dlpack(self, obj: Any) -> Array:
        """Imports a DLPack producer onto this stream.

        Adopts a same-device producer zero-copy (lending this stream to order
        the producer's work) and host-blocks until it is settled, like
        :meth:`Array.from_dlpack`.
        """
        return Array._from_dlpack(self._context, obj, executor=self)

    def __repr__(self) -> str:
        return "Stream()"

    __str__ = __repr__
