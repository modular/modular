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
"""``Queue`` — Python projection of HAL ``Queue``."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from .buffer import Buffer, BufferView
from .event import Event
from .function import Function


class Queue:
    """A command queue bound to a ``Context``.

    Not constructed directly; obtain via ``context.create_queue()``.
    """

    _inner: Any

    __slots__ = ("_inner",)

    def __init__(self) -> None:
        raise TypeError(
            "Queue is not directly constructible; use Context.create_queue()"
        )

    @classmethod
    def _wrap(cls, inner: object) -> Queue:
        obj = cls.__new__(cls)
        obj._inner = inner
        return obj

    def synchronize(self) -> None:
        self._inner.synchronize()

    def record_event(self) -> Event:
        return Event._wrap(self._inner.record_event())

    def copy_to_device(self, dst: BufferView, src_address: int) -> None:
        self._inner.copy_to_device(dst._inner, src_address)

    def copy_from_device(self, dst_address: int, src: BufferView) -> None:
        self._inner.copy_from_device(dst_address, src._inner)

    def copy_intra_device(self, dst: BufferView, src: BufferView) -> None:
        self._inner.copy_intra_device(dst._inner, src._inner)

    def set_memory(self, dst: Buffer, value: int, byte_size: int) -> None:
        self._inner.set_memory(dst._inner, value, byte_size)

    def fill(
        self, dst: Buffer, value: int, value_size: int, byte_size: int
    ) -> None:
        self._inner.fill(dst._inner, value, value_size, byte_size)

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

    def __repr__(self) -> str:
        return "Queue()"

    __str__ = __repr__
