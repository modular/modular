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
"""``Context`` — Python projection of HAL ``Context``."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from .buffer import Buffer
from .bundle import Bundle
from .function import Function
from .queue import Queue
from .stream import Stream


class Context:
    """A context bound to a ``Device``.

    Not constructed directly; obtain via ``device.get_context()``.
    """

    _inner: Any

    __slots__ = ("_inner",)

    def __init__(self) -> None:
        raise TypeError(
            "Context is not directly constructible; use Device.get_context()"
        )

    @classmethod
    def _wrap(cls, inner: object) -> Context:
        obj = cls.__new__(cls)
        obj._inner = inner
        return obj

    def create_queue(self) -> Queue:
        return Queue._wrap(self._inner.create_queue())

    def create_stream(self) -> Stream:
        return Stream._wrap(self._inner.create_stream())

    def alloc_sync(self, byte_size: int) -> Buffer:
        return Buffer._wrap(self._inner.alloc_sync(byte_size))

    def alloc_host_pinned(self, byte_size: int) -> Buffer:
        return Buffer._wrap(self._inner.alloc_host_pinned(byte_size))

    def memory_get_address(self, buf: Buffer) -> int:
        return self._inner.memory_get_address(buf._inner)

    def compile(self, compile_fn: Callable[[Any], Any]) -> Bundle:
        """Compile a Mojo kernel for this context, returning a ``Bundle``.

        Invokes the compile thunk associated with a kernel and returns
        a ``Bundle``. Each thunk is a Python callable exported by a
        kernel module's shared library and corresponds to a single
        kernel whose GPU bytecode was emitted at build time.

        Args:
            compile_fn: A callable taking the inner Mojo ``Context``
                ``PythonObject`` and returning the inner Mojo
                ``Bundle`` ``PythonObject``. In practice, a Mojo
                function instantiated from
                ``compile_to_python_bundle[type_of(func), func]``.

        Returns:
            A ``Bundle`` containing the loaded device module.
        """
        return Bundle._wrap(compile_fn(self._inner))

    def load_function(self, bundle: Bundle, name: str) -> Function:
        """Resolve a symbol inside ``bundle`` into a launchable Function."""
        return Function._wrap(self._inner.load_function(bundle._inner, name))

    def __repr__(self) -> str:
        return "Context()"

    __str__ = __repr__
