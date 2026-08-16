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

from .buffer import Buffer, BufferView
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

    @property
    def driver_name(self) -> str:
        """Name of the HAL plugin driver backing this context (e.g. "CUDA")."""
        return self._inner.get_driver_name()

    @property
    def device_id(self) -> int:
        """Id of the device this context is bound to."""
        return self._inner.get_device_id()

    def get_dlpack_device(self, pinned: bool) -> tuple[int, int]:
        """Returns the DLPack ``(device_type, device_id)`` for this device.

        The backing plugin reports its own DLPack device type (and its
        host-pinned variant when ``pinned`` is True), mirroring the legacy
        driver's ``getDLDevice``.

        Args:
            pinned: Whether the allocation is host-pinned, selecting the
                plugin's host-accessible DLPack variant.

        Returns:
            The ``(device_type, device_id)`` pair per the DLPack protocol.
        """
        return self._inner.get_dlpack_device(pinned)

    def create_queue(self) -> Queue:
        return Queue._wrap(self._inner.create_queue(), self)

    def create_stream(self) -> Stream:
        return Stream._wrap(self._inner.create_stream(), self)

    def alloc_sync(self, byte_size: int) -> Buffer:
        return Buffer._wrap(self._inner.alloc_sync(byte_size))

    def alloc_host_pinned(self, byte_size: int) -> Buffer:
        return Buffer._wrap(self._inner.alloc_host_pinned(byte_size))

    def wrap_memory(
        self,
        address: int,
        byte_size: int,
        *,
        owning: bool = False,
        pinned: bool = False,
    ) -> Buffer:
        """Wraps an existing device memory region in a buffer handle.

        With ``owning=False`` (the default) the region is externally
        owned: the plugin never frees it, dropping the returned buffer
        releases only the HAL's bookkeeping, and the caller must keep the
        underlying allocation alive for the buffer's lifetime. With
        ``owning=True`` the buffer frees the region through the plugin's
        normal path — only valid for an address that came from this
        plugin's own allocator (e.g. one released with
        :meth:`unwrap_memory`).

        Args:
            address: Base address of the region, in the address space the
                backing plugin uses for device memory.
            byte_size: Size of the region in bytes.
            owning: Whether dropping the buffer frees the region.
            pinned: Whether the region is host-pinned memory. Recorded on the
                buffer so its free path matches a pinned allocation; a
                non-owning wrapped buffer frees nothing either way.

        Returns:
            A buffer handle over the region.
        """
        return Buffer._wrap(
            self._inner.wrap_memory(address, byte_size, owning, pinned)
        )

    def unwrap_memory(self, buffer: Buffer) -> int:
        """Releases ownership of ``buffer``'s region and returns its address.

        The inverse of :meth:`wrap_memory`: after this call the buffer is
        non-owning — dropping it releases only the HAL's bookkeeping — and
        the caller is responsible for the region at the returned address
        (e.g. to hand it to a third-party library, or to re-adopt it with
        ``wrap_memory(..., owning=True)``). Only device buffers can be
        unwrapped: a wrapped handle always frees through the device path,
        so host-pinned buffers are rejected.

        Args:
            buffer: The buffer to release ownership from.

        Returns:
            Base address of the underlying region.
        """
        return self._inner.unwrap_memory(buffer._inner)

    def memory_get_address(self, buf: Buffer) -> int:
        return self._inner.memory_get_address(buf._inner)

    # ------------------------------------------------------------------
    # Synchronous, queue-less copies
    # ------------------------------------------------------------------

    def copy_to_device(self, dst: BufferView, src_address: int) -> None:
        """Copies ``dst.byte_size`` bytes from host memory into ``dst``.

        Runs directly on this context, creating no queue or stream, and returns
        only once the transfer completes. ``src_address`` is a host pointer read
        for exactly ``dst.byte_size`` bytes.
        """
        self._inner.copy_to_device(dst._inner, src_address)

    def copy_from_device(self, dst_address: int, src: BufferView) -> None:
        """Copies ``src.byte_size`` bytes from ``src`` into host memory.

        Runs directly on this context, creating no queue or stream, and returns
        only once the transfer completes. ``dst_address`` is a host pointer
        written with exactly ``src.byte_size`` bytes.
        """
        self._inner.copy_from_device(dst_address, src._inner)

    def copy_intra_device(self, dst: BufferView, src: BufferView) -> None:
        """Copies ``dst.byte_size`` bytes from ``src`` into ``dst``.

        Runs directly on this context, creating no queue or stream, and returns
        only once the transfer completes. Both views must reside on this
        context's device.
        """
        self._inner.copy_intra_device(dst._inner, src._inner)

    def set_memory(self, dst: BufferView, value: int) -> None:
        """Sets every byte of ``dst`` to ``value``, blocking until complete.

        Runs directly on this context, creating no queue or stream, and returns
        only once the fill completes.
        """
        self._inner.set_memory(dst._inner, value)

    def fill(self, dst: BufferView, value: int, value_size: int) -> None:
        """Fills ``dst`` with a repeated ``value_size``-byte ``value``.

        Runs directly on this context, creating no queue or stream, and returns
        only once the fill completes. ``value_size`` must be one of 1, 2, 4,
        or 8.
        """
        self._inner.fill(dst._inner, value, value_size)

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
