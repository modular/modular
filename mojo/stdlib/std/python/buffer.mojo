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

# ===----------------------------------------------------------------------=== #
# BufferProtocolBuilder — bf_getbuffer / bf_releasebuffer slots
#
# Enables Mojo extension module types to expose their internal memory via
# Python's buffer protocol, allowing zero-copy access from numpy, memoryview,
# bytes(), and other consumers.
#
# Target: 1D C-contiguous buffers (most common use case).
# ===----------------------------------------------------------------------=== #

from std.ffi import c_int
from std.memory import OpaquePointer, UnsafePointer, alloc
from std.python import Python, PythonObject
from std.python._cpython import (
    PyObjectPtr,
    Py_ssize_t,
    PySlotIndex,
    PyType_Slot,
)
from std.python.bindings import PythonTypeBuilder

from .adapters import _unwrap_self


# PyBUF_ flag constants (from CPython Include/cpython/object.h).
comptime _PyBUF_WRITABLE = Int32(0x0001)
comptime _PyBUF_FORMAT = Int32(0x0004)


struct BufferInfo(Movable):
    """User-friendly buffer descriptor returned by a `bf_getbuffer` handler.

    Fill this in your handler to describe a 1D C-contiguous buffer.
    The `buf` pointer must remain valid until the matching `bf_releasebuffer`
    is called.  Do **not** resize the backing allocation while a buffer view
    is active.

    The wrapper heap-promotes the returned `BufferInfo` so its address is
    stable across the `bf_getbuffer` / `bf_releasebuffer` window; the
    `nitems` and `itemsize` fields are used directly as the single-element
    `shape` and `strides` arrays (`Py_ssize_t *` of length 1).

    Example:
        ```mojo
        @staticmethod
        def get_buffer(
            self_ptr: UnsafePointer[Self, MutAnyOrigin], flags: Int32
        ) raises -> BufferInfo:
            var data_ptr = self_ptr[].data.unsafe_ptr()
            return BufferInfo(
                buf=rebind[UnsafePointer[UInt8, MutAnyOrigin]](data_ptr),
                nitems=len(self_ptr[].data),
                itemsize=8,
                format="d",
                readonly=True,
            )
        ```
    """

    var buf: UnsafePointer[UInt8, MutAnyOrigin]
    """Pointer to the first byte of the buffer data."""
    var nitems: Int
    """Number of elements in the buffer."""
    var itemsize: Int
    """Size of one element in bytes (e.g. 8 for `Float64`)."""
    var format: String
    """Python struct-module format character (e.g. `"d"` for `Float64`)."""
    var readonly: Bool
    """Whether the buffer is read-only."""

    def __init__(
        out self,
        buf: UnsafePointer[UInt8, MutAnyOrigin],
        nitems: Int,
        itemsize: Int,
        format: String,
        readonly: Bool = True,
    ):
        self.buf = buf
        self.nitems = nitems
        self.itemsize = itemsize
        self.format = format
        self.readonly = readonly


# ===----------------------------------------------------------------------=== #
# _PyBuffer — Mojo mirror of CPython's Py_buffer struct
#
# Layout (80 bytes on 64-bit platforms) must match Include/cpython/object.h:
#   offset  0: void *buf              (8 bytes)
#   offset  8: PyObject *obj          (8 bytes)
#   offset 16: Py_ssize_t len         (8 bytes)
#   offset 24: Py_ssize_t itemsize    (8 bytes)
#   offset 32: int readonly           (4 bytes)
#   offset 36: int ndim               (4 bytes)
#   offset 40: char *format           (8 bytes)
#   offset 48: Py_ssize_t *shape      (8 bytes)
#   offset 56: Py_ssize_t *strides    (8 bytes)
#   offset 64: Py_ssize_t *suboffsets (8 bytes)
#   offset 72: void *internal         (8 bytes)
# ===----------------------------------------------------------------------=== #
struct _PyBuffer:
    var buf: OpaquePointer[MutAnyOrigin]
    var obj: PyObjectPtr
    var len: Int
    var itemsize: Int
    var readonly: Int32
    var ndim: Int32
    var format: UnsafePointer[UInt8, MutAnyOrigin]
    var shape: UnsafePointer[Int, MutAnyOrigin]
    var strides: UnsafePointer[Int, MutAnyOrigin]
    var suboffsets: UnsafePointer[Int, MutAnyOrigin]
    var internal: Optional[OpaquePointer[MutAnyOrigin]]


# ===----------------------------------------------------------------------=== #
# Adapter functions
# ===----------------------------------------------------------------------=== #


def _bf_getbuffer_wrapper[
    self_type: ImplicitlyDestructible,
    method: def(
        UnsafePointer[self_type, MutAnyOrigin], Int32
    ) thin raises -> BufferInfo,
](
    raw_self: PyObjectPtr,
    view: UnsafePointer[_PyBuffer, MutAnyOrigin],
    flags: c_int,
) abi("C") -> c_int:
    """CPython `getbufferproc` adapter for the `bf_getbuffer` slot.

    Calls the user's handler to get a `BufferInfo`, then fills in the
    `Py_buffer` view.  Allocates a small heap block for shape, strides, and
    the format string; the pointer is stashed in `view->internal` and freed
    by `_bf_releasebuffer_impl`.

    Parameters:
        self_type: The Mojo struct type whose instances back the Python object.
        method: User function
            `def(self_ptr: UnsafePointer[T, MutAnyOrigin], flags: Int32) raises -> BufferInfo`.

    Returns:
        0 on success, -1 with an exception set on error.
    """
    ref cpython = Python().cpython()
    try:
        var self_ptr = _unwrap_self[self_type](raw_self)
        var info = method(self_ptr, Int32(flags))

        # Reject writable requests for read-only buffers.
        if Int32(flags) & _PyBUF_WRITABLE and info.readonly:
            var error_type = cpython.get_error_global("PyExc_BufferError")
            cpython.PyErr_SetString(
                error_type,
                "buffer is not writable".as_c_string_slice().unsafe_ptr(),
            )
            return c_int(-1)

        # Heap-promote `info` so its address is stable across the
        # `bf_getbuffer` / `bf_releasebuffer` window. CPython requires that
        # `view->shape`, `view->strides`, and `view->format` remain valid
        # until release; pointing them at fields inside the heap-allocated
        # `BufferInfo` keeps everything alive with a single allocation
        # (no separate metadata block, no format-string copy).
        var heap_info = alloc[BufferInfo](1)
        heap_info.init_pointee_move(info^)

        # Fill the Py_buffer view.
        view[].buf = rebind[OpaquePointer[MutAnyOrigin]](heap_info[].buf)
        view[].obj = cpython.Py_NewRef(raw_self)
        view[].len = heap_info[].nitems * heap_info[].itemsize
        view[].itemsize = heap_info[].itemsize
        view[].readonly = Int32(1) if heap_info[].readonly else Int32(0)
        view[].ndim = Int32(1)
        view[].suboffsets = UnsafePointer[Int, MutAnyOrigin](
            unsafe_from_address=0
        )

        # shape[0] = nitems, strides[0] = itemsize — addresses of the
        # corresponding Int fields inside the heap BufferInfo serve as the
        # single-element Py_ssize_t arrays CPython expects.
        view[].shape = UnsafePointer(to=heap_info[].nitems)
        view[].strides = UnsafePointer(to=heap_info[].itemsize)

        # Provide format string only when the consumer requests it. Mojo
        # strings already store a trailing null byte, so the pointer is
        # safe to hand to CPython directly.
        if Int32(flags) & _PyBUF_FORMAT:
            view[].format = rebind[UnsafePointer[UInt8, MutAnyOrigin]](
                heap_info[].format.unsafe_ptr()
            )
        else:
            view[].format = UnsafePointer[UInt8, MutAnyOrigin](
                unsafe_from_address=0
            )

        # Stash the heap_info pointer for releasebuffer to destroy and free.
        view[].internal = rebind[OpaquePointer[MutAnyOrigin]](heap_info)

        return c_int(0)
    except e:
        var error_type = cpython.get_error_global("PyExc_BufferError")
        var msg = String(e)
        cpython.PyErr_SetString(
            error_type, msg.as_c_string_slice().unsafe_ptr()
        )
        return c_int(-1)


def _bf_releasebuffer_impl(
    raw_self: PyObjectPtr, view: UnsafePointer[_PyBuffer, MutAnyOrigin]
) abi("C") -> None:
    """Default `releasebufferproc` that destroys and frees the heap-
    promoted `BufferInfo` referenced by `view->internal`.

    Called by CPython after a consumer is done with a buffer view. The
    `BufferInfo` heap slot allocated by `_bf_getbuffer_wrapper` owns the
    format `String` and serves as backing storage for the `shape` and
    `strides` pointers, so we must run its destructor before freeing the
    slot itself.
    """
    if view[].internal:
        var heap_info = rebind[UnsafePointer[BufferInfo, MutAnyOrigin]](
            view[].internal
        )
        heap_info.destroy_pointee()
        heap_info.free()
        view[].internal = OpaquePointer[MutAnyOrigin](unsafe_from_address=0)


# ===----------------------------------------------------------------------=== #
# Slot-install helpers
# ===----------------------------------------------------------------------=== #


struct _BfSlotInstaller:
    """Static-method namespace for inserting CPython buffer-protocol slot
    function pointers (`bf_getbuffer`, `bf_releasebuffer`) into a
    `PythonTypeBuilder`. Kept local to `buffer.mojo` so the generic
    `_SlotInstaller` in `adapters.mojo` doesn't need to reach into this
    module's private types.
    """

    @staticmethod
    def getbuffer[
        self_type: ImplicitlyDestructible,
        method: def(
            UnsafePointer[self_type, MutAnyOrigin], Int32
        ) thin raises -> BufferInfo,
    ](ptr: UnsafePointer[mut=True, PythonTypeBuilder, MutAnyOrigin]):
        """Insert the `bf_getbuffer` slot into the builder pointed to by `ptr`.
        """
        comptime _getbufferproc = def(
            PyObjectPtr, UnsafePointer[_PyBuffer, MutAnyOrigin], c_int
        ) thin abi("C") -> c_int
        var fn_ptr: _getbufferproc = _bf_getbuffer_wrapper[self_type, method]
        ptr[]._insert_slot(
            PyType_Slot(
                PySlotIndex.bf_getbuffer,
                rebind[OpaquePointer[MutAnyOrigin]](fn_ptr),
            )
        )

    @staticmethod
    def releasebuffer(
        ptr: UnsafePointer[mut=True, PythonTypeBuilder, MutAnyOrigin]
    ):
        """Insert the default `bf_releasebuffer` slot into the builder pointed to by `ptr`.
        """
        comptime _releasebufferproc = def(
            PyObjectPtr, UnsafePointer[_PyBuffer, MutAnyOrigin]
        ) thin abi("C") -> None
        var fn_ptr: _releasebufferproc = _bf_releasebuffer_impl
        ptr[]._insert_slot(
            PyType_Slot(
                PySlotIndex.bf_releasebuffer,
                rebind[OpaquePointer[MutAnyOrigin]](fn_ptr),
            )
        )


# ===----------------------------------------------------------------------=== #
# BufferProtocolBuilder
# ===----------------------------------------------------------------------=== #


struct BufferProtocolBuilder[self_type: ImplicitlyDestructible]:
    """Wraps a `PythonTypeBuilder` reference and installs CPython buffer protocol slots.

    `BufferProtocolBuilder` holds a pointer to a `PythonTypeBuilder` that is
    owned by the enclosing `PythonModuleBuilder`.  The caller must ensure the
    module builder (and its type_builders list) outlives this object, which is
    naturally satisfied when both are used within the same `PyInit_*` function.

    Only 1D C-contiguous buffers are supported.  The handler must return a
    `BufferInfo` describing the data; it is called with the `flags` bitmask
    so the handler can raise `BufferError` for unsupported combinations (e.g.
    `PyBUF_WRITABLE` against a read-only buffer).

    Usage:
        ```mojo
        ref tb = b.add_type[FloatBuf]("FloatBuf")
            .def_init_defaultable[FloatBuf]()
            .def_staticmethod[FloatBuf.new]("new")
        BufferProtocolBuilder[FloatBuf](tb)
            .def_getbuffer[FloatBuf.get_buffer]()
            .def_releasebuffer()
        ```
    """

    var _ptr: UnsafePointer[mut=True, PythonTypeBuilder, MutAnyOrigin]

    def __init__(out self, mut inner: PythonTypeBuilder):
        self._ptr = UnsafePointer(to=inner)

    def __init__(
        out self,
        ptr: UnsafePointer[mut=True, PythonTypeBuilder, MutAnyOrigin],
    ):
        self._ptr = ptr

    def def_getbuffer[
        method: def(
            UnsafePointer[Self.self_type, MutAnyOrigin], Int32
        ) thin raises -> BufferInfo
    ](mut self) -> ref[self] Self:
        """Install `__buffer__` via the `bf_getbuffer` slot.

        Called by `memoryview(obj)`, `numpy.frombuffer(obj)`, etc.

        The handler receives the consumer's `flags` bitmask.  Raise a
        standard `Error` from the handler to propagate a Python `BufferError`.
        Raise with message `"buffer is not writable"` — or check
        `flags & 0x0001` yourself — to reject writable requests.

        Parameters:
            method: Static method with signature
                `def(self_ptr: UnsafePointer[T, MutAnyOrigin], flags: Int32) raises -> BufferInfo`.

        See: https://docs.python.org/3/c-api/typeobj.html#c.PyBufferProcs.bf_getbuffer
        """
        _BfSlotInstaller.getbuffer[Self.self_type, method](self._ptr)
        return self

    def def_releasebuffer(mut self) -> ref[self] Self:
        """Install the default `bf_releasebuffer` slot.

        The default implementation frees the shape/strides/format block that
        `def_getbuffer` allocates.  Call this after `def_getbuffer` whenever
        you install a getbuffer handler.

        See: https://docs.python.org/3/c-api/typeobj.html#c.PyBufferProcs.bf_releasebuffer
        """
        _BfSlotInstaller.releasebuffer(self._ptr)
        return self
