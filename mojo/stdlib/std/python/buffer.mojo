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
from std.memory import OpaquePointer, UnsafePointer
from std.python import Python, PythonObject
from std.python._cpython import PyObjectPtr, Py_ssize_t, PyType_Slot
from std.python.bindings import PythonTypeBuilder

from .adapters import _unwrap_self


# Slot indices for the buffer protocol (from CPython Include/typeslots.h).
comptime _BF_GETBUFFER = Int32(1)
comptime _BF_RELEASEBUFFER = Int32(2)

# PyBUF_ flag constants (from CPython Include/cpython/object.h).
comptime _PyBUF_WRITABLE = Int32(0x0001)
comptime _PyBUF_FORMAT = Int32(0x0004)
comptime _PyBUF_ND = Int32(0x0008)
comptime _PyBUF_STRIDES = Int32(0x0018)  # 0x0010 | PyBUF_ND


struct BufferInfo:
    """User-friendly buffer descriptor returned by a `bf_getbuffer` handler.

    Fill this in your handler to describe a 1D C-contiguous buffer.
    The `buf` pointer must remain valid until the matching `bf_releasebuffer`
    is called.  Do **not** resize the backing allocation while a buffer view
    is active.

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
    var internal: OpaquePointer[MutAnyOrigin]


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

        # Allocate storage for: shape[0] (8 bytes) + strides[0] (8 bytes)
        # + format string (fmt_len bytes) + null terminator (1 byte).
        # On 64-bit platforms Int = 8 bytes, so the two Int fields occupy 16 bytes.
        var fmt_bytes = info.format.as_bytes()
        var fmt_len = len(fmt_bytes)
        var alloc_size = 16 + fmt_len + 1  # 2 * sizeof(Int64) + format + NUL

        # List.steal_data() gives us an owned UnsafePointer we can free later.
        var store = List[UInt8](capacity=alloc_size)
        store.resize(alloc_size, 0)
        var alloc = store.steal_data()

        # shape[0] = nitems  (Py_ssize_t at byte offset 0)
        var shape_ptr = rebind[UnsafePointer[Int, MutAnyOrigin]](alloc)
        shape_ptr[0] = info.nitems

        # strides[0] = itemsize  (Py_ssize_t at byte offset 8)
        var stride_ptr = shape_ptr + 1
        stride_ptr[0] = info.itemsize

        # format string: copy bytes then null-terminate (byte offset 16)
        var fmt_ptr = alloc + 16  # 2 * 8 bytes past the two Int fields
        for i in range(fmt_len):
            fmt_ptr[i] = fmt_bytes[i]
        fmt_ptr[fmt_len] = 0

        # Fill the Py_buffer view.
        view[].buf = rebind[OpaquePointer[MutAnyOrigin]](info.buf)
        view[].obj = cpython.Py_NewRef(raw_self)
        view[].len = info.nitems * info.itemsize
        view[].itemsize = info.itemsize
        view[].readonly = Int32(1) if info.readonly else Int32(0)
        view[].ndim = Int32(1)
        view[].suboffsets = UnsafePointer[Int, MutAnyOrigin](
            unsafe_from_address=0
        )

        # Always provide shape; strides are provided so consumers requesting
        # PyBUF_STRIDES / PyBUF_FULL_RO (e.g. memoryview) work correctly.
        view[].shape = shape_ptr
        view[].strides = stride_ptr

        # Provide format string only when the consumer requests it.
        if Int32(flags) & _PyBUF_FORMAT:
            view[].format = rebind[UnsafePointer[UInt8, MutAnyOrigin]](fmt_ptr)
        else:
            view[].format = UnsafePointer[UInt8, MutAnyOrigin](
                unsafe_from_address=0
            )

        # Stash the allocation for releasebuffer to free.
        view[].internal = rebind[OpaquePointer[MutAnyOrigin]](alloc)

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
    """Default `releasebufferproc` that frees the shape/strides/format block.

    Called by CPython after a consumer is done with a buffer view.  The
    heap block allocated by `_bf_getbuffer_wrapper` is stored in
    `view->internal`; this function frees it and clears the field.
    """
    if view[].internal:
        rebind[UnsafePointer[UInt8, MutAnyOrigin]](view[].internal).free()
        view[].internal = OpaquePointer[MutAnyOrigin](unsafe_from_address=0)


# ===----------------------------------------------------------------------=== #
# Slot-install helpers
# ===----------------------------------------------------------------------=== #


def _install_bf_getbuffer[
    self_type: ImplicitlyDestructible,
    method: def(
        UnsafePointer[self_type, MutAnyOrigin], Int32
    ) thin raises -> BufferInfo,
](ptr: UnsafePointer[mut=True, PythonTypeBuilder, MutAnyOrigin]):
    """Insert the `bf_getbuffer` slot into the builder pointed to by `ptr`."""
    comptime _getbufferproc = def(
        PyObjectPtr, UnsafePointer[_PyBuffer, MutAnyOrigin], c_int
    ) thin abi("C") -> c_int
    var fn_ptr: _getbufferproc = _bf_getbuffer_wrapper[self_type, method]
    ptr[]._insert_slot(
        PyType_Slot(_BF_GETBUFFER, rebind[OpaquePointer[MutAnyOrigin]](fn_ptr))
    )


def _install_bf_releasebuffer(
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
            _BF_RELEASEBUFFER, rebind[OpaquePointer[MutAnyOrigin]](fn_ptr)
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
        _install_bf_getbuffer[Self.self_type, method](self._ptr)
        return self

    def def_releasebuffer(mut self) -> ref[self] Self:
        """Install the default `bf_releasebuffer` slot.

        The default implementation frees the shape/strides/format block that
        `def_getbuffer` allocates.  Call this after `def_getbuffer` whenever
        you install a getbuffer handler.

        See: https://docs.python.org/3/c-api/typeobj.html#c.PyBufferProcs.bf_releasebuffer
        """
        _install_bf_releasebuffer(self._ptr)
        return self
