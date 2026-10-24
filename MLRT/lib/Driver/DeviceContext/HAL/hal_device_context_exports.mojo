# ===----------------------------------------------------------------------=== #
# Copyright (c) 2026, Modular Inc. All rights reserved.
#
# Licensed under the MAX License:
# https://modular.com/legal/max-license
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ===----------------------------------------------------------------------=== #
"""C-ABI exports for the HAL-backed `DeviceContext`.

This is the Mojo half of the HAL C++ compatibility layer: each `MojoHAL_*`
function is a C-ABI entry point the C++ `DeviceContext` shim forwards to. The
handle exchanged with C++ is a heap-boxed Mojo `DeviceContext` value; the box
carries no extra reference count of its own - the `DeviceContext`'s internal
`ArcPointer`s manage sharing, and `MojoHAL_DeviceContext_release` destroys the
box exactly once.

Error convention (mirrors `MojoBindings.cpp`, inverted): functions return a
heap-allocated null-terminated `const char *` error message, or null on
success — typed as the same `_CString` the legacy bindings use. The C++ side
frees it with `MojoHAL_strfree`.
"""

from std.atomic import Atomic, Ordering
from std.ffi import c_char, CStringSpan, external_call
from std.sys import size_of
from max._driver import get_device_spec
from max._driver.max_license.hal.context import Buffer

comptime _HALRawBuffer = Buffer[get_device_spec[0]()]
from max.gpu.host import (
    DeviceBuffer,
    DeviceContext,
    DeviceEvent,
    DeviceStream,
    HostBuffer,
)
from max.gpu.host import _CString
from max.gpu.host import DeviceAttribute
from max._driver.max_license.device_context_hal.device_context_hal import (
    _DeviceContextScopeHAL,
)
from std.memory import (
    alloc,
    dealloc,
    Layout,
    ThinAllocation,
    unsafe_memcpy,
    Pointer,
)


# ===---------------------------------------------------------------------=== #
# C-side handle types
# ===---------------------------------------------------------------------=== #
# Every handle the C++ shim exchanges with this file is a pointer to a
# heap-boxed Mojo value, so the handle types name their actual pointee (the
# analogue of the legacy `_DeviceContextCpp` marker structs, with the real
# type available). `MutUntrackedOrigin` throughout: box lifetimes are managed
# by the create/release protocol across the ABI (Mojo allocates, the C++
# shim's refcounted wrapper releases), not by Mojo's origin tracking.

comptime _DeviceContextBoxPtr = Pointer[DeviceContext, MutUntrackedOrigin]
# Buffer handles are layout-punned: `_MmapBox` and `_PinnedBox` keep a
# `DeviceBuffer[DType.uint8]` as their leading field, so every buffer handle
# reads as a pointer to that view; the release entries bitcast back to the
# concrete box type.
comptime _DeviceBufferBoxPtr = Pointer[DeviceBuffer[.uint8], MutUntrackedOrigin]
comptime _DeviceStreamBoxPtr = Pointer[DeviceStream, MutUntrackedOrigin]
comptime _DeviceEventBoxPtr = Pointer[DeviceEvent, MutUntrackedOrigin]
comptime _DeviceContextScopeBoxPtr = Pointer[
    _DeviceContextScopeHAL, MutUntrackedOrigin
]
comptime _CompletionFlagBoxPtr = Pointer[_CompletionFlagBox, MutUntrackedOrigin]

# Raw device or host memory crossing the ABI as untyped bytes.
comptime _RawMemoryPtr = Pointer[Byte, MutUntrackedOrigin]

# An out-parameter slot the export writes for the C++ caller.
comptime _OutPtr[T: AnyType] = Pointer[mut=True, T, MutUntrackedOrigin]


def _err_cstr(msg: String) -> _CString[]:
    """Copies `msg` into a heap-allocated null-terminated C string.

    Freed by the C++ caller via `MojoHAL_strfree`.
    """
    var n = msg.byte_length()
    var p = alloc[UInt8]({length = n + 1}).unsafe_leak()
    unsafe_memcpy(dest=p, src=msg.unsafe_ptr(), count=n)
    p[unsafe_offset=n] = 0
    return CStringSpan(unsafe_from_ptr=p.unsafe_bitcast[c_char]().as_imm())


@export
def MojoHAL_strfree(s: _CString[]) abi("C"):
    if case .Some(s) = s:
        # `_err_cstr` allocated this via `alloc[UInt8]`. `CStringSpan` hands
        # back a safe immutable pointer, so rebuild the matching mutable,
        # untracked `UInt8` allocation pointer from its address and free that.
        var addr = Int(s.ptr())
        Pointer[UInt8, MutUntrackedOrigin](
            unsafe_from_address=addr
        ).unsafe_free()


@export
def MojoHAL_Device_numberOfDevices(
    api: Pointer[mut=False, c_char, ImmUntrackedOrigin],
) abi("C") -> Int32:
    return Int32(
        DeviceContext.number_of_devices(api=String(unsafe_from_utf8_ptr=api))
    )


@export
def MojoHAL_DeviceContext_enableAllPeerAccess() abi("C") -> _CString[]:
    try:
        DeviceContext.enable_all_peer_access()
        return None
    except e:
        return _err_cstr(String(e))


@export
def MojoHAL_DeviceContext_create(
    out_ctx: _OutPtr[_DeviceContextBoxPtr],
    api: Pointer[mut=False, c_char, ImmUntrackedOrigin],
    device_id: Int32,
) abi("C") -> _CString[]:
    try:
        var ctx = DeviceContext(
            Int(device_id), api=String(unsafe_from_utf8_ptr=api)
        )
        var boxed = alloc(Layout[DeviceContext].single())
        boxed.unsafe_ptr().unsafe_write(ctx^)
        out_ctx[] = boxed^.unsafe_leak()
        return None
    except e:
        return _err_cstr(String(e))


@export
def MojoHAL_DeviceContext_release(ctx: _DeviceContextBoxPtr) abi("C"):
    var boxed = ctx
    boxed.unsafe_deinit_pointee()
    dealloc(
        ThinAllocation(unsafe_owned_ptr=boxed).unsafe_with_layout(
            Layout[DeviceContext].single()
        )
    )


@export
def MojoHAL_DeviceContext_synchronize(
    ctx: _DeviceContextBoxPtr,
) abi("C") -> _CString[]:
    try:
        ctx[].synchronize()
        return None
    except e:
        return _err_cstr(String(e))


def _alloc_cstr(msg: String) -> Pointer[c_char, MutUntrackedOrigin]:
    var n = msg.byte_length()
    var p = alloc[UInt8]({length = n + 1}).unsafe_leak()
    unsafe_memcpy(dest=p, src=msg.unsafe_ptr(), count=n)
    p[unsafe_offset=n] = 0
    return p.unsafe_bitcast[c_char]()


# ===---------------------------------------------------------------------=== #
# Buffers
# ===---------------------------------------------------------------------=== #


@export
def MojoHAL_DeviceContext_createBuffer(
    out_buf: _OutPtr[_DeviceBufferBoxPtr],
    out_ptr: _OutPtr[_RawMemoryPtr],
    ctx: _DeviceContextBoxPtr,
    byte_size: UInt64,
) abi("C") -> _CString[]:
    try:
        var buf = DeviceBuffer[.uint8](ctx[], Int(byte_size))
        out_ptr[] = buf.unsafe_ptr()
        var boxed = alloc(Layout[DeviceBuffer[.uint8]].single())
        boxed.unsafe_ptr().unsafe_write(buf^)
        out_buf[] = boxed^.unsafe_leak()
        return None
    except e:
        return _err_cstr(String(e))


@export
def MojoHAL_DeviceContext_createBufferOwning(
    out_buf: _OutPtr[_DeviceBufferBoxPtr],
    ctx: _DeviceContextBoxPtr,
    ptr: _RawMemoryPtr,
    byte_size: UInt64,
    owning: Bool,
) abi("C") -> _CString[]:
    var buf = DeviceBuffer[.uint8](ctx[], ptr, Int(byte_size), owning=owning)
    var boxed = alloc(Layout[DeviceBuffer[.uint8]].single())
    boxed.unsafe_ptr().unsafe_write(buf^)
    out_buf[] = boxed^.unsafe_leak()
    return None


@export
def MojoHAL_DeviceBuffer_release(buf: _DeviceBufferBoxPtr) abi("C"):
    var boxed = buf
    boxed.unsafe_deinit_pointee()
    dealloc(
        ThinAllocation(unsafe_owned_ptr=boxed).unsafe_with_layout(
            Layout[DeviceBuffer[.uint8]].single()
        )
    )


@export
def MojoHAL_DeviceBuffer_relinquishOwnership(
    buf: _DeviceBufferBoxPtr,
) abi("C") -> _CString[]:
    # Flips the plugin-side `owning` flag to false (the borrowing half of
    # `DeviceBuffer.take_ptr`): releasing the box later frees only the
    # wrapper, not the underlying allocation. Backs the C++
    # `DeviceBuffer::releasePtr` and `setDestructorFn` paths, where the caller
    # (or its custom destructor) takes over the allocation.
    try:
        var boxed = buf
        _ = boxed[]._ctx._context[].unwrap_memory(boxed[]._inner[]._buffer)
        return None
    except e:
        return _err_cstr(String(e))


@export
def MojoHAL_DeviceContext_createHostBuffer(
    out_buf: _OutPtr[_DeviceBufferBoxPtr],
    out_ptr: _OutPtr[_RawMemoryPtr],
    ctx: _DeviceContextBoxPtr,
    byte_size: UInt64,
) abi("C") -> _CString[]:
    try:
        var buf = HostBuffer[.uint8](ctx[], Int(byte_size))
        out_ptr[] = buf.unsafe_ptr()
        var boxed = alloc(Layout[HostBuffer[.uint8]].single())
        boxed.unsafe_ptr().unsafe_write(buf^)
        # Host-buffer boxes travel through the same buffer handle; the C++
        # side keys host/device by the entry that produced the handle.
        out_buf[] = boxed^.unsafe_leak().unsafe_bitcast[DeviceBuffer[.uint8]]()
        return None
    except e:
        return _err_cstr(String(e))


@export
def MojoHAL_HostBuffer_release(buf: _DeviceBufferBoxPtr) abi("C"):
    var boxed = buf.unsafe_bitcast[HostBuffer[.uint8]]()
    boxed.unsafe_deinit_pointee()
    dealloc(
        ThinAllocation(unsafe_owned_ptr=boxed).unsafe_with_layout(
            Layout[HostBuffer[.uint8]].single()
        )
    )


# ===---------------------------------------------------------------------=== #
# Copies / memset
# ===---------------------------------------------------------------------=== #


@export
def MojoHAL_DeviceContext_enqueueCopy(
    ctx: _DeviceContextBoxPtr,
    dst: _RawMemoryPtr,
    src: _RawMemoryPtr,
    byte_size: UInt64,
) abi("C") -> _CString[]:
    # Direction-agnostic, mirroring the legacy two-pointer copy (DtoD under
    # unified virtual addressing). Only used for genuine device-to-device
    # copies; host<->device dispatch happens in the C++ shim via the
    # direction-specific entries below.
    # Zero-sized copies are no-ops (legacy semantics); the pointers may be
    # null, which the plugin rejects.
    if byte_size == 0:
        return None
    try:
        ctx[].enqueue_copy(dst, src, Int(byte_size))
        return None
    except e:
        return _err_cstr(String(e))


@export
def MojoHAL_DeviceContext_enqueueCopyToDevice(
    ctx: _DeviceContextBoxPtr,
    dst: _RawMemoryPtr,
    src: _RawMemoryPtr,
    byte_size: UInt64,
) abi("C") -> _CString[]:
    # Host-to-device copy (cuMemcpyHtoD): `dst` is a device address, `src` a
    # raw host pointer. Wrap the device address as a non-owning buffer so the
    # plugin's typed copy entry handles pageable host memory correctly.
    # Zero-sized copies are no-ops (legacy semantics); the pointers may be
    # null, which the plugin rejects.
    if byte_size == 0:
        return None
    try:
        var ctxref = ctx[]
        var dst_buf = DeviceBuffer[.uint8](
            ctxref, dst, Int(byte_size), owning=False
        )
        ctxref.enqueue_copy(dst_buf, src)
        return None
    except e:
        return _err_cstr(String(e))


@export
def MojoHAL_DeviceContext_enqueueCopyFromDevice(
    ctx: _DeviceContextBoxPtr,
    dst: _RawMemoryPtr,
    src: _RawMemoryPtr,
    byte_size: UInt64,
) abi("C") -> _CString[]:
    # Device-to-host copy: `dst` is a raw host pointer, `src` a  device
    # address. Zero-sized copies are no-ops (legacy semantics); the
    # pointers may be null, which the plugin rejects.
    if byte_size == 0:
        return None
    try:
        var ctxref = ctx[]
        var src_buf = DeviceBuffer[.uint8](
            ctxref, src, Int(byte_size), owning=False
        )
        ctxref.enqueue_copy(dst, src_buf)
        return None
    except e:
        return _err_cstr(String(e))


@export
def MojoHAL_DeviceContext_enqueueSetMemory(
    ctx: _DeviceContextBoxPtr,
    dst: _RawMemoryPtr,
    byte_size: UInt64,
    value: UInt64,
    value_size: UInt64,
) abi("C") -> _CString[]:
    # Zero-sized memsets are no-ops (legacy semantics); `dst` may be null,
    # which the plugin rejects.
    if byte_size == 0:
        return None
    try:
        ref device_ctx = ctx[]
        var view = DeviceBuffer[.uint8](
            device_ctx, dst, Int(byte_size), owning=False
        )
        device_ctx._stream[].fill(
            view._inner[]._buffer.view(), value, value_size
        )
        # `fill` receives a weak `BufferView` (a detached POD copy, not a
        # borrow of `view`), so ASAP-drop would otherwise release the plugin
        # memory wrapper before the queue op reads it.
        _ = view^
        return None
    except e:
        return _err_cstr(String(e))


# ===---------------------------------------------------------------------=== #
# Device info
# ===---------------------------------------------------------------------=== #


@export
def MojoHAL_DeviceContext_getArchName(
    out_name: _OutPtr[Pointer[c_char, MutUntrackedOrigin]],
    ctx: _DeviceContextBoxPtr,
) abi("C") -> _CString[]:
    try:
        out_name[] = _alloc_cstr(ctx[].arch_name())
        return None
    except e:
        return _err_cstr(String(e))


@export
def MojoHAL_DeviceContext_isHostUnified(
    out_unified: _OutPtr[Bool],
    ctx: _DeviceContextBoxPtr,
) abi("C") -> _CString[]:
    try:
        out_unified[] = ctx[].is_host_unified()
        return None
    except e:
        return _err_cstr(String(e))


@export
def MojoHAL_DeviceContext_maxAllocationSize(
    out_has_limit: _OutPtr[Bool],
    out_size: _OutPtr[UInt64],
    ctx: _DeviceContextBoxPtr,
) abi("C") -> _CString[]:
    try:
        ref device = ctx[]._device[]
        var size = device._raw[].get_optional_device_property[
            "max_allocation_size", UInt64
        ](device._handle)
        out_has_limit[] = Bool(size)
        out_size[] = size.or_else(0)
        return None
    except e:
        return _err_cstr(String(e))


@export
def MojoHAL_DeviceContext_isCompatible(
    ctx: _DeviceContextBoxPtr,
) abi("C") -> Bool:
    return ctx[].is_compatible()


@export
def MojoHAL_DeviceContext_getName(
    out_name: _OutPtr[Pointer[c_char, MutUntrackedOrigin]],
    ctx: _DeviceContextBoxPtr,
) abi("C") -> _CString[]:
    try:
        out_name[] = _alloc_cstr(ctx[].name())
        return None
    except e:
        return _err_cstr(String(e))


@export
def MojoHAL_DeviceContext_getApi(
    out_api: _OutPtr[Pointer[c_char, MutUntrackedOrigin]],
    ctx: _DeviceContextBoxPtr,
) abi("C") -> _CString[]:
    # The concrete device API ("cuda"/"hip"/"metal"/"cpu"), as opposed to the
    # generic "gpu" a caller may have requested at creation time. `api()` does
    # not raise.
    out_api[] = _alloc_cstr(ctx[].api())
    return None


@export
def MojoHAL_DeviceContext_id(ctx: _DeviceContextBoxPtr) abi("C") -> Int64:
    # Field read, bypassing `id()` (which raises only for signature parity).
    return ctx[]._device[].id


@export
def MojoHAL_DeviceContext_canAccess(
    ctx: _DeviceContextBoxPtr,
    peer: _DeviceContextBoxPtr,
) abi("C") -> Bool:
    # The C++ signature has no error channel; a failed query is "no".
    try:
        return ctx[].can_access(peer[])
    except:
        return False


@export
def MojoHAL_DeviceContext_getMemoryInfo(
    out_free: _OutPtr[UInt64],
    out_total: _OutPtr[UInt64],
    ctx: _DeviceContextBoxPtr,
) abi("C") -> _CString[]:
    try:
        var info = ctx[].get_memory_info()
        out_free[] = UInt64(info[0])
        out_total[] = UInt64(info[1])
        return None
    except e:
        return _err_cstr(String(e))


@export
def MojoHAL_DeviceContext_getAttribute(
    out_value: _OutPtr[Int32],
    ctx: _DeviceContextBoxPtr,
    attr: Int32,
) abi("C") -> _CString[]:
    # `attr` is a CUDA-style `CUdevice_attribute` code (the values of
    # `DeviceAttribute`); the plugin translates it to its native equivalent.
    try:
        out_value[] = Int32(ctx[].get_attribute(DeviceAttribute(attr)))
        return None
    except e:
        return _err_cstr(String(e))


# ===---------------------------------------------------------------------=== #
# Sub-buffers / ownership / memmap (phase 2)
# ===---------------------------------------------------------------------=== #


@export
def MojoHAL_HostBuffer_createSubBuffer(
    out_buf: _OutPtr[_DeviceBufferBoxPtr],
    parent: _DeviceBufferBoxPtr,
    byte_offset: UInt64,
    byte_size: UInt64,
) abi("C") -> _CString[]:
    # Host boxes carry a `HostBuffer` (see MojoHAL_HostBuffer_release), but the
    # C++ side boxes every sub-buffer as a `DeviceBuffer`, so the view is a
    # plain `DeviceBuffer` over the pinned host region (host-pinned memory is
    # device-visible) rather than `HostBuffer.create_sub_buffer`'s `HostBuffer`.
    ref parent_buf = parent.unsafe_bitcast[HostBuffer[.uint8]]()[]
    var parent_size = UInt64(len(parent_buf))
    if byte_size > parent_size or parent_size - byte_size < byte_offset:
        return _err_cstr("offset, len and elem_size are out of range")
    var view = DeviceBuffer[.uint8](
        parent_buf.context(),
        parent_buf.unsafe_ptr().unsafe_offset(Int(byte_offset)),
        Int(byte_size),
        owning=False,
    )
    # The view's wrapper is non-owning; retaining the parent's inner keeps
    # the pinned allocation alive for the view's lifetime.
    view._inner[]._parent_host = [parent_buf._inner]
    var boxed = alloc(Layout[DeviceBuffer[.uint8]].single())
    boxed.unsafe_ptr().unsafe_write(view^)
    out_buf[] = boxed^.unsafe_leak()
    return None


@export
def MojoHAL_DeviceBuffer_createSubBuffer(
    out_buf: _OutPtr[_DeviceBufferBoxPtr],
    parent: _DeviceBufferBoxPtr,
    byte_offset: UInt64,
    byte_size: UInt64,
) abi("C") -> _CString[]:
    # The returned box is a non-owning view into the parent allocation; the
    # C++ DeviceBuffer holds a reference to its parent to keep it alive.
    ref parent_buf = parent[]
    var parent_size = UInt64(len(parent_buf))
    if byte_size > parent_size or parent_size - byte_size < byte_offset:
        return _err_cstr("offset, len and elem_size are out of range")
    var view = DeviceBuffer[.uint8](
        parent_buf.context(),
        parent_buf.unsafe_ptr().unsafe_offset(Int(byte_offset)),
        Int(byte_size),
        owning=False,
    )
    var boxed = alloc(Layout[DeviceBuffer[.uint8]].single())
    boxed.unsafe_ptr().unsafe_write(view^)
    out_buf[] = boxed^.unsafe_leak()
    return None


@export
def MojoHAL_DeviceContext_reassignWait(
    dst_ctx: _DeviceContextBoxPtr,
    src_ctx: _DeviceContextBoxPtr,
) abi("C") -> _CString[]:
    # Ordering half of DeviceBuffer::reassignOwnershipTo: the destination
    # context waits for all work enqueued on the source context, after which
    # the C++ shim rebinds the buffer's context reference.
    try:
        dst_ctx[].enqueue_wait_for(src_ctx[])
        return None
    except e:
        return _err_cstr(String(e))


# A memmapped file wrapped as a host-visible buffer view; munmaps on release.
@fieldwise_init
struct _MmapBox(Movable):
    # `view` MUST stay the first field: MojoHAL_DeviceBuffer_createSubBuffer
    # bitcasts buffer boxes to `DeviceBuffer[DType.uint8]`, which reads the
    # leading view of an `_MmapBox` parent.
    var view: DeviceBuffer[.uint8]
    var base: _RawMemoryPtr
    var length: UInt64


@export
def MojoHAL_DeviceContext_memmapFile(
    out_buf: _OutPtr[_DeviceBufferBoxPtr],
    out_ptr: _OutPtr[_RawMemoryPtr],
    out_size: _OutPtr[UInt64],
    ctx: _DeviceContextBoxPtr,
    path: Pointer[mut=False, c_char, ImmUntrackedOrigin],
) abi("C") -> _CString[]:
    # Mirrors CpuDeviceContext::memmapFileImpl: read-only mapping of the whole
    # file; the mapping is released (munmap) when the buffer box is released.
    comptime O_RDONLY = Int32(0)
    comptime PROT_READ = Int32(1)
    comptime MAP_PRIVATE = Int32(2)
    comptime SEEK_END = Int32(2)

    var fd = external_call["open", Int32](path, O_RDONLY)
    if fd < 0:
        return _err_cstr(
            String("failed to open file: ", String(unsafe_from_utf8_ptr=path))
        )
    var size = external_call["lseek", Int64](fd, Int64(0), SEEK_END)
    if size < 0:
        _ = external_call["close", Int32](fd)
        return _err_cstr("failed to query file size")
    # NULL address hint, modeled as Optional's null niche.
    var no_hint = OptionalPointer[NoneType, MutUntrackedOrigin](None)
    var base = external_call["mmap", _RawMemoryPtr](
        no_hint,
        UInt64(size),
        PROT_READ,
        MAP_PRIVATE,
        fd,
        Int64(0),
    )
    # The mapping holds its own reference to the file.
    _ = external_call["close", Int32](fd)
    if Int(base) == -1:  # MAP_FAILED
        return _err_cstr("failed to mmap file")

    var view = DeviceBuffer[.uint8](ctx[], base, Int(size), owning=False)
    var boxed = alloc(Layout[_MmapBox].single())
    boxed.unsafe_ptr().unsafe_write(_MmapBox(view^, base, UInt64(size)))
    out_buf[] = boxed^.unsafe_leak().unsafe_bitcast[DeviceBuffer[.uint8]]()
    out_ptr[] = base
    out_size[] = UInt64(size)
    return None


@export
def MojoHAL_MmapBuffer_release(buf: _DeviceBufferBoxPtr) abi("C"):
    var boxed = buf.unsafe_bitcast[_MmapBox]()
    var base = boxed[].base
    var length = boxed[].length
    boxed.unsafe_deinit_pointee()
    dealloc(
        ThinAllocation(unsafe_owned_ptr=boxed).unsafe_with_layout(
            Layout[_MmapBox].single()
        )
    )
    _ = external_call["munmap", Int32](base, length)


# ===---------------------------------------------------------------------=== #
# Streams / events / thread binding (phase 3)
# ===---------------------------------------------------------------------=== #


@export
def MojoHAL_DeviceContext_stream(
    out_stream: _OutPtr[_DeviceStreamBoxPtr],
    ctx: _DeviceContextBoxPtr,
) abi("C") -> _CString[]:
    var stream = ctx[].stream()
    var boxed = alloc(Layout[DeviceStream].single())
    boxed.unsafe_ptr().unsafe_write(stream^)
    out_stream[] = boxed^.unsafe_leak()
    return None


@export
def MojoHAL_DeviceContext_createStream(
    out_stream: _OutPtr[_DeviceStreamBoxPtr],
    ctx: _DeviceContextBoxPtr,
    priority: Int32,
) abi("C") -> _CString[]:
    try:
        var stream = ctx[].create_stream(priority=Int(priority))
        var boxed = alloc(Layout[DeviceStream].single())
        boxed.unsafe_ptr().unsafe_write(stream^)
        out_stream[] = boxed^.unsafe_leak()
        return None
    except e:
        return _err_cstr(String(e))


@export
def MojoHAL_DeviceContext_withStream(
    out_ctx: _OutPtr[_DeviceContextBoxPtr],
    ctx: _DeviceContextBoxPtr,
    stream: _DeviceStreamBoxPtr,
) abi("C") -> _CString[]:
    # A copy of `ctx` whose current stream is `stream` — backs the
    # `DeviceContext::selectStream` view on the C++ side.
    var view = ctx[]
    view._stream = stream[]._hal_stream()
    var boxed = alloc(Layout[DeviceContext].single())
    boxed.unsafe_ptr().unsafe_write(view^)
    out_ctx[] = boxed^.unsafe_leak()
    return None


@export
def MojoHAL_DeviceStream_release(stream: _DeviceStreamBoxPtr) abi("C"):
    var boxed = stream
    boxed.unsafe_deinit_pointee()
    dealloc(
        ThinAllocation(unsafe_owned_ptr=boxed).unsafe_with_layout(
            Layout[DeviceStream].single()
        )
    )


@export
def MojoHAL_DeviceStream_synchronize(
    stream: _DeviceStreamBoxPtr,
) abi("C") -> _CString[]:
    try:
        stream[].synchronize()
        return None
    except e:
        return _err_cstr(String(e))


@export
def MojoHAL_DeviceStream_waitForEvent(
    stream: _DeviceStreamBoxPtr,
    event: _DeviceEventBoxPtr,
) abi("C") -> _CString[]:
    try:
        stream[].enqueue_wait_for(event[])
        return None
    except e:
        return _err_cstr(String(e))


@export
def MojoHAL_DeviceStream_nativeHandle(
    out_handle: _OutPtr[_RawMemoryPtr],
    stream: _DeviceStreamBoxPtr,
) abi("C") -> _CString[]:
    try:
        out_handle[] = stream[]._native_stream().unsafe_bitcast[Byte]()
        return None
    except e:
        return _err_cstr(String(e))


@export
def MojoHAL_DeviceEvent_isReady(
    out_ready: _OutPtr[Bool],
    event: _DeviceEventBoxPtr,
) abi("C") -> _CString[]:
    try:
        var ev = event
        if not ev[]._event[]:
            # An unrecorded event reports ready, matching the legacy
            # backends (cuEventQuery on a fresh event succeeds).
            out_ready[] = True
        else:
            out_ready[] = ev[]._event[].value().is_ready()
        return None
    except e:
        return _err_cstr(String(e))


@export
def MojoHAL_DeviceContext_eventCreate(
    out_event: _OutPtr[_DeviceEventBoxPtr],
    ctx: _DeviceContextBoxPtr,
    record: Bool,
) abi("C") -> _CString[]:
    # `record=True` creates an event already recorded on the context's stream
    # (DeviceContext::enqueueEvent); `record=False` creates an unrecorded event
    # (DeviceContext::eventCreate).
    try:
        var event: DeviceEvent
        if record:
            event = DeviceEvent(ctx[])
        else:
            event = DeviceEvent._create_unrecorded(ctx[])
        var boxed = alloc(Layout[DeviceEvent].single())
        boxed.unsafe_ptr().unsafe_write(event^)
        out_event[] = boxed^.unsafe_leak()
        return None
    except e:
        return _err_cstr(String(e))


@export
def MojoHAL_DeviceEvent_release(event: _DeviceEventBoxPtr) abi("C"):
    var boxed = event
    boxed.unsafe_deinit_pointee()
    dealloc(
        ThinAllocation(unsafe_owned_ptr=boxed).unsafe_with_layout(
            Layout[DeviceEvent].single()
        )
    )


@export
def MojoHAL_DeviceEvent_synchronize(
    event: _DeviceEventBoxPtr,
) abi("C") -> _CString[]:
    try:
        event[].synchronize()
        return None
    except e:
        return _err_cstr(String(e))


@export
def MojoHAL_DeviceContext_enqueueEventRecord(
    ctx: _DeviceContextBoxPtr,
    event: _DeviceEventBoxPtr,
) abi("C") -> _CString[]:
    try:
        ctx[].stream().record_event(event[])
        return None
    except e:
        return _err_cstr(String(e))


@export
def MojoHAL_DeviceContext_enqueueWaitForEvent(
    ctx: _DeviceContextBoxPtr,
    event: _DeviceEventBoxPtr,
) abi("C") -> _CString[]:
    try:
        ctx[].stream().enqueue_wait_for(event[])
        return None
    except e:
        return _err_cstr(String(e))


@export
def MojoHAL_DeviceContext_bindToCurrentThread(
    ctx: _DeviceContextBoxPtr,
) abi("C") -> _CString[]:
    try:
        ctx[]._context[].set_current()
        return None
    except e:
        return _err_cstr(String(e))


@export
def MojoHAL_DeviceContext_pushContext(
    out_scope: _OutPtr[_DeviceContextScopeBoxPtr],
    ctx: _DeviceContextBoxPtr,
) abi("C") -> _CString[]:
    # Saves the calling thread's current driver context and makes `ctx`'s
    # current; restored by MojoHAL_DeviceContextScope_pop.
    try:
        var pushed = ctx[].push_context()
        var boxed = alloc(Layout[_DeviceContextScopeHAL].single())
        var ptr = boxed.unsafe_ptr()
        ptr.unsafe_write(pushed^)
        try:
            _ = ptr[].__enter__()
        except e:
            ptr.unsafe_deinit_pointee()
            dealloc(boxed^)
            return _err_cstr(String(e))
        out_scope[] = boxed^.unsafe_leak()
        return None
    except e:
        return _err_cstr(String(e))


@export
def MojoHAL_DeviceContextScope_pop(scope: _DeviceContextScopeBoxPtr) abi("C"):
    var boxed = scope
    try:
        boxed[].__exit__()
    except e:
        print("warning: restoring driver context failed:", e)
    boxed.unsafe_deinit_pointee()
    dealloc(
        ThinAllocation(unsafe_owned_ptr=boxed).unsafe_with_layout(
            Layout[_DeviceContextScopeHAL].single()
        )
    )


@export
def MojoHAL_DeviceContext_getDLPackStream(
    out_stream: _OutPtr[_RawMemoryPtr],
    ctx: _DeviceContextBoxPtr,
) abi("C") -> Bool:
    # Returns False when the backend exposes no native stream handle (the C++
    # shim maps that to std::nullopt).
    try:
        # The handle's tied origin (and its immutability, from the borrow of
        # the temporary `DeviceStream`) ends at the C ABI: the C++ caller's
        # wrapper keeps the stream alive.
        out_stream[] = (
            ctx[]
            .stream()
            ._native_stream()
            .unsafe_bitcast[Byte]()
            .unsafe_mut_cast[True]()
            .unsafe_origin_cast[MutUntrackedOrigin]()
        )
        return True
    except:
        return False


# ===---------------------------------------------------------------------=== #
# Completion flags / host-value waits
# ===---------------------------------------------------------------------=== #


# A device-mapped pinned 8-byte slot the host signals and the device waits on
# (mirrors M::Driver::CompletionFlag over the legacy backend).
@fieldwise_init
struct _CompletionFlagBox(Movable):
    var ctx: DeviceContext
    var buffer: _HALRawBuffer
    var host_ptr: _RawMemoryPtr
    var device_addr: UInt64


@export
def MojoHAL_DeviceContext_allocateCompletionFlag(
    out_flag: _OutPtr[_CompletionFlagBoxPtr],
    out_host_ptr: _OutPtr[_RawMemoryPtr],
    out_device_addr: _OutPtr[UInt64],
    ctx: _DeviceContextBoxPtr,
) abi("C") -> _CString[]:
    try:
        ref device_ctx = ctx[]
        ref hal_ctx = device_ctx._context[]
        comptime slot_size = UInt64(size_of[UInt64]())
        var buffer = hal_ctx.alloc_host_pinned(slot_size)
        var host_ptr = hal_ctx.memory_get_host_address[MutUntrackedOrigin](
            buffer
        ).unsafe_bitcast[Byte]()
        var device_addr = hal_ctx.memory_get_address(buffer)
        # Initialize the flag to zero so the first wait observes a clean state.
        Atomic[UInt64].store[ordering=Ordering.RELAXED](
            host_ptr.unsafe_bitcast[UInt64](), 0
        )
        var boxed = alloc(Layout[_CompletionFlagBox].single())
        boxed.unsafe_ptr().unsafe_write(
            _CompletionFlagBox(device_ctx, buffer^, host_ptr, device_addr)
        )
        out_flag[] = boxed^.unsafe_leak()
        out_host_ptr[] = host_ptr
        out_device_addr[] = device_addr
        return None
    except e:
        return _err_cstr(String(e))


@export
def MojoHAL_CompletionFlag_release(flag: _CompletionFlagBoxPtr) abi("C"):
    var boxed = flag
    var box = boxed.unsafe_take_pointee()
    dealloc(
        ThinAllocation(unsafe_owned_ptr=boxed).unsafe_with_layout(
            Layout[_CompletionFlagBox].single()
        )
    )
    # `Buffer` has no destructor (a plain handle + size), so rebuild one from
    # the box's copyable fields for the consuming `free_host_pinned` call and
    # let `box` drop normally.
    var buffer = _HALRawBuffer(
        _handle=box.buffer._handle,
        byte_size=box.buffer.byte_size,
        is_host_pinned=True,
        _context=box.ctx._context,
        _extensions=box.buffer._extensions,
    )
    try:
        box.ctx._context[].free_host_pinned(buffer^)
    except e:
        print("warning: failed to free completion flag:", e)


@export
def MojoHAL_CompletionFlag_signal(
    host_ptr: _RawMemoryPtr, value: UInt64
) abi("C"):
    # Release pairs with the device-side wait (and `load`'s acquire), mirroring
    # CompletionFlag::signal.
    Atomic[UInt64].store[ordering=Ordering.RELEASE](
        host_ptr.unsafe_bitcast[UInt64](), value
    )


@export
def MojoHAL_CompletionFlag_load(
    host_ptr: _RawMemoryPtr,
) abi("C") -> UInt64:
    return Atomic[UInt64].load[ordering=Ordering.ACQUIRE](
        host_ptr.unsafe_bitcast[UInt64]()
    )


@export
def MojoHAL_CompletionFlag_reset(host_ptr: _RawMemoryPtr) abi("C"):
    # Relaxed, mirroring CompletionFlag::reset: stream-ordered enqueue of the
    # wait happens-after this store.
    Atomic[UInt64].store[ordering=Ordering.RELAXED](
        host_ptr.unsafe_bitcast[UInt64](), 0
    )


@export
def MojoHAL_DeviceStream_waitOnHostValue(
    stream: _DeviceStreamBoxPtr,
    device_addr: UInt64,
    value: UInt64,
) abi("C") -> _CString[]:
    try:
        stream[]._stream[].wait_value64(device_addr, value)
        return None
    except e:
        return _err_cstr(String(e))
