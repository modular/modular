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
"""Dispatching `DeviceBuffer`, and the host-mapped view of one."""

from std.builtin.device_passable import (
    DevicePassable,
    DeviceTypeEncoder,
)
from std.format import Writer
from std.ffi import external_call
from std.memory import ArcPointer, Pointer, alloc
from std.os import abort
from std.sys import is_gpu, size_of

from max._driver.max_license.hal import Buffer, Context, get_device_spec
from max._driver.max_license.hal.extensions.traits import SupportsHostPointerV1
from max.gpu.host._driver_handles import (
    _DeviceBufferPtr as _HALDeviceBufferPtr,
)
from max._driver.max_license.device_context_dispatch.device_pointer import (
    DevicePointer,
)
from max.gpu.host._legacy.device_context import LegacyDeviceBuffer
from max.gpu.host._driver_handles import (
    _DeviceBufferPtr as _LegacyDeviceBufferPtr,
)
from .device_context import use_legacy_backend
from max.gpu.host.info import default_accelerator
from .host_types import Device
from max._driver.max_license.device_context_dispatch.device_context import (
    DeviceContext,
)
from max._driver.max_license.device_context_dispatch.host_buffer import (
    HostBuffer,
    _HostBufferInner,
)


@fieldwise_init
struct _HALBufferInner(Movable):
    """Owning wrapper around a non-owning HAL Buffers.

    Owns the `Buffer` and holds a reference to the parent `Context` so
    destruction can call `Context.free_sync`.
    """

    var _buffer: Buffer[get_device_spec[0]()]
    var _context: ArcPointer[Context[get_device_spec[0]()]]
    var _device_addr: UInt64
    # For sub-buffer views: keeps the parent allocation alive for the view's
    # lifetime (a view's own `_buffer` is a non-owning wrapper). A `List` (heap
    # storage) rather than `Optional` so the struct does not embed itself.
    var _parent: List[ArcPointer[_HALBufferInner]]
    # Same, for `DeviceBuffer` views over a `HostBuffer`'s pinned allocation,
    # which the HAL C exports create for the C++ `DeviceContext`.
    var _parent_host: List[ArcPointer[_HostBufferInner]]

    def __deinit__(deinit self):
        try:
            self._context[].free_sync(self._buffer^)
        except e:
            print("warning: free_sync failed:", e)
        # Release the parent only after the view's wrapper is freed.
        _ = self._parent^


def _sub_buffer_byte_range(
    offset: Int, size: Int, elem_size: Int, parent_byte_size: UInt64
) raises -> Tuple[Int, Int]:
    """Returns the byte offset and byte size of a sub-buffer view.

    Args:
        offset: The view's first element.
        size: The view's element count.
        elem_size: The size of one view element, in bytes.
        parent_byte_size: The size of the parent buffer, in bytes.

    Returns:
        The view's byte offset and byte size within the parent.

    Raises:
        If the view does not fit inside the parent.
    """
    # Check in elements so a huge `offset` or `size` can't wrap the byte
    # arithmetic back into range.
    var max_elems = Int(parent_byte_size) // elem_size
    if offset < 0 or size < 0 or size > max_elems or offset > max_elems - size:
        raise Error("create_sub_buffer: view is out of range")
    return (offset * elem_size, size * elem_size)


@fieldwise_init
struct _HALDeviceBuffer[dtype: DType](ImplicitlyCopyable, Movable):
    """The state of a `DeviceBuffer` on the HAL backend."""

    comptime _DevicePtr = Pointer[Scalar[Self.dtype], MutUntrackedOrigin]

    # `_device_ptr` must be the first member in the struct. Kernel dispatch code
    # `rebind`s a `DeviceBuffer` directly to its `device_type` (an `Pointer`),
    # which reinterprets the struct's initial data as the device address.
    var _device_ptr: Self._DevicePtr
    var _ctx: DeviceContext.Unknown
    # Wrap the inner buffer in an ArcPointer so copies of this DeviceBuffer hold
    # a reference to the same underlying buffer.
    var _inner: ArcPointer[_HALBufferInner]


struct DeviceBuffer[dtype: DType](
    DevicePassable, ImplicitlyCopyable, Sized, Writable
):
    """Represents a block of device-resident storage. For GPU devices, a device
    buffer is allocated in the device's global memory.

    To allocate a `DeviceBuffer`, use one of the methods provided by
    `DeviceContext`, such as
    [`enqueue_create_buffer()`](/api/mojo/max/gpu/host/device_context/DeviceContext/#enqueue_create_buffer).

    Parameters:
        dtype: Data dtype to be stored in the buffer.
    """

    # ===-------------------------------------------------------------------===#
    # Backend selection
    # ===-------------------------------------------------------------------===#

    # `DeviceBuffer` carries no target, so it takes the default accelerator's
    # backend. A context that produces or consumes one asserts agreement.
    comptime _Legacy = LegacyDeviceBuffer[Self.dtype]
    comptime _HAL = _HALDeviceBuffer[Self.dtype]
    comptime _Impl = Self._Legacy if use_legacy_backend() else Self._HAL

    # ===-------------------------------------------------------------------===#
    # Fields
    # ===-------------------------------------------------------------------===#

    # The backend buffer must stay the only field, and it must keep the device
    # address as its own first word: kernel dispatch reinterprets the start of a
    # `DeviceBuffer` as the device pointer.
    var _impl: Self._Impl

    # ===-------------------------------------------------------------------===#
    # Aliases
    # ===-------------------------------------------------------------------===#

    comptime _DevicePtr = Pointer[Scalar[Self.dtype], MutUntrackedOrigin]

    # ===-------------------------------------------------------------------===#
    # Life cycle methods
    # ===-------------------------------------------------------------------===#

    @doc_hidden
    @inline(.always)
    def __init__(out self, ctx: DeviceContext, size: Int) raises:
        """This init takes in a constructed `DeviceContext` and schedules an
        owned buffer allocation using the stream in the device context.
        """
        comptime __guard not use_legacy_backend() else:
            self._impl = Self._Legacy(
                ctx._impl,
                size=size,
            )
            return

        var byte_size = UInt64(size * size_of[Self.dtype]())
        # Cache the GPU address up front so `unsafe_ptr` is non-raising.
        var buffer = ctx._impl._context[].alloc_sync(byte_size)
        var addr = UInt64(0)
        if byte_size > 0:
            addr = ctx._impl._context[].memory_get_address(buffer)
        self._impl = Self._HAL(
            Self._DevicePtr(unsafe_from_address=Int(addr)),
            ctx.into_unknown_target(),
            ArcPointer(
                _HALBufferInner(buffer^, ctx._impl._context, addr, [], [])
            ),
        )

    @doc_hidden
    def __init__(
        out self,
        ctx: DeviceContext,
        ptr: Pointer[Scalar[Self.dtype], ...],
        size: Int,
        *,
        owning: Bool,
    ) raises:
        """Constructs a `DeviceBuffer` that wraps an externally-managed device
        pointer.

        When `owning` is True the wrapper takes ownership of the underlying
        allocation and will free it when the last reference drops. When
        `owning` is False the wrapper is non-owning; releasing it leaves the
        underlying pointer to whoever allocated it.

        Args:
            ctx: The device context the pointer belongs to.
            ptr: The device pointer to wrap.
            size: Number of elements addressable through the pointer.
            owning: Whether releasing this buffer should free the pointer.

        Raises:
            If the HAL plugin refuses to wrap the memory.
        """
        comptime __guard not use_legacy_backend() else:
            self._impl = Self._Legacy(
                ctx._impl,
                rebind[Self._DevicePtr](ptr),
                size,
                owning=owning,
            )
            return

        var byte_size = UInt64(size * size_of[Self.dtype]())
        var addr = UInt64(Int(ptr))
        try:
            # The plugin rejects wrapping a null address; represent the empty
            # buffer the way a zero-byte allocation does (null memory handle).
            var buffer = ctx._impl._context[].alloc_sync(
                0
            ) if addr == 0 else ctx._impl._context[].wrap_memory(
                addr, byte_size, owning=owning
            )
            # The plugin may give the wrapped region an address of its own.
            # If it can't report one, keep the address we were given.
            if addr != 0:
                try:
                    addr = ctx._impl._context[].memory_get_address(buffer)
                except:
                    pass
            self._impl = Self._HAL(
                Self._DevicePtr(unsafe_from_address=Int(addr)),
                ctx.into_unknown_target(),
                ArcPointer(
                    _HALBufferInner(buffer^, ctx._impl._context, addr, [], [])
                ),
            )
        except e:
            raise Error(t"DeviceBuffer: failed to wrap external memory: {e}")

    def __init__(out self, *, var _impl: Self._Impl):
        """Wraps an already-constructed backend buffer.

        Args:
            _impl: The backend buffer to take ownership of.
        """
        self._impl = _impl^

    # ===-------------------------------------------------------------------===#
    # Factory methods
    # ===-------------------------------------------------------------------===#

    @staticmethod
    @doc_hidden
    def empty(context: DeviceContext) -> Self:
        comptime __guard not use_legacy_backend() else:
            return Self(_impl=Self._Legacy.empty(context._impl))

        # `empty` has no error channel, and only a broken plugin refuses to
        # wrap zero bytes.
        try:
            return Self(
                context, Self._DevicePtr.unsafe_dangling(), 0, owning=False
            )
        except e:
            abort(String(e))

    # ===-------------------------------------------------------------------===#
    # Trait: DevicePassable
    # ===-------------------------------------------------------------------===#

    comptime device_type: AnyType = Pointer[
        mut=True, Scalar[Self.dtype], AnyOrigin[mut=True]
    ]
    """DeviceBuffer dtypes are remapped to a device `Pointer` when passed to accelerator devices."""

    def _to_device_type(
        self, mut encoder: Some[DeviceTypeEncoder], target: MutOpaquePointer[_]
    ):
        """Device dtype mapping from DeviceBuffer to the device's `Pointer`."""
        comptime __guard not use_legacy_backend() else:
            self._impl._to_device_type(encoder, target)
            return

        # An empty buffer has no device pointer, so nothing is encoded.
        try:
            encoder.encode_device_ptr(self.device_ptr(), target)
        except:
            pass

    @staticmethod
    def get_type_name() -> String:
        """Gets this dtype's name, for use in error messages when handing
        arguments to kernels.

        Returns:
            This dtype's name.
        """
        return String(t"DeviceBuffer[{Self.dtype}]")

    # ===-------------------------------------------------------------------===#
    # Escape hatches
    # ===-------------------------------------------------------------------===#

    def unwrap_legacy(self) -> Self._Legacy:
        """Returns the underlying legacy buffer.

        Only valid in a build whose backend is the legacy one.

        Returns:
            The legacy buffer this wrapper holds.
        """
        comptime assert (
            use_legacy_backend()
        ), "unwrap_legacy is only available in a legacy DeviceContext build"
        return self._impl

    def unwrap_hal(self) -> Self._HAL:
        """Returns the underlying HAL buffer.

        Only valid in a build whose backend is the HAL one.

        Returns:
            The HAL buffer this wrapper holds.
        """
        comptime assert (
            not use_legacy_backend()
        ), "unwrap_hal is only available in a HAL DeviceContext build"
        return self._impl

    def _legacy_buffer(ref self) -> ref[self] Self._Legacy:
        comptime assert (
            use_legacy_backend()
        ), "_legacy_buffer is only available in a legacy DeviceContext build"
        return rebind[Pointer[Self._Legacy, origin_of(self)]](
            Pointer(to=self._impl)
        )[]

    # ===-------------------------------------------------------------------===#
    # Trait implementations
    # ===-------------------------------------------------------------------===#

    def __len__(self) -> Int:
        """Returns the number of elements in this buffer.

        This method calculates the number of elements by dividing the total byte size
        of the buffer by the size of each element.

        Returns:
            The number of elements in the buffer.
        """
        comptime __guard not use_legacy_backend() else:
            return len(self._impl)

        return (
            Int(self._impl._inner[]._buffer.byte_size) // size_of[Self.dtype]()
        )

    # ===-------------------------------------------------------------------===#
    # Methods
    # ===-------------------------------------------------------------------===#

    @inline(.always)
    def create_sub_buffer[
        view_type: DType
    ](self, offset: Int, size: Int) raises -> DeviceBuffer[view_type]:
        """Creates a sub-buffer view of this buffer with a different element dtype.

        This method creates a new buffer that references a subset of the memory in this
        buffer, potentially with a different element dtype. The sub-buffer shares the
        underlying memory with the original buffer.

        Parameters:
            view_type: Element dtype of the sub-buffer.

        Args:
            offset: Offset in bytes from the start of this buffer.
            size: Size of the sub-buffer in elements of `view_type`.

        Returns:
            A device buffer viewing the requested range.

        Raises:
            If the range is invalid.
        """
        comptime __guard not use_legacy_backend() else:
            return DeviceBuffer[view_type](
                _impl=self._impl.create_sub_buffer[view_type](offset, size)
            )

        var byte_range = _sub_buffer_byte_range(
            offset,
            size,
            size_of[view_type](),
            self._impl._inner[]._buffer.byte_size,
        )
        var view = DeviceBuffer[view_type](
            self.context(),
            Pointer[Scalar[view_type], MutUntrackedOrigin](
                unsafe_from_address=Int(
                    self._impl._inner[]._device_addr + UInt64(byte_range[0])
                )
            ),
            size,
            owning=False,
        )
        # The view's wrapper is non-owning; retaining the parent's inner keeps
        # the shared allocation alive for the view's lifetime.
        view._impl._inner[]._parent = [self._impl._inner]
        return view^

    def unsafe_ptr[
        mut: Bool, //, origin: Origin[mut=mut]
    ](ref[origin] self) -> Pointer[Scalar[Self.dtype], origin]:
        """Returns the raw device pointer without transferring ownership.

        This method provides direct access to the underlying device pointer
        for advanced use cases. The buffer retains ownership of the pointer.

        Parameters:
            mut: Whether the returned pointer is mutable.
            origin: The origin of the buffer reference.

        Returns:
            The raw device pointer owned by this buffer.
        """
        comptime __guard not use_legacy_backend() else:
            return rebind[Pointer[Scalar[Self.dtype], origin]](
                self._impl.unsafe_ptr()
            )

        # The HAL reports an untracked origin; rebuild it against the caller's
        # borrow so the result behaves like the legacy pointer.
        return Pointer[Scalar[Self.dtype], origin](
            unsafe_from_address=Int(self._impl._device_ptr)
        )

    def unsafe_host_ptr[
        mut: Bool, //, origin: Origin[mut=mut]
    ](ref[origin] self) raises -> Pointer[Scalar[Self.dtype], origin]:
        """Returns the host-accessible pointer for this device buffer.

        Only unified-memory devices expose one; other devices raise.

        Parameters:
            mut: Whether the returned pointer is mutable.
            origin: The origin of the buffer reference.

        Returns:
            A pointer to the same memory, addressable from the host.

        Raises:
            If this device's memory is not host-accessible.
        """
        comptime __guard case False = use_legacy_backend() else:
            comptime assert use_legacy_backend()  # TODO(MOCO-5393)
            return rebind[Pointer[Scalar[Self.dtype], origin]](
                self._impl.unsafe_host_ptr()
            )
        comptime assert not use_legacy_backend()  # TODO(MOCO-5393)

        comptime assert not is_gpu(), "DeviceBuffer is not supported on GPUs"
        # A backend without the extension raises rather than failing to
        # compile, matching the legacy `DeviceContext`.
        comptime Ext = type_of(self._impl._inner[]._context[].extensions())
        comptime if conforms_to(Ext, SupportsHostPointerV1):
            ref ext = self._impl._inner[]._context[].extensions()
            var host_ptr = ext.unsafe_host_ptr(self._impl._inner[]._buffer)
            return (
                host_ptr.unsafe_bitcast[Scalar[Self.dtype]]()
                .unsafe_mut_cast[mut]()
                .unsafe_origin_cast[origin]()
            )
        else:
            raise Error(
                "DeviceBuffer.unsafe_host_ptr: this device's HAL plugin does"
                " not support host pointers"
            )

    def context(self) raises -> DeviceContext[]:
        """Returns the device context associated with this buffer.

        Returns:
            The device context that owns this buffer.

        Raises:
            If the query fails.
        """
        comptime __guard not use_legacy_backend() else:
            return DeviceContext[](_impl=self._impl.context())

        return rebind_var[DeviceContext[]](self._impl._ctx)

    @inline(.always)
    def take_ptr(var self) -> Self._DevicePtr:
        """Takes ownership of the device pointer from this buffer.

        This method releases the device pointer from the buffer's control and
        returns it to the caller. After this call, the buffer no longer owns
        the pointer, and the caller is responsible for managing its lifecycle.

        Returns:
            The raw device pointer that was owned by this buffer.
        """
        comptime __guard not use_legacy_backend() else:
            return self._impl.take_ptr()

        # Ownership of the underlying allocation transfers to the caller: the
        # wrapper handle (freed when the last reference drops) no longer frees
        # the memory.
        try:
            _ = self._impl._ctx._impl._context[].unwrap_memory(
                self._impl._inner[]._buffer
            )
        except e:
            abort("DeviceBuffer.take_ptr: failed to release ownership")
        return Self._DevicePtr(
            unsafe_from_address=Int(self._impl._inner[]._device_addr)
        )

    def enqueue_copy_to(self, dst: DeviceBuffer[Self.dtype]) raises:
        """Enqueues an asynchronous copy from this buffer to another device
        buffer.

        This method schedules a memory copy operation from this buffer to the
        destination buffer. The operation is asynchronous and will be executed
        in the stream associated with this buffer's context.

        Args:
            dst: Device buffer to copy to.

        Raises:
            If the operation fails.
        """
        comptime __guard not use_legacy_backend() else:
            self._impl.enqueue_copy_to(dst._impl)
            return

        self.context().enqueue_copy(dst, self)

    def enqueue_copy_to(self, dst: HostBuffer[Self.dtype]) raises:
        """Enqueues an asynchronous copy from this buffer to a host buffer.

        This method schedules a memory copy operation from this buffer to the
        destination host buffer. The operation is asynchronous and will be
        executed in the stream associated with this buffer's context.

        Args:
            dst: Host buffer to copy to.

        Raises:
            If the operation fails.
        """
        comptime __guard not use_legacy_backend() else:
            self._impl.enqueue_copy_to(dst._impl)
            return

        self.context().enqueue_copy(dst, self)

    def enqueue_copy_to(
        self, dst_ptr: Pointer[mut=True, Scalar[Self.dtype], _]
    ) raises:
        """Enqueues an asynchronous copy from this buffer to host memory.

        Args:
            dst_ptr: Pointer to the destination host memory location.

        Raises:
            If the operation fails.
        """
        self.context().enqueue_copy(dst_ptr, self)

    def enqueue_copy_from(self, src: DeviceBuffer[Self.dtype]) raises:
        """Enqueues an asynchronous copy to this buffer from another device
        buffer.

        This method schedules a memory copy operation to this buffer from the
        source buffer. The operation is asynchronous and will be executed in
        the stream associated with this buffer's context.

        Args:
            src: Device buffer to copy from.

        Raises:
            If the operation fails.
        """
        comptime __guard not use_legacy_backend() else:
            self._impl.enqueue_copy_from(src._impl)
            return

        self.context().enqueue_copy(self, src)

    def enqueue_copy_from(self, src: HostBuffer[Self.dtype]) raises:
        """Enqueues an asynchronous copy from a host buffer to this buffer.

        This method schedules a memory copy operation from the source host
        buffer to this buffer. The operation is asynchronous and will be
        executed in the stream associated with this buffer's context.

        Args:
            src: Host buffer to copy from.

        Raises:
            If the operation fails.
        """
        comptime __guard not use_legacy_backend() else:
            self._impl.enqueue_copy_from(src._impl)
            return

        self.context().enqueue_copy(self, src)

    def enqueue_copy_from(
        self, src_ptr: Pointer[mut=False, Scalar[Self.dtype], _]
    ) raises:
        """Enqueues an asynchronous copy to this buffer from host memory.

        This method schedules a memory copy operation to this buffer from the
        given host pointer. The operation is asynchronous and will be executed
        in the stream associated with this buffer's context.

        Args:
            src_ptr: Pointer to the source host memory.

        Raises:
            If the operation fails.
        """
        comptime __guard not use_legacy_backend() else:
            self._impl.enqueue_copy_from(src_ptr)
            return

        self.context().enqueue_copy(self, src_ptr)

    def enqueue_copy_from(
        self, src: Span[mut=False, Scalar[Self.dtype], _]
    ) raises:
        """Enqueues an asynchronous copy from a host `Span` to this buffer.

        Args:
            src: Host span to copy from.

        Raises:
            If the operation fails.
        """
        comptime __guard not use_legacy_backend() else:
            self._impl.enqueue_copy_from(src)
            return

        self.context().enqueue_copy(self, src)

    def enqueue_copy_to(
        self, dst: Span[mut=True, Scalar[Self.dtype], _]
    ) raises:
        """Enqueues an asynchronous copy from this buffer to a host `Span`.

        Args:
            dst: Host span to copy to.

        Raises:
            If the operation fails.
        """
        comptime __guard not use_legacy_backend() else:
            self._impl.enqueue_copy_to(dst)
            return

        self.context().enqueue_copy(dst, self)

    def enqueue_fill(self, val: Scalar[Self.dtype]) raises:
        """Enqueues an operation to fill this buffer with a specified value.

        This method schedules a memory set operation that fills the entire
        buffer with the specified value. The operation is asynchronous and will
        be executed in the stream associated with this buffer's context.

        Args:
            val: The value to fill the buffer with.

        Raises:
            If the operation fails.
        """
        comptime __guard not use_legacy_backend() else:
            self._impl.enqueue_fill(val)
            return

        self.context().enqueue_memset(self, val)

    comptime _Handle = (
        _LegacyDeviceBufferPtr[
            mut=True
        ] if use_legacy_backend() else _HALDeviceBufferPtr[mut=True]
    )

    @doc_hidden
    def __init__(
        out self,
        handle: Self._Handle,
        device_ptr: Self._DevicePtr,
    ):
        """Wraps a native buffer handle and its device pointer.

        Args:
            handle: The native handle of the buffer.
            device_ptr: The device pointer the buffer holds.
        """
        comptime assert (
            use_legacy_backend()
        ), "this constructor is only available on the legacy DeviceContext"
        self._impl = Self._Legacy(handle, device_ptr)

    def device_ptr(
        ref self,
    ) raises -> DevicePointer[Self.dtype, origin_of(self)]:
        """Returns a `DevicePointer` referencing the start of this buffer.

        The returned `DevicePointer` is a non-owning borrow of this
        `DeviceBuffer` and must not outlive it. A function that returns a
        `DevicePointer` must also return (or otherwise keep alive) the backing
        `DeviceBuffer`; returning a pointer into a buffer created locally within
        the function is a borrow-check error.

        Returns:
            A `DevicePointer` referencing offset 0 of this buffer.

        Raises:
            If this buffer has size 0.
        """
        comptime __guard not use_legacy_backend() else:
            return rebind_var[DevicePointer[Self.dtype, origin_of(self)]](
                self._impl.device_ptr()
            )

        comptime assert not is_gpu(), "DeviceBuffer is not supported on GPUs"
        return DevicePointer[Self.dtype, origin_of(self)](self)

    def take_handle(deinit self) -> Self._Handle:
        """Transfers the owning native handle out without releasing it.

        Unlike `take_ptr()`, the buffer's destructor does not run, so the
        caller adopts the reference this buffer held.

        Returns:
            The native handle of the buffer.
        """
        comptime __guard not use_legacy_backend() else:
            return self._impl^.take_handle()

        var ptr = self._impl._device_ptr
        var byte_size = Int(self._impl._inner[]._buffer.byte_size)
        var box = alloc[DeviceBuffer[Self.dtype]]({length = 1}).unsafe_leak()
        box.unsafe_write(self^)
        return external_call[
            "AsyncRT_DeviceBuffer_adoptMojoBox", _HALDeviceBufferPtr[mut=True]
        ](box, ptr, byte_size)

    def _tensor_map_encode_tiled(
        self,
        tensor_map: MutOpaquePointer[_],
        data_type: Int32,
        rank: Int32,
        global_dim: Pointer[mut=False, Int64, _],
        global_strides: Pointer[mut=False, Int64, _],
        box_dim: Pointer[mut=False, Int32, _],
        element_strides: Pointer[mut=False, Int32, _],
        interleave: Int32,
        swizzle: Int32,
        l2_promotion: Int32,
        oob_fill: Int32,
    ) raises:
        """Encodes a tiled TMA descriptor for this buffer via the HAL plugin.

        Used by `max.gpu.host._legacy._tensormap.create_tensormap`.
        """
        comptime __guard not use_legacy_backend() else:
            self._impl._tensor_map_encode_tiled(
                tensor_map,
                data_type,
                rank,
                global_dim,
                global_strides,
                box_dim,
                element_strides,
                interleave,
                swizzle,
                l2_promotion,
                oob_fill,
            )
            return

        # Same-width reinterpret of the callers' signed arrays to the C ABI's
        # unsigned element types.
        self._impl._ctx._impl._context[].tensor_map_encode_tiled(
            tensor_map,
            data_type,
            rank,
            self._impl._inner[]._device_addr,
            global_dim.unsafe_bitcast[UInt64](),
            global_strides.unsafe_bitcast[UInt64](),
            box_dim.unsafe_bitcast[UInt32](),
            element_strides.unsafe_bitcast[UInt32](),
            interleave,
            swizzle,
            l2_promotion,
            oob_fill,
        )

    def _tensor_map_encode_im2col(
        self,
        tensor_map: MutOpaquePointer[_],
        data_type: Int32,
        rank: Int32,
        global_dim: Pointer[mut=False, Int64, _],
        global_strides: Pointer[mut=False, Int64, _],
        pixel_box_lower_corner: Pointer[mut=False, Int32, _],
        pixel_box_upper_corner: Pointer[mut=False, Int32, _],
        channels_per_pixel: Int32,
        pixels_per_column: Int32,
        element_strides: Pointer[mut=False, Int32, _],
        interleave: Int32,
        swizzle: Int32,
        l2_promotion: Int32,
        oob_fill: Int32,
    ) raises:
        """Encodes an im2col TMA descriptor for this buffer via the HAL plugin.

        Used by `max.gpu.host._legacy._tensormap.create_tensormap_im2col`.
        """
        comptime __guard not use_legacy_backend() else:
            self._impl._tensor_map_encode_im2col(
                tensor_map,
                data_type,
                rank,
                global_dim,
                global_strides,
                pixel_box_lower_corner,
                pixel_box_upper_corner,
                channels_per_pixel,
                pixels_per_column,
                element_strides,
                interleave,
                swizzle,
                l2_promotion,
                oob_fill,
            )
            return

        self._impl._ctx._impl._context[].tensor_map_encode_im2col(
            tensor_map,
            data_type,
            rank,
            self._impl._inner[]._device_addr,
            global_dim.unsafe_bitcast[UInt64](),
            global_strides.unsafe_bitcast[UInt64](),
            pixel_box_lower_corner,
            pixel_box_upper_corner,
            channels_per_pixel,
            pixels_per_column,
            element_strides.unsafe_bitcast[UInt32](),
            interleave,
            swizzle,
            l2_promotion,
            oob_fill,
        )

    def map_to_host(
        self,
        out mapped_buffer: _HostMappedBuffer[Self.dtype],
    ) raises:
        """Maps this device buffer to host memory for CPU access.

        This method creates a host-accessible view of the device buffer's
        contents. The mapping operation may involve copying data from device to
        host memory.

        Returns:
            A host-mapped buffer that provides CPU access to the device buffer's
            contents inside a with-statement.

        Raises:
            If there's an error during buffer creation or data transfer.

        Notes:

        Values modified inside the `with` statement are updated on the
        device when the `with` statement exits.

        Example:

        ```mojo
        from max.gpu.host import DeviceContext

        var ctx = DeviceContext()
        var length = 1024
        var in_dev = ctx.enqueue_create_buffer[.float32](length)
        var out_dev = ctx.enqueue_create_buffer[.float32](length)

        # Initialize the input and output with known values.
        with in_dev.map_to_host() as in_host, out_dev.map_to_host() as out_host:
            for i in range(length):
                in_host[i] = Float32(i)
                out_host[i] = 255
        ```
        """
        comptime assert not is_gpu(), "DeviceBuffer is not supported on GPUs"
        mapped_buffer = _HostMappedBuffer[Self.dtype](self.context(), self)

    def write_to(self, mut writer: Some[Writer]):
        """Writes a string representation of this buffer to the provided writer.

        Only the legacy backend prints buffer contents.

        Args:
            writer: The writer to output the formatted string to.
        """
        comptime assert (
            use_legacy_backend()
        ), "write_to is only available on the legacy DeviceContext"
        self._impl.write_to(writer)


struct _HostMappedBuffer[dtype: DType]:
    """Copies a device buffer to the host on entry and back on exit.

    Written against the wrapper types, so both backends share it.
    """

    var _ctx: DeviceContext[]
    var _dev_buf: DeviceBuffer[Self.dtype]
    var _cpu_buf: HostBuffer[Self.dtype]

    def __init__(
        out self,
        ctx: DeviceContext[],
        buf: DeviceBuffer[Self.dtype],
    ) raises:
        var cpu_buf = ctx.enqueue_create_host_buffer[Self.dtype](len(buf))
        self._ctx = ctx
        self._dev_buf = buf
        self._cpu_buf = cpu_buf

    def __deinit__(deinit self):
        pass

    def __enter__(mut self) raises -> HostBuffer[Self.dtype]:
        self._dev_buf.enqueue_copy_to(self._cpu_buf)
        self._ctx.synchronize()
        return self._cpu_buf

    def __exit__(mut self) raises:
        self._ctx.synchronize()
        self._cpu_buf.enqueue_copy_to(self._dev_buf)
        self._ctx.synchronize()
