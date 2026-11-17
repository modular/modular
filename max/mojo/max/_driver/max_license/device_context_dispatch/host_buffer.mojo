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
"""Dispatching `HostBuffer`."""

from std.format import Writer
from std.memory import ArcPointer, Pointer
from std.sys import is_gpu, size_of

from max._driver.max_license.hal import Buffer, Context, get_device_spec
from max.gpu.host._legacy.device_context import LegacyHostBuffer
from .device_context import use_legacy_backend
from max.gpu.host.info import default_accelerator
from .host_types import Device
from max._driver.max_license.device_context_dispatch.device_context import (
    DeviceContext,
)
from max._driver.max_license.device_context_dispatch.device_buffer import (
    DeviceBuffer,
)


@fieldwise_init
struct _HostBufferInner(Movable):
    """Refcountable wrapper around a pinned-host HAL `Buffer`.

    Owns the pinned `Buffer` and the parent `Context` so destruction can
    call `Context.free_host_pinned`. Caches the host pointer up front so
    `unsafe_ptr` / `__getitem__` / `__setitem__` are non-raising.
    """

    var _buffer: Buffer[get_device_spec[0]()]
    var _context: ArcPointer[Context[get_device_spec[0]()]]
    var _host_ptr: Pointer[UInt8, MutUntrackedOrigin]
    # For sub-buffer views: keeps the parent allocation alive for the view's
    # lifetime. A view's `_buffer` is a non-owning wrapper over the parent's
    # pinned memory, so it is released with `free_sync`, not freed.
    var _parent: List[ArcPointer[_HostBufferInner]]

    def __deinit__(deinit self):
        if self._parent:
            try:
                self._context[].free_sync(self._buffer^)
            except e:
                print("warning: free_sync failed:", e)
            # Release the parent only after the view's wrapper is freed.
            _ = self._parent^
            return
        try:
            self._context[].free_host_pinned(self._buffer^)
        except e:
            print("warning: free_host_pinned failed:", e)


@fieldwise_init
struct _HALHostBuffer[dtype: DType](ImplicitlyCopyable, Movable):
    """The state of a `HostBuffer` on the HAL backend."""

    var _ctx: DeviceContext.Unknown
    var _inner: ArcPointer[_HostBufferInner]


struct HostBuffer[dtype: DType](ImplicitlyCopyable, Sized, Writable):
    """Represents a block of host-resident storage. For GPU devices, a host
    buffer is allocated in the host's global memory.

    To allocate a `HostBuffer`, use one of the methods provided by
    `DeviceContext`, such as
    [`enqueue_create_host_buffer()`](/api/mojo/max/gpu/host/device_context/DeviceContext/#enqueue_create_host_buffer).

    Parameters:
        dtype: Data type to be stored in the buffer.
    """

    # ===-------------------------------------------------------------------===#
    # Backend selection
    # ===-------------------------------------------------------------------===#

    comptime _Legacy = LegacyHostBuffer[Self.dtype]
    comptime _HAL = _HALHostBuffer[Self.dtype]
    comptime _Impl = Self._Legacy if use_legacy_backend() else Self._HAL

    # ===-------------------------------------------------------------------===#
    # Fields
    # ===-------------------------------------------------------------------===#

    var _impl: Self._Impl

    # ===-------------------------------------------------------------------===#
    # Life cycle methods
    # ===-------------------------------------------------------------------===#

    @doc_hidden
    def __init__(out self, ctx: DeviceContext, size: Int) raises:
        """This init takes in a constructed `DeviceContext` and schedules an
        owned buffer allocation using the stream in the device context.
        """
        comptime __guard case False = use_legacy_backend() else:
            comptime assert use_legacy_backend()  # TODO(MOCO-5393)
            self._impl = Self._Legacy(
                ctx._impl,
                size=size,
            )
            return
        comptime assert not use_legacy_backend()  # TODO(MOCO-5393)

        var byte_size = UInt64(size * size_of[Self.dtype]())
        var buffer = ctx._impl._context[].alloc_host_pinned(byte_size)
        var addr = UInt64(0)
        var host_ptr: Pointer[UInt8, MutUntrackedOrigin]
        if byte_size > 0:
            host_ptr = ctx._impl._context[].memory_get_host_address[
                MutUntrackedOrigin
            ](buffer)
        else:
            host_ptr = Pointer[UInt8, MutUntrackedOrigin](
                unsafe_from_address=Int(addr)
            )
        self._impl = Self._HAL(
            ctx.into_unknown_target(),
            ArcPointer(
                _HostBufferInner(buffer^, ctx._impl._context, host_ptr, [])
            ),
        )

    @doc_hidden
    def __init__(
        out self,
        ctx: DeviceContext,
        host_ptr: Self._HostPtr,
        size: Int,
        *,
        owning: Bool,
    ):
        """Constructs a `HostBuffer` that wraps an externally-managed pointer.

        Args:
            ctx: The device context the pointer belongs to.
            host_ptr: The host pointer to wrap.
            size: Number of elements addressable through the pointer.
            owning: Whether releasing this buffer should free the pointer.
        """
        comptime assert use_legacy_backend(), (
            "wrapping an external host pointer is only available on the"
            " legacy DeviceContext"
        )
        self._impl = Self._Legacy(
            ctx._impl,
            host_ptr,
            size,
            owning=owning,
        )

    def __init__(out self, *, var _impl: Self._Impl):
        """Wraps an already-constructed backend buffer.

        Args:
            _impl: The backend buffer to take ownership of.
        """
        self._impl = _impl^

    # ===-------------------------------------------------------------------===#
    # Aliases
    # ===-------------------------------------------------------------------===#

    comptime _HostPtr = Pointer[Scalar[Self.dtype], MutUntrackedOrigin]

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

    # ===-------------------------------------------------------------------===#
    # Operator dunders
    # ===-------------------------------------------------------------------===#

    @inline(.always)
    def __getitem__(self, idx: Int) -> Scalar[Self.dtype]:
        """Retrieves the element at the specified index from the host buffer.

        This operator allows direct access to individual elements in the host
        buffer using array indexing syntax.

        Args:
            idx: The index of the element to retrieve.

        Returns:
            The scalar value at the specified index.
        """
        comptime __guard case False = use_legacy_backend() else:
            comptime assert use_legacy_backend()  # TODO(MOCO-5393)
            return self._impl[idx]
        comptime assert not use_legacy_backend()  # TODO(MOCO-5393)

        return self.unsafe_ptr()[unsafe_offset=idx]

    @inline(.always)
    def __setitem__(self, idx: Int, val: Scalar[Self.dtype]):
        """Sets the element at the specified index in the host buffer.

        This operator allows direct modification of individual elements in the
        host buffer using array indexing syntax.

        Args:
            idx: The index of the element to set.
            val: The value to store at the specified index.
        """
        comptime __guard case False = use_legacy_backend() else:
            comptime assert use_legacy_backend()  # TODO(MOCO-5393)
            self._impl[idx] = val
            return
        comptime assert not use_legacy_backend()  # TODO(MOCO-5393)

        self.unsafe_ptr()[unsafe_offset=idx] = val

    def __len__(self) -> Int:
        """Returns the number of elements in this buffer.

        This method calculates the number of elements by dividing the total
        byte size of the buffer by the size of each element.

        Returns:
            The number of elements in the buffer.
        """
        comptime __guard case False = use_legacy_backend() else:
            comptime assert use_legacy_backend()  # TODO(MOCO-5393)
            return len(self._impl)
        comptime assert not use_legacy_backend()  # TODO(MOCO-5393)

        return (
            Int(self._impl._inner[]._buffer.byte_size) // size_of[Self.dtype]()
        )

    # ===-------------------------------------------------------------------===#
    # Methods
    # ===-------------------------------------------------------------------===#

    def unsafe_ptr(self) -> Self._HostPtr:
        """Returns the raw host pointer without transferring ownership.

        This method provides direct access to the underlying host pointer for
        advanced use cases. The buffer retains ownership of the pointer.

        Returns:
            The raw host pointer owned by this buffer.
        """
        comptime __guard case False = use_legacy_backend() else:
            comptime assert use_legacy_backend()  # TODO(MOCO-5393)
            return rebind[Self._HostPtr](self._impl.unsafe_ptr())
        comptime assert not use_legacy_backend()  # TODO(MOCO-5393)

        return self._impl._inner[]._host_ptr.unsafe_bitcast[
            Scalar[Self.dtype]
        ]()

    @__unsafe_nested_origins_read_only
    def as_span[
        origin: Origin, //
    ](ref[origin] self) -> Span[
        Scalar[Self.dtype], origin_of(self)._get_owned_interior["buffer"]
    ]:
        """Returns a `Span` pointing to the underlying memory of the `HostBuffer`.

        Parameters:
            origin: The origin of the buffer reference.

        Returns:
            A `Span` over the buffer's memory. The span carries an interior
            origin derived from `self`, so any subsequent mutation of the
            `HostBuffer` invalidates it at compile time.
        """
        # Built from the buffer's own pointer rather than forwarded: the
        # interior origin belongs to this wrapper, so neither backend's span
        # carries it.
        return {
            unsafe_ptr = Pointer[
                Scalar[Self.dtype],
                origin_of(self)._get_owned_interior["buffer"],
            ](
                to=self.unsafe_ptr()._get_ref_with_unsafe_interior_origin[
                    "buffer", origin_of(self)
                ]()
            ),
            length = len(self),
        }

    def context(self) raises -> DeviceContext[]:
        """Returns the device context associated with this buffer.

        Returns:
            The device context that owns this buffer.

        Raises:
            If the query fails.
        """
        comptime __guard case False = use_legacy_backend() else:
            comptime assert use_legacy_backend()  # TODO(MOCO-5393)
            return DeviceContext[](_impl=self._impl.context())
        comptime assert not use_legacy_backend()  # TODO(MOCO-5393)

        return rebind_var[DeviceContext[]](self._impl._ctx)

    def create_sub_buffer[
        view_type: DType
    ](self, offset: Int, size: Int) raises -> HostBuffer[view_type]:
        """Creates a sub-buffer view of this buffer with a different element dtype.

        Parameters:
            view_type: Element dtype of the sub-buffer.

        Args:
            offset: Offset in bytes from the start of this buffer.
            size: Size of the sub-buffer in elements of `view_type`.

        Returns:
            A host buffer viewing the requested range.

        Raises:
            If the range is invalid.
        """
        comptime __guard case False = use_legacy_backend() else:
            comptime assert use_legacy_backend()  # TODO(MOCO-5393)
            return HostBuffer[view_type](
                _impl=self._impl.create_sub_buffer[view_type](offset, size)
            )
        comptime assert not use_legacy_backend()  # TODO(MOCO-5393)

        comptime assert not is_gpu(), "HostBuffer is not supported on GPUs"
        comptime elem_size = size_of[view_type]()
        # Check in elements so a huge `offset` or `size` can't wrap the byte
        # arithmetic back into range.
        var max_elems = Int(self._impl._inner[]._buffer.byte_size) // elem_size
        if (
            offset < 0
            or size < 0
            or size > max_elems
            or offset > max_elems - size
        ):
            raise Error("create_sub_buffer: view is out of range")
        var byte_offset = offset * elem_size
        var byte_size = size * elem_size
        var host_ptr = self._impl._inner[]._host_ptr.unsafe_offset(byte_offset)
        var addr = UInt64(Int(host_ptr))
        # The view's buffer is a non-owning wrapper over the parent's pinned
        # memory. The plugin rejects wrapping a null address, so an empty
        # parent's view uses a zero-byte allocation instead, as `DeviceBuffer`
        # does; both are released with `free_sync`.
        var context = self._impl._inner[]._context
        var buffer = context[].alloc_sync(
            0
        ) if addr == 0 else context[].wrap_memory(addr, UInt64(byte_size))
        return HostBuffer[view_type](
            _impl=_HALHostBuffer[view_type](
                self._impl._ctx,
                ArcPointer(
                    _HostBufferInner(
                        buffer^, context, host_ptr, [self._impl._inner]
                    )
                ),
            )
        )

    def enqueue_copy_to(self, dst: HostBuffer[Self.dtype]) raises:
        """Enqueues an asynchronous copy to another host buffer.

        Args:
            dst: Host buffer to copy to.

        Raises:
            If the operation fails.
        """
        comptime __guard case False = use_legacy_backend() else:
            comptime assert use_legacy_backend()  # TODO(MOCO-5393)
            self._impl.enqueue_copy_to(dst._impl)
            return
        comptime assert not use_legacy_backend()  # TODO(MOCO-5393)

        # The HAL host buffer copies through spans; it has no direct
        # host-to-host overload.
        self.context().enqueue_copy(dst, self.as_span())

    def enqueue_copy_to(self, dst: DeviceBuffer[Self.dtype]) raises:
        """Enqueues an asynchronous copy from this buffer to a device buffer.

        This method schedules a memory copy operation from this buffer to the
        destination device buffer. The operation is asynchronous and will be
        executed in the stream associated with this buffer's context.

        Args:
            dst: Device buffer to copy to.

        Raises:
            If the operation fails.
        """
        comptime __guard case False = use_legacy_backend() else:
            comptime assert use_legacy_backend()  # TODO(MOCO-5393)
            self._impl.enqueue_copy_to(dst._impl)
            return
        comptime assert not use_legacy_backend()  # TODO(MOCO-5393)

        self.context().enqueue_copy(dst, self)

    def enqueue_copy_to(
        self, dst_ptr: Pointer[mut=True, Scalar[Self.dtype], _]
    ) raises:
        """Enqueues an asynchronous copy to host memory.

        Args:
            dst_ptr: Pointer to the destination host memory.

        Raises:
            If the operation fails.
        """
        comptime assert use_legacy_backend(), (
            "enqueue_copy_to a raw pointer is only available on the legacy"
            " DeviceContext"
        )
        self._impl.enqueue_copy_to(dst_ptr)

    def enqueue_copy_from(self, src: HostBuffer[Self.dtype]) raises:
        """Enqueues an asynchronous copy from another host buffer.

        Args:
            src: Host buffer to copy from.

        Raises:
            If the operation fails.
        """
        comptime __guard case False = use_legacy_backend() else:
            comptime assert use_legacy_backend()  # TODO(MOCO-5393)
            self._impl.enqueue_copy_from(src._impl)
            return
        comptime assert not use_legacy_backend()  # TODO(MOCO-5393)

        # The HAL host buffer copies through spans; it has no direct
        # host-to-host overload.
        self.context().enqueue_copy(self, src.as_span())

    def enqueue_copy_from(self, src: DeviceBuffer[Self.dtype]) raises:
        """Enqueues an asynchronous copy from a device buffer to this buffer.

        This method schedules a memory copy operation from the source device
        buffer to this buffer. The operation is asynchronous and will be
        executed in the stream associated with this buffer's context.

        Args:
            src: Device buffer to copy from.

        Raises:
            If the operation fails.
        """
        comptime __guard case False = use_legacy_backend() else:
            comptime assert use_legacy_backend()  # TODO(MOCO-5393)
            self._impl.enqueue_copy_from(src._impl)
            return
        comptime assert not use_legacy_backend()  # TODO(MOCO-5393)

        self.context().enqueue_copy(self, src)

    def enqueue_copy_from(
        self, src_ptr: Pointer[mut=False, Scalar[Self.dtype], _]
    ) raises:
        """Enqueues an asynchronous copy from host memory.

        Args:
            src_ptr: Pointer to the source host memory.

        Raises:
            If the operation fails.
        """
        comptime assert use_legacy_backend(), (
            "enqueue_copy_from a raw pointer is only available on the legacy"
            " DeviceContext"
        )
        self._impl.enqueue_copy_from(src_ptr)

    def enqueue_copy_from(
        self, src: Span[mut=False, Scalar[Self.dtype], _]
    ) raises:
        """Enqueues an asynchronous copy from a host span.

        Args:
            src: Host span to copy from.

        Raises:
            If the operation fails.
        """
        comptime __guard case False = use_legacy_backend() else:
            comptime assert use_legacy_backend()  # TODO(MOCO-5393)
            self._impl.enqueue_copy_from(src)
            return
        comptime assert not use_legacy_backend()  # TODO(MOCO-5393)

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
        comptime __guard case False = use_legacy_backend() else:
            comptime assert use_legacy_backend()  # TODO(MOCO-5393)
            self._impl.enqueue_copy_to(dst)
            return
        comptime assert not use_legacy_backend()  # TODO(MOCO-5393)

        self.context().enqueue_copy(dst, self)

    def enqueue_fill(self, val: Scalar[Self.dtype]) raises:
        """Enqueues an operation to fill this buffer with a specified value.

        Args:
            val: The value to fill the buffer with.

        Raises:
            If the operation fails.
        """
        comptime assert (
            use_legacy_backend()
        ), "enqueue_fill is only available on the legacy DeviceContext"
        self._impl.enqueue_fill(val)

    def take_ptr(deinit self) -> Self._HostPtr:
        """Takes ownership of the host pointer from this buffer.

        Returns:
            The raw host pointer that was owned by this buffer.
        """
        comptime assert (
            use_legacy_backend()
        ), "take_ptr is only available on the legacy DeviceContext"
        return rebind[Self._HostPtr](self._impl.take_ptr())

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
