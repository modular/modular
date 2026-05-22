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

# Implementation of DeviceContext backed by the HAL

from std.memory import ArcPointer, UnsafePointer
from std.os import getenv
from std.sys import size_of
from std.sys._hal import (
    Buffer,
    Context,
    Device,
    Driver,
    Event,
    Stream,
    get_device_spec,
)
from std.sys._hal.event import EVENT_FLAG_CPU_VISIBLE

from .info import GPUInfo


struct DeviceContext(ImplicitlyCopyable, RegisterPassable):
    """Represents a single stream of execution on a particular accelerator
    (GPU).

    A `DeviceContext` serves as the low-level interface to the
    accelerator inside a MAX [custom operation](/max/develop/custom-ops/) and provides
    methods for allocating buffers on the device, copying data between host and
    device, and for compiling and running functions (also known as kernels) on
    the device.

    The device context can be used as a
    [context manager](/docs/manual/errors/#use-a-context-manager). For example:

    ```mojo
    from std.gpu.host import DeviceContext
    from std.gpu import thread_idx

    def kernel():
        print("hello from thread:", thread_idx.x, thread_idx.y, thread_idx.z)

    with DeviceContext() as ctx:
        ctx.enqueue_function[kernel, kernel](grid_dim=1, block_dim=(2, 2, 2))
        ctx.synchronize()
    ```

    A custom operation receives an opaque `DeviceContextPtr`, which provides
    a `get_device_context()` method to retrieve the device context:

    ```text
    from std.runtime.asyncrt import DeviceContextPtr
    from compiler import register

    @register("custom_op")
    struct CustomOp:
        @staticmethod
        def execute(ctx_ptr: DeviceContextPtr) raises:
            var ctx = ctx_ptr.get_device_context()
            ctx.enqueue_function[kernel, kernel](grid_dim=1, block_dim=(2, 2, 2))
            ctx.synchronize()
    ```
    """

    comptime device_spec = get_device_spec[0]()

    comptime default_device_info = GPUInfo.from_target[
        Self.device_spec.target.value
    ]()

    var _driver: ArcPointer[Driver]
    var _device: ArcPointer[Device[Self.device_spec]]
    var _context: ArcPointer[Context[Self.device_spec]]
    var _stream: ArcPointer[Stream[Self.device_spec]]

    @always_inline
    def __init__(
        out self,
        device_id: Int = 0,
        *,
        var api: String = String(Self.default_device_info.api),
    ) raises:
        """Constructs a `DeviceContext` for the specified device.

        This initializer creates a new device context for the specified accelerator device.
        The device context provides an interface for interacting with the GPU, including
        memory allocation, data transfer, and kernel execution.

        Args:
            device_id: ID of the accelerator device. If not specified, uses
                the default accelerator (device 0).
            api: Requested device API (for example, "cuda" or "hip"). Defaults
                to the device API specified by current target accelerator.

        Raises:
            If device initialization fails or the specified device is not available.

        Example:

        ```mojo
        from std.gpu.host import DeviceContext

        # Create a context for the default GPU
        var ctx = DeviceContext()

        # Create a context for a specific GPU (device 1)
        var ctx2 = DeviceContext(1)
        ```
        """

        var plugin_spec = getenv("MODULAR_DRIVER_PLUGINS")
        if not plugin_spec:
            raise Error("MODULAR_DRIVER_PLUGINS not set")

        self._driver = Driver.create(plugin_spec)

        # Validate that the loaded plugin matches the requested api.
        var driver_name = self._driver[].get_name()
        if String(driver_name).lower() != String(api).lower():
            raise Error(
                String(
                    t"Requested API {api} not supported by driver {driver_name}"
                )
            )

        # TODO: DRIV-163 - Use real device_id
        self._device = self._driver[].get_device[0]()
        self._context = self._device[].get_context()
        self._stream = self._context[].create_stream()

    def __enter__(var self) -> Self:
        """Enables the use of `DeviceContext` in a `with` statement context manager.

        Returns:
            The `DeviceContext` instance to be used within the context manager block.
        """
        return self^

    def synchronize(self) raises:
        """Blocks until all asynchronous calls on the stream associated with
        this device context have completed.

        Raises:
            If the operation fails.
        """
        self._stream[].synchronize()

    def id(self) -> Int64:
        """Returns the ID associated with this device.

        Returns:
            The unique device ID as an `Int64`.
        """
        return self._device[].id

    def stream(self) -> DeviceStream:
        return DeviceStream(self)

    def create_stream(self) raises -> DeviceStream:
        """Creates a new stream associated with the given device context.

        Returns:
            The newly created device stream.

        Raises:
            If stream creation fails.
        """
        var hal_stream = self._context[].create_stream()
        return DeviceStream(self, hal_stream^)

    def create_event(self) -> DeviceEvent:
        """Creates a new event for synchronization between streams.

        Returns:
            A DeviceEvent that can be used for synchronization.

        Example:

        ```mojo
        from std.gpu.host import DeviceContext

        var ctx = DeviceContext()

        var default_stream = ctx.stream()
        var new_stream = ctx.create_stream()

        # Create an event
        var event = ctx.create_event()

        # Wait for the event in new_stream
        new_stream.enqueue_wait_for(event)

        # new_stream can continue
        default_stream.record_event(event)
        default_stream.synchronize()
        ```
        """

        # The HAL uses logical events where creation and recording happen in
        # one step. To work around this, initially create a DeviceEvent without
        # a backing HAL event.
        return DeviceEvent._create_unrecorded(self)

    # ===-------------------------------------------------------------------===#
    # Buffer operations
    # ===-------------------------------------------------------------------===#

    def enqueue_create_buffer[
        dtype: DType
    ](self, size: Int) raises -> DeviceBuffer[dtype]:
        """Enqueues a buffer creation using the `DeviceBuffer` constructor.

        For GPU devices, the space is allocated in the device's global memory.

        Parameters:
            dtype: The data type to be stored in the allocated memory.

        Args:
            size: The number of elements of `type` to allocate memory for.

        Returns:
            The allocated buffer.

        Raises:
            If the operation fails.
        """

        # NOTE: The HAL supports doing this asynchronously, but to match
        # existing DeviceContext semantics we create a buffer immediately.
        return DeviceBuffer[dtype](self, size)

    def create_buffer_sync[
        dtype: DType
    ](self, size: Int) raises -> DeviceBuffer[dtype]:
        """Creates a buffer synchronously using the `DeviceBuffer` constructor.

        Parameters:
            dtype: The data type to be stored in the allocated memory.

        Args:
            size: The number of elements of `type` to allocate memory for.

        Returns:
            The allocated buffer.

        Raises:
            If the operation fails.
        """
        return DeviceBuffer[dtype](self, size)

    def enqueue_create_host_buffer[
        dtype: DType
    ](self, size: Int) raises -> HostBuffer[dtype]:
        """Enqueues the creation of a HostBuffer.

        This function allocates memory on the host that is accessible by the device.
        The memory is page-locked (pinned) for efficient data transfer between host and device.

        Pinned memory is guaranteed to remain resident in the host's RAM, not be
        paged/swapped out to disk. Memory allocated normally (for example, using
        [`alloc()`](/docs/std/memory/unsafe_pointer/alloc/))
        is pageable—individual pages of memory can be moved to secondary storage
        (disk/SSD) when main memory fills up.

        Using pinned memory allows devices to make fast transfers
        between host memory and device memory, because they can use direct
        memory access (DMA) to transfer data without relying on the CPU.

        Allocating too much pinned memory can cause performance issues, since it
        reduces the amount of memory available for other processes.

        Parameters:
            dtype: The data type to be stored in the allocated memory.

        Args:
            size: The number of elements of `type` to allocate memory for.

        Returns:
            A `HostBuffer` object that wraps the allocated host memory.

        Raises:
            If memory allocation fails or if the device context is invalid.

        Example:

        ```mojo
        from std.gpu.host import DeviceContext

        with DeviceContext() as ctx:
            # Allocate host memory accessible by the device
            var host_buffer = ctx.enqueue_create_host_buffer[DType.float32](1024)

            # Use the host buffer for device operations
            # ...
        ```
        """
        return HostBuffer[dtype](self, size)

    def enqueue_copy[
        dtype: DType
    ](
        self,
        dst_buf: DeviceBuffer[dtype],
        src_ptr: UnsafePointer[Scalar[dtype], _],
    ) raises:
        """Enqueues an async copy from the host to the provided device
        buffer. The number of bytes copied is determined by the size of the
        device buffer.

        Parameters:
            dtype: Type of the data being copied.

        Args:
            dst_buf: Device buffer to copy to.
            src_ptr: Host pointer to copy from.

        Raises:
            If the operation fails.
        """
        self._stream[].copy_to_device(
            dst_buf._inner[]._buffer,
            src_ptr.bitcast[UInt8](),
            dst_buf._inner[]._buffer.byte_size,
        )

    def enqueue_copy[
        dtype: DType
    ](
        self,
        dst_ptr: UnsafePointer[mut=True, Scalar[dtype], _],
        src_buf: DeviceBuffer[dtype],
    ) raises:
        """Enqueues an async copy from the device to the host. The
        number of bytes copied is determined by the size of the device buffer.

        Parameters:
            dtype: Type of the data being copied.

        Args:
            dst_ptr: Host pointer to copy to.
            src_buf: Device buffer to copy from.

        Raises:
            If the operation fails.
        """
        self._stream[].copy_from_device(
            dst_ptr.bitcast[UInt8](),
            src_buf._inner[]._buffer,
            src_buf._inner[]._buffer.byte_size,
        )

    def enqueue_copy[
        dtype: DType
    ](self, dst_buf: DeviceBuffer[dtype], src_buf: DeviceBuffer[dtype],) raises:
        """Enqueues an async copy from one device buffer to another. The amount
        of data transferred is determined by the size of the destination buffer.

        Parameters:
            dtype: Type of the data being copied.

        Args:
            dst_buf: Device buffer to copy to.
            src_buf: Device buffer to copy from. Must be at least as large as
                `dst`.

        Raises:
            If the operation fails.
        """
        self._stream[].copy_intra_device(
            dst_buf._inner[]._buffer,
            src_buf._inner[]._buffer,
            dst_buf._inner[]._buffer.byte_size,
        )

    def enqueue_copy[
        dtype: DType
    ](self, dst_buf: DeviceBuffer[dtype], src_buf: HostBuffer[dtype],) raises:
        """Enqueues an async copy from one device buffer to another. The amount
        of data transferred is determined by the size of the destination buffer.

        Parameters:
            dtype: Type of the data being copied.

        Args:
            dst_buf: Device buffer to copy to.
            src_buf: Device buffer to copy from. Must be at least as large as
                `dst`.

        Raises:
            If the operation fails.
        """
        self._stream[].copy_to_device(
            dst_buf._inner[]._buffer,
            src_buf.unsafe_ptr().bitcast[UInt8](),
            dst_buf._inner[]._buffer.byte_size,
        )

    def enqueue_copy[
        dtype: DType
    ](self, dst_buf: HostBuffer[dtype], src_buf: DeviceBuffer[dtype],) raises:
        """Enqueues an async copy from one device buffer to another. The amount
        of data transferred is determined by the size of the destination buffer.

        Parameters:
            dtype: Type of the data being copied.

        Args:
            dst_buf: Device buffer to copy to.
            src_buf: Device buffer to copy from. Must be at least as large as
                `dst`.

        Raises:
            If the operation fails.
        """
        self._stream[].copy_from_device(
            dst_buf.unsafe_ptr().bitcast[UInt8](),
            src_buf._inner[]._buffer,
            src_buf._inner[]._buffer.byte_size,
        )

    def enqueue_copy[
        dtype: DType
    ](
        self,
        dst_buf: HostBuffer[dtype],
        src_ptr: UnsafePointer[Scalar[dtype], _],
    ) raises:
        """Enqueues an async copy from the host to the provided device
        buffer. The number of bytes copied is determined by the size of the
        device buffer.

        Parameters:
            dtype: Type of the data being copied.

        Args:
            dst_buf: Device buffer to copy to.
            src_ptr: Host pointer to copy from.

        Raises:
            If the operation fails.
        """
        self._stream[].copy_to_device(
            dst_buf._inner[]._buffer,
            src_ptr.bitcast[UInt8](),
            dst_buf._inner[]._buffer.byte_size,
        )

    def enqueue_copy[
        dtype: DType
    ](
        self,
        dst_ptr: UnsafePointer[mut=True, Scalar[dtype], _],
        src_buf: HostBuffer[dtype],
    ) raises:
        """Enqueues an async copy from the device to the host. The
        number of bytes copied is determined by the size of the device buffer.

        Parameters:
            dtype: Type of the data being copied.

        Args:
            dst_ptr: Host pointer to copy to.
            src_buf: Device buffer to copy from.

        Raises:
            If the operation fails.
        """
        self._stream[].copy_from_device(
            dst_ptr.bitcast[UInt8](),
            src_buf._inner[]._buffer,
            src_buf._inner[]._buffer.byte_size,
        )

    def enqueue_copy[
        dtype: DType
    ](self, dst_buf: HostBuffer[dtype], src_buf: HostBuffer[dtype],) raises:
        """Enqueues an async copy from one device buffer to another. The amount
        of data transferred is determined by the size of the destination buffer.

        Parameters:
            dtype: Type of the data being copied.

        Args:
            dst_buf: Device buffer to copy to.
            src_buf: Device buffer to copy from. Must be at least as large as
                `dst`.

        Raises:
            If the operation fails.
        """
        self._stream[].copy_intra_device(
            dst_buf._inner[]._buffer,
            src_buf._inner[]._buffer,
            dst_buf._inner[]._buffer.byte_size,
        )

    def enqueue_copy[
        dtype: DType
    ](self, dst_buf: DeviceBuffer[dtype], src: Span[Scalar[dtype], _],) raises:
        """Enqueues an async copy from a host `Span` to a device buffer.

        The number of bytes copied is determined by the size of the device
        buffer. The span must contain at least as many elements as the
        destination buffer; this invariant is checked via `debug_assert`.

        Parameters:
            dtype: Type of the data being copied.

        Args:
            dst_buf: Device buffer to copy to.
            src: Host span to copy from.

        Raises:
            If the operation fails.
        """
        debug_assert(
            len(src) >= len(dst_buf),
            "source span length must be >= destination buffer length",
        )
        self.enqueue_copy(dst_buf, src.unsafe_ptr())

    def enqueue_copy[
        dtype: DType
    ](
        self,
        dst: Span[mut=True, Scalar[dtype], _],
        src_buf: DeviceBuffer[dtype],
    ) raises:
        """Enqueues an async copy from a device buffer to a host `Span`.

        The number of bytes copied is determined by the size of the device
        buffer. The span must contain at least as many elements as the source
        buffer; this invariant is checked via `debug_assert` (debug builds
        only).

        Parameters:
            dtype: Type of the data being copied.

        Args:
            dst: Host span to copy to.
            src_buf: Device buffer to copy from.

        Raises:
            If the operation fails.
        """
        debug_assert(
            len(dst) >= len(src_buf),
            "destination span length must be >= source buffer length",
        )
        self.enqueue_copy(dst.unsafe_ptr(), src_buf)

    def enqueue_copy[
        dtype: DType
    ](self, dst_buf: HostBuffer[dtype], src: Span[Scalar[dtype], _],) raises:
        """Enqueues an async copy from a host `Span` to a pinned host buffer.

        The number of bytes copied is determined by the size of the device
        buffer. The span must contain at least as many elements as the
        destination buffer; this invariant is checked via `debug_assert`.

        Parameters:
            dtype: Type of the data being copied.

        Args:
            dst_buf: Device buffer to copy to.
            src: Host span to copy from.

        Raises:
            If the operation fails.
        """
        debug_assert(
            len(src) >= len(dst_buf),
            "source span length must be >= destination buffer length",
        )
        self.enqueue_copy(dst_buf, src.unsafe_ptr())

    def enqueue_copy[
        dtype: DType
    ](
        self,
        dst: Span[mut=True, Scalar[dtype], _],
        src_buf: HostBuffer[dtype],
    ) raises:
        """Enqueues an async copy from a host buffer to a host `Span`.

        The number of bytes copied is determined by the size of the source
        buffer. The span must contain at least as many elements as the source
        buffer; this invariant is checked via `debug_assert` (debug builds
        only).

        Parameters:
            dtype: Type of the data being copied.

        Args:
            dst: Host span to copy to.
            src_buf: Host buffer to copy from.

        Raises:
            If the operation fails.
        """
        debug_assert(
            len(dst) >= len(src_buf),
            "destination span length must be >= source buffer length",
        )
        self.enqueue_copy(dst.unsafe_ptr(), src_buf)


# ===-----------------------------------------------------------------------===#
# DeviceStream
# ===-----------------------------------------------------------------------===#


struct DeviceStream(ImplicitlyCopyable, Movable):
    """Represents a CUDA/HIP stream for asynchronous GPU operations.

    A DeviceStream provides a queue for GPU operations that can execute concurrently
    with operations in other streams. Operations within a single stream execute in
    the order they are issued, but operations in different streams may execute in
    any relative order or concurrently.

    This abstraction allows for better utilization of GPU resources by enabling
    overlapping of computation and data transfers.

    Example:

    ```mojo
    from std.gpu.host import DeviceContext, DeviceStream
    var ctx = DeviceContext(0)  # Select first GPU
    var stream = DeviceStream(ctx)

    # Launch operations on the stream
    # ...

    # Wait for all operations in the stream to complete
    stream.synchronize()
    ```
    """

    var _ctx: DeviceContext
    var _stream: ArcPointer[Stream[get_device_spec[0]()]]

    @doc_hidden
    def __init__(out self, ctx: DeviceContext):
        """Retrieves the stream associated with the given device context.

        Args:
            ctx: The device context to retrieve the stream from.
        """
        self._ctx = ctx
        self._stream = ctx._stream

    @doc_hidden
    def __init__(
        out self,
        ctx: DeviceContext,
        var hal_stream: ArcPointer[Stream[get_device_spec[0]()]],
    ):
        """Initializes a new DeviceStream with the given stream handle.

        Args:
            ctx: The device context that owns the stream.
            hal_stream: The stream handle to initialize the DeviceStream with.
        """
        self._ctx = ctx
        self._stream = hal_stream^

    def synchronize(self) raises:
        """Blocks the calling CPU thread until all operations in this stream complete.

        This function waits until all previously issued commands in this stream
        have completed execution. It provides a synchronization point between
        host and device code.

        Raises:
            If synchronization fails.

        Example:

        ```mojo
        from std.gpu.host import DeviceContext

        var ctx = DeviceContext()
        var stream = ctx.create_stream()

        # Launch kernel or memory operations on the stream
        # ...

        # Wait for completion
        stream.synchronize()

        # Now it's safe to use results on the host
        ```
        """
        self._stream[].synchronize()

    def enqueue_wait_for(self, event: DeviceEvent) raises:
        """Makes this stream wait for the specified event.

        This function inserts a wait operation into this stream that will
        block all subsequent operations in the stream until the specified
        event has been recorded and completed.

        Args:
            event: The event to wait for.

        Raises:
            If the wait operation fails.
        """
        # If an event hasn't been recorded yet, there's nothing to wait on
        if not event._event[]:
            return
        self._stream[].wait_for_events(event._event[].value())

    def record_event(self, event: DeviceEvent) raises:
        """Records an event in this stream.

        This function records the given event at the current point in this stream.
        All operations in the stream that were enqueued before this call will
        complete before the event is triggered.

        Args:
            event: The event to record.

        Raises:
            If event recording fails.

        Example:

        ```mojo
        from std.gpu.host import DeviceContext

        var ctx = DeviceContext()

        var default_stream = ctx.stream()
        var new_stream = ctx.create_stream()

        # Create event on the context
        var event = ctx.create_event()

        # Wait for the event on the new stream
        new_stream.enqueue_wait_for(event)

        # Stream 2 can continue
        default_stream.record_event(event)
        ```
        """
        # Create and record the backing event for this DeviceEvent. If it was
        # previously recorded and no other DeviceEvents have been constructed
        # by copy from this event, the existing backing event will be released.
        var hal_event = self._stream[].record_event[EVENT_FLAG_CPU_VISIBLE]()
        event._event[] = Optional(hal_event^)


# ===-----------------------------------------------------------------------===#
# DeviceEvent
# ===-----------------------------------------------------------------------===#


struct DeviceEvent(ImplicitlyCopyable, Movable):
    """Represents a GPU event for synchronization between streams.

    A DeviceEvent allows for fine-grained synchronization between different
    GPU streams. Events can be recorded in one stream and waited for in another,
    enabling efficient coordination of asynchronous GPU operations.

    Example:

    ```mojo
    from std.gpu.host import DeviceContext

    var ctx = DeviceContext()

    var default_stream = ctx.stream()
    var new_stream = ctx.create_stream()

    # Create event in default_stream
    var event = ctx.create_event()

    # Wait for the event in new_stream
    new_stream.enqueue_wait_for(event)

    # Stream 2 can continue
    default_stream.record_event(event)
    ```
    """

    # `_event` is an `ArcPointer[Optional[...]]` so that all copies of a
    # DeviceEvent  share the same backing event, which matches the behavior
    # of the existing reference-counted DeviceEvent implemenetation.
    var _ctx: DeviceContext
    var _event: ArcPointer[Optional[Event[EVENT_FLAG_CPU_VISIBLE]]]

    @doc_hidden
    def __init__(out self, ctx: DeviceContext) raises:
        """Creates a new event recorded on the given context's default stream.

        Args:
            ctx: The device context to record the event on.

        Raises:
            If event creation or recording fails.
        """
        var hal_event = ctx._stream[].record_event[EVENT_FLAG_CPU_VISIBLE]()
        self._ctx = ctx
        self._event = ArcPointer(Optional(hal_event^))

    @doc_hidden
    def __init__(
        out self,
        ctx: DeviceContext,
        var event: Optional[Event[EVENT_FLAG_CPU_VISIBLE]],
    ):
        """Initializes a DeviceEvent wrapping a possibly-empty HAL event.

        Args:
            ctx: The device context that owns the event.
            event: The (possibly-unrecorded) HAL event to wrap.
        """
        self._ctx = ctx
        self._event = ArcPointer(event^)

    @doc_hidden
    @staticmethod
    def _create_unrecorded(ctx: DeviceContext) -> DeviceEvent:
        """Returns a `DeviceEvent` with no HAL event allocated yet."""
        return DeviceEvent(ctx, Optional[Event[EVENT_FLAG_CPU_VISIBLE]](None))

    def synchronize(self) raises:
        """Blocks the calling CPU thread until this event completes.

        This function waits until the event has been recorded and all
        operations before the event in the stream have completed.

        Raises:
            If synchronization fails.
        """
        # If an event hasn't been recorded yet, there's nothing to wait on
        if not self._event[]:
            return
        self._event[].value().synchronize()


# ===-----------------------------------------------------------------------===#
# DeviceBuffer
# ===-----------------------------------------------------------------------===#


@fieldwise_init
struct _HALBufferInner(Movable):
    """Owning wrapper around a non-owning HAL Buffers.

    Owns the `Buffer` and holds a reference to the parent `Context` so
    destruction can call `Context.free_sync`.
    """

    var _buffer: Buffer
    var _context: ArcPointer[Context[get_device_spec[0]()]]
    var _device_addr: UInt64

    def __del__(deinit self):
        try:
            self._context[].free_sync(self._buffer^)
        except e:
            print("warning: free_sync failed:", e)


struct DeviceBuffer[dtype: DType](ImplicitlyCopyable, Movable, Sized):
    """Represents a block of device-resident storage. For GPU devices, a device
    buffer is allocated in the device's global memory.

    To allocate a `DeviceBuffer`, use one of the methods provided by
    `DeviceContext`, such as
    [`enqueue_create_buffer()`](/docs/std/gpu/host/device_context/DeviceContext/#enqueue_create_buffer).

    Parameters:
        dtype: Data dtype to be stored in the buffer.
    """

    # Wrap the inner buffer in an ArcPointer so copies of this DeviceBuffer hold
    # a reference to the same underlying buffer.
    var _ctx: DeviceContext
    var _inner: ArcPointer[_HALBufferInner]

    @doc_hidden
    def __init__(out self, ctx: DeviceContext, size: Int) raises:
        """This init takes in a constructed `DeviceContext` and schedules an
        owned buffer allocation using the stream in the device context.
        """
        var byte_size = UInt64(size * size_of[Self.dtype]())
        # Cache the GPU address up front so `unsafe_ptr` is non-raising.
        var buffer = ctx._context[].alloc_sync(byte_size)
        var addr = UInt64(0)
        if byte_size > 0:
            addr = ctx._context[].memory_get_address(buffer)
        self._ctx = ctx
        self._inner = ArcPointer(_HALBufferInner(buffer^, ctx._context, addr))

    @staticmethod
    @doc_hidden
    def empty(context: DeviceContext) raises -> Self:
        return Self(context, 0)

    def context(self) -> DeviceContext:
        """Returns the device context associated with this buffer.

        This method retrieves the device context that owns this buffer and is
        responsible for managing its lifecycle and operations.

        Returns:
            The device context associated with this buffer.
        """
        return self._ctx

    def __len__(self) -> Int:
        """Returns the number of elements in this buffer.

        This method calculates the number of elements by dividing the total byte size
        of the buffer by the size of each element.

        Returns:
            The number of elements in the buffer.
        """
        return Int(self._inner[]._buffer.byte_size) // size_of[Self.dtype]()

    def unsafe_ptr(
        self,
    ) -> UnsafePointer[Scalar[Self.dtype], MutAnyOrigin]:
        """Returns the raw device pointer without transferring ownership.

        This method provides direct access to the underlying device pointer
        for advanced use cases. The buffer retains ownership of the pointer.

        Returns:
            The raw device pointer owned by this buffer.
        """
        return UnsafePointer[Scalar[Self.dtype], MutAnyOrigin](
            unsafe_from_address=Int(self._inner[]._device_addr)
        )

    def enqueue_copy_to(self, dst: HostBuffer[Self.dtype]) raises:
        """Enqueues an asynchronous copy from this buffer to a host buffer.

        This method schedules a memory copy operation from this buffer to the destination
        host buffer. The operation is asynchronous and will be executed in the stream
        associated with this buffer's context.

        Args:
            dst: The destination host buffer to copy data to.

        Raises:
            If the operation fails.
        """
        self._ctx.enqueue_copy(dst, self)

    def enqueue_copy_from(self, src: HostBuffer[Self.dtype]) raises:
        """Enqueues an asynchronous copy from a host buffer to this buffer.

        This method schedules a memory copy operation from the source host buffer
        to this buffer. The operation is asynchronous and will be executed in the stream
        associated with this buffer's context.

        Args:
            src: The source host buffer to copy data from.

        Raises:
            If the operation fails.
        """
        self._ctx.enqueue_copy(self, src)

    def enqueue_copy_to(
        self, dst: Span[mut=True, Scalar[Self.dtype], _]
    ) raises:
        """Enqueues an asynchronous copy from this buffer to a host `Span`.

        Args:
            dst: The destination host span to copy data to.

        Raises:
            If the operation fails.
        """
        self._ctx.enqueue_copy(dst, self)

    def enqueue_copy_from(self, src: Span[Scalar[Self.dtype], _]) raises:
        """Enqueues an asynchronous copy from a host `Span` to this buffer.

        Args:
            src: The source host span to copy data from.

        Raises:
            If the operation fails.
        """
        self._ctx.enqueue_copy(self, src)

    def map_to_host(
        self,
        out mapped_buffer: _HostMappedBuffer[Self.dtype],
    ) raises:
        """Maps this device buffer to host memory for CPU access.

        This method creates a host-accessible view of the device buffer's contents.
        The mapping operation may involve copying data from device to host memory.

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
        from std.gpu.host import DeviceContext

        var ctx = DeviceContext()
        var length = 1024
        var in_dev = ctx.enqueue_create_buffer[DType.float32](length)
        var out_dev = ctx.enqueue_create_buffer[DType.float32](length)

        # Initialize the input and output with known values.
        with in_dev.map_to_host() as in_host, out_dev.map_to_host() as out_host:
            for i in range(length):
                in_host[i] = i
                out_host[i] = 255
        ```
        """
        mapped_buffer = _HostMappedBuffer[Self.dtype](self.context(), self)


# ===-----------------------------------------------------------------------===#
# HostBuffer
# ===-----------------------------------------------------------------------===#


@fieldwise_init
struct _HostBufferInner(Movable):
    """Refcountable wrapper around a pinned-host HAL `Buffer`.

    Owns the pinned `Buffer` and the parent `Context` so destruction can
    call `Context.free_host_pinned`. Caches the host pointer up front so
    `unsafe_ptr` / `__getitem__` / `__setitem__` are non-raising.
    """

    var _buffer: Buffer
    var _context: ArcPointer[Context[get_device_spec[0]()]]
    var _host_ptr: UnsafePointer[UInt8, MutAnyOrigin]

    def __del__(deinit self):
        try:
            self._context[].free_host_pinned(self._buffer^)
        except e:
            print("warning: free_host_pinned failed:", e)


struct HostBuffer[dtype: DType](ImplicitlyCopyable, Movable, Sized):
    """Represents a block of host-resident storage. For GPU devices, a host
    buffer is allocated in the host's global memory.

    To allocate a `HostBuffer`, use one of the methods provided by
    `DeviceContext`, such as
    [`enqueue_create_host_buffer()`](/docs/std/gpu/host/device_context/DeviceContext/#enqueue_create_host_buffer).

    Parameters:
        dtype: Data type to be stored in the buffer.
    """

    var _ctx: DeviceContext
    var _inner: ArcPointer[_HostBufferInner]

    @doc_hidden
    def __init__(out self, ctx: DeviceContext, size: Int) raises:
        """This init takes in a constructed `DeviceContext` and schedules an
        owned buffer allocation using the stream in the device context.
        """
        var byte_size = UInt64(size * size_of[Self.dtype]())
        var buffer = ctx._context[].alloc_host_pinned(byte_size)
        var addr = UInt64(0)
        if byte_size > 0:
            addr = ctx._context[].memory_get_address(buffer)
        var host_ptr = UnsafePointer[UInt8, MutAnyOrigin](
            unsafe_from_address=Int(addr)
        )
        self._ctx = ctx
        self._inner = ArcPointer(
            _HostBufferInner(buffer^, ctx._context, host_ptr)
        )

    def __len__(self) -> Int:
        """Returns the number of elements in this buffer.

        This method calculates the number of elements by dividing the total byte size
        of the buffer by the size of each element.

        Returns:
            The number of elements in the buffer.
        """
        return Int(self._inner[]._buffer.byte_size) // size_of[Self.dtype]()

    def context(self) -> DeviceContext:
        """Returns the device context associated with this buffer.

        This method retrieves the device context that owns this buffer and is
        responsible for managing its lifecycle and operations.

        Returns:
            The device context associated with this buffer.
        """
        return self._ctx

    def unsafe_ptr(
        self,
    ) -> UnsafePointer[Scalar[Self.dtype], MutAnyOrigin]:
        """Returns the raw device pointer without transferring ownership.

        This method provides direct access to the underlying device pointer
        for advanced use cases. The buffer retains ownership of the pointer.

        Returns:
            The raw device pointer owned by this buffer.
        """
        return self._inner[]._host_ptr.bitcast[Scalar[Self.dtype]]()

    def __getitem__(self, idx: Int) -> Scalar[Self.dtype]:
        """Retrieves the element at the specified index from the host buffer.

        This operator allows direct access to individual elements in the host buffer
        using array indexing syntax.

        Args:
            idx: The index of the element to retrieve.

        Returns:
            The scalar value at the specified index.
        """
        return self.unsafe_ptr()[idx]

    def __setitem__(self, idx: Int, val: Scalar[Self.dtype]):
        """Sets the element at the specified index in the host buffer.

        This operator allows direct modification of individual elements in the host buffer
        using array indexing syntax.

        Args:
            idx: The index of the element to modify.
            val: The new value to store at the specified index.
        """
        self.unsafe_ptr()[idx] = val

    def as_span[
        mut: Bool, origin: Origin[mut=mut], //
    ](ref[origin] self) -> Span[Scalar[Self.dtype], origin]:
        """Returns a `Span` pointing to the underlying memory of the `HostBuffer`.

        Parameters:
            mut: Whether the span should be mutable.
            origin: The origin of the buffer reference.

        Returns:
            A `Span` pointing to the underlying memory of the `HostBuffer`.
        """
        # Safety: We are casting the pointer to the mutability and origin of
        # self and the pointer is already mutable.
        return {
            ptr = self.unsafe_ptr()
            .unsafe_mut_cast[mut]()
            .unsafe_origin_cast[origin](),
            length = len(self),
        }

    def enqueue_copy_to(self, dst: DeviceBuffer[Self.dtype]) raises:
        """Enqueues an asynchronous copy from this buffer to a device buffer.

        This method schedules a memory copy operation from this buffer to the destination
        device buffer. The operation is asynchronous and will be executed in the stream
        associated with this buffer's context.

        Args:
            dst: The destination device buffer to copy data to.

        Raises:
            If the operation fails.
        """
        self._ctx.enqueue_copy(dst, self)

    def enqueue_copy_from(self, src: DeviceBuffer[Self.dtype]) raises:
        """Enqueues an asynchronous copy from a device buffer to this buffer.

        This method schedules a memory copy operation from the source device buffer
        to this buffer. The operation is asynchronous and will be executed in the stream
        associated with this buffer's context.

        Args:
            src: The source device buffer to copy data from.

        Raises:
            If the operation fails.
        """
        self._ctx.enqueue_copy(self, src)

    def enqueue_copy_to(
        self, dst: Span[mut=True, Scalar[Self.dtype], _]
    ) raises:
        """Enqueues an asynchronous copy from this buffer to a host `Span`.

        Args:
            dst: The destination host span to copy data to.

        Raises:
            If the operation fails.
        """
        self._ctx.enqueue_copy(dst, self)

    def enqueue_copy_from(self, src: Span[Scalar[Self.dtype], _]) raises:
        """Enqueues an asynchronous copy from a host `Span` to this buffer.

        Args:
            src: The source host span to copy data from.

        Raises:
            If the operation fails.
        """
        self._ctx.enqueue_copy(self, src)


# ===-----------------------------------------------------------------------===#
# _HostMappedBuffer
# ===-----------------------------------------------------------------------===#


struct _HostMappedBuffer[dtype: DType]:
    var _ctx: DeviceContext
    var _dev_buf: DeviceBuffer[Self.dtype]
    var _cpu_buf: HostBuffer[Self.dtype]

    def __init__(
        out self, ctx: DeviceContext, buf: DeviceBuffer[Self.dtype]
    ) raises:
        var cpu_buf = ctx.enqueue_create_host_buffer[Self.dtype](len(buf))
        self._ctx = ctx
        self._dev_buf = buf
        self._cpu_buf = cpu_buf

    def __del__(deinit self):
        pass

    def __enter__(mut self) raises -> HostBuffer[Self.dtype]:
        self._dev_buf.enqueue_copy_to(self._cpu_buf)
        self._ctx.synchronize()
        return self._cpu_buf

    def __exit__(mut self) raises:
        self._ctx.synchronize()
        self._cpu_buf.enqueue_copy_to(self._dev_buf)
        self._ctx.synchronize()
