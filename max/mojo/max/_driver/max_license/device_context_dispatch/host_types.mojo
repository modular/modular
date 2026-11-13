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
"""Backend-dispatching wrappers for the remaining host types.

Streams, events, completion flags, device pointers, device views, context
arrays and multicast buffers. Each forwards to the legacy or the HAL
implementation, and keeps the legacy signature where the two disagree.
"""

from std.builtin.device_passable import DevicePassable
from std.collections.optional import OptionalReg
from std.ffi import external_call
from std.format import Writer
from std.memory import ArcPointer
from std.os import abort
from std.sys.info import CompilationTarget

from max._driver.max_license.device_context_hal.device_context_hal import (
    HALDevice,
    HALDeviceBuffer,
    HALDeviceContext,
    HALDeviceMulticastBuffer,
    HALDeviceStreamBase,
    HALHostBuffer,
    _CompletionFlagBox,
)
from max._driver.max_license.hal import Event
from max._driver.max_license.hal.event import EVENT_FLAG_CPU_VISIBLE
from .device_context import use_legacy_backend
from max.gpu.host._legacy.device_context import (
    LegacyCompletionFlag,
    LegacyDevice,
    LegacyDeviceBuffer,
    LegacyDeviceContext,
    LegacyDeviceEvent,
    LegacyDeviceMulticastBuffer,
    LegacyDeviceStream,
    LegacyHostBuffer,
)
from max.gpu.host._driver_handles import (
    _CompletionFlagPtr as _LegacyCompletionFlagPtr,
)
from max.gpu.host._legacy.constant_memory_mapping import ConstantMemoryMapping
from max.gpu.host._legacy.device_attribute import DeviceAttribute
from max.gpu.host._legacy.dim import Dim
from max.gpu.host._legacy.launch_attribute import LaunchAttribute
from max.gpu.host.info import default_accelerator
from max._driver.max_license.device_context_dispatch.function import (
    DeviceFunction,
    _DispatchEnqueuer,
)
from max._driver.max_license.device_context_dispatch.device_context import (
    DeviceContext,
)
from max._driver.max_license.device_context_dispatch.device_buffer import (
    DeviceBuffer,
)
from max._driver.max_license.device_context_dispatch.host_buffer import (
    HostBuffer,
)


# `DeviceEvent` and `CompletionFlag` are forward-declared below the stream, so
# these aliases keep the dispatch readable.
comptime _Event = DeviceEvent
comptime _Flag = CompletionFlag


struct DeviceStreamBase[target: CompilationTarget = default_accelerator()](
    ImplicitlyCopyable, _DispatchEnqueuer
):
    """A single stream of execution within a `DeviceContext`.

    Parameters:
        target: The accelerator targeted by this stream.
    """

    # ===-------------------------------------------------------------------===#
    # Backend selection
    # ===-------------------------------------------------------------------===#

    comptime _Legacy = LegacyDeviceStream
    comptime _HAL = HALDeviceStreamBase[Self.target]
    comptime _Impl = Self._Legacy if use_legacy_backend() else Self._HAL

    # ===-------------------------------------------------------------------===#
    # Fields
    # ===-------------------------------------------------------------------===#

    var _impl: Self._Impl

    # ===-------------------------------------------------------------------===#
    # Life cycle methods
    # ===-------------------------------------------------------------------===#

    def __init__(out self, ctx: DeviceContext[Self.target]) raises:
        """Creates a stream on the given device context's device.

        Args:
            ctx: The device context that owns the stream.

        Raises:
            If the stream cannot be created.
        """
        comptime if use_legacy_backend():
            self._impl = Self._Legacy(ctx._impl)
        else:
            self._impl = Self._HAL(ctx._impl)

    def __init__(out self, *, var _impl: Self._Impl):
        """Wraps an already-constructed backend stream.

        Args:
            _impl: The backend stream to take ownership of.
        """
        self._impl = _impl^

    # ===-------------------------------------------------------------------===#
    # Escape hatches
    # ===-------------------------------------------------------------------===#

    def unwrap_legacy(self) -> Self._Legacy:
        """Returns the underlying legacy stream.

        Returns:
            The legacy stream this wrapper holds.
        """
        comptime assert (
            use_legacy_backend()
        ), "unwrap_legacy is only available in a legacy DeviceContext build"
        return self._impl

    def unwrap_hal(self) -> Self._HAL:
        """Returns the underlying HAL stream.

        Returns:
            The HAL stream this wrapper holds.
        """
        comptime assert (
            not use_legacy_backend()
        ), "unwrap_hal is only available in a HAL DeviceContext build"
        return self._impl

    # ===-------------------------------------------------------------------===#
    # Function enqueuer
    # ===-------------------------------------------------------------------===#

    comptime _LegacyEnqueuer = LegacyDeviceStream
    comptime _HALEnqueuer = HALDeviceStreamBase[Self.target]

    def _legacy_enqueuer(self) -> Self._LegacyEnqueuer:
        comptime assert (
            use_legacy_backend()
        ), "this stream is not on the legacy backend"
        return self._impl

    def _hal_enqueuer(self) -> Self._HALEnqueuer:
        comptime assert (
            not use_legacy_backend()
        ), "this stream is not on the HAL backend"
        return self._impl

    # ===-------------------------------------------------------------------===#
    # Methods
    # ===-------------------------------------------------------------------===#

    def synchronize(self) raises:
        """Blocks until all work enqueued on this stream has completed.

        Raises:
            If the synchronization fails.
        """
        comptime if use_legacy_backend():
            self._impl.synchronize()
        else:
            self._impl.synchronize()

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
        comptime if use_legacy_backend():
            self._impl.enqueue_wait_for(event._impl)
            return
        comptime assert not use_legacy_backend()  # TODO(MOCO-4206)

        # An event that was never recorded has nothing to wait on.
        if case .Some(recorded) = event._impl._event[]:
            self._impl._stream[].wait_for_events(recorded)

    def record_event(self, event: DeviceEvent) raises:
        """Records an event in this stream.

        This function records the given event at the current point in this
        stream. All operations in the stream that were enqueued before this call
        will complete before the event is triggered.

        Args:
            event: The event to record.

        Raises:
            If event recording fails.

        Example:

        ```mojo
        from max.gpu.host import DeviceContext

        var ctx = DeviceContext()

        var default_stream = ctx.stream()
        var new_stream = ctx.create_stream()

        # Create an event
        var event = ctx.create_event()

        # Record the event in the default stream
        default_stream.record_event(event)

        # Work enqueued on new_stream after this call waits for the event
        new_stream.enqueue_wait_for(event)
        ```
        """
        comptime if use_legacy_backend():
            self._impl.record_event(event._impl)
            return
        comptime assert not use_legacy_backend()  # TODO(MOCO-4206)

        # Create and record the backing event for this `DeviceEvent`. If it was
        # previously recorded and no other `DeviceEvent`s have been constructed
        # by copy from this event, the existing backing event will be released.
        var hal_event = self._impl._stream[].record_event[
            EVENT_FLAG_CPU_VISIBLE
        ]()
        event._impl._event[] = Optional(hal_event^)

    def enqueue_host_func(
        self,
        func: def(OpaquePointer[MutAnyOrigin]) thin -> None,
        user_data: OpaquePointer[MutAnyOrigin],
    ) raises:
        """Enqueues a host function to run once this stream's work drains.

        Args:
            func: The host function to call.
            user_data: An opaque value passed to `func`.

        Raises:
            If the operation fails.
        """
        comptime if use_legacy_backend():
            self._impl.enqueue_host_func(func, user_data)
        else:
            self._impl.enqueue_host_func(func, user_data)

    def wait_for_flag(self, flag: CompletionFlag, value: UInt64) raises:
        """Stalls the stream until `flag` holds at least `value`.

        While the wait is pending, the thread that signals `flag` must not
        block on this stream first: no `synchronize`, no copy into or out of
        pageable host memory, and don't drop the flag.

        Args:
            flag: A `M::Driver::CompletionFlag`.
            value: The 64-bit value to wait for. Keep values below 2^63.

        Raises:
            If the device has no completion flags, the flag belongs to
            another backend, or enqueueing the wait fails.
        """
        comptime if use_legacy_backend():
            self._impl.wait_for_flag(flag._impl, value)
            return
        comptime assert not use_legacy_backend()  # TODO(MOCO-4206)

        flag._impl._inner[].wait(self._impl, value)

    @deprecated("Use `wait_for_flag` instead.")
    def wait_for_host_value(self, flag: CompletionFlag, value: UInt64) raises:
        """Blocks the host until `flag` reaches `value`.

        Args:
            flag: The completion flag to poll.
            value: The value to wait for.

        Raises:
            If the operation fails.
        """
        self.wait_for_flag(flag, value)

    def enqueue_function[
        *Ts: DevicePassable
    ](
        self,
        f: DeviceFunction,
        *args: *Ts,
        grid_dim: Dim,
        block_dim: Dim,
        cluster_dim: OptionalReg[Dim] = None,
        shared_mem_bytes: OptionalReg[Int] = None,
        var attributes: List[LaunchAttribute] = [],
        var constant_memory: List[ConstantMemoryMapping] = [],
    ) raises:
        """Enqueues a compiled function for execution on this stream.

        Parameters:
            Ts: The types of the arguments.

        Args:
            f: The compiled function to launch.
            args: The arguments to pass to the function.
            grid_dim: The grid dimensions.
            block_dim: The block dimensions.
            cluster_dim: The cluster dimensions.
            shared_mem_bytes: The dynamic shared memory in bytes.
            attributes: The launch attributes.
            constant_memory: The constant memory mappings.

        Raises:
            If the launch fails.
        """
        comptime if use_legacy_backend():
            self._impl.enqueue_function(
                f.unwrap_legacy(),
                *args,
                grid_dim=grid_dim,
                block_dim=block_dim,
                cluster_dim=cluster_dim,
                shared_mem_bytes=shared_mem_bytes,
                attributes=attributes^,
                constant_memory=constant_memory^,
            )
        else:
            self._impl.enqueue_function(
                f.unwrap_hal(),
                *args,
                grid_dim=grid_dim,
                block_dim=block_dim,
                cluster_dim=cluster_dim,
                shared_mem_bytes=shared_mem_bytes,
                attributes=attributes^,
                constant_memory=constant_memory^,
            )


comptime DeviceStream = DeviceStreamBase[]
"""The stream type on the default accelerator."""


@fieldwise_init
struct _HALDeviceEvent(ImplicitlyCopyable, Movable):
    """The state of a `DeviceEvent` on the HAL backend."""

    # The context keeps the device alive for as long as the event exists.
    var _ctx: DeviceContext.Unknown
    # The HAL has logical events, where creation and recording happen in one
    # step, so the backing event is empty until the event is first recorded.
    # `_event` is an `ArcPointer[Optional[...]]` so that all copies of a
    # `DeviceEvent` share the same backing event, which matches the behavior of
    # the existing reference-counted legacy `DeviceEvent`.
    var _event: ArcPointer[Optional[Event[EVENT_FLAG_CPU_VISIBLE]]]


struct DeviceEvent(ImplicitlyCopyable):
    """Represents a GPU event for synchronization between streams.

    A `DeviceEvent` allows for fine-grained synchronization between different
    GPU streams. Events can be recorded in one stream and waited for in another,
    enabling efficient coordination of asynchronous GPU operations.

    Example:

    ```mojo
    from max.gpu.host import DeviceContext

    var ctx = DeviceContext()

    var default_stream = ctx.stream()
    var new_stream = ctx.create_stream()

    # Create an event
    var event = ctx.create_event()

    # Record the event in the default stream
    default_stream.record_event(event)

    # Work enqueued on new_stream after this call waits for the event
    new_stream.enqueue_wait_for(event)
    ```
    """

    comptime _Legacy = LegacyDeviceEvent
    comptime _HAL = _HALDeviceEvent
    comptime _Impl = Self._Legacy if use_legacy_backend() else Self._HAL

    var _impl: Self._Impl

    def __init__(out self, ctx: DeviceContext[default_accelerator()]) raises:
        """Creates a new event recorded on the given context's default stream.

        Args:
            ctx: The device context to record the event on.

        Raises:
            If event creation or recording fails.
        """
        comptime if use_legacy_backend():
            self._impl = Self._Legacy(ctx._impl)
            return
        comptime assert not use_legacy_backend()  # TODO(MOCO-4206)

        var hal_event = ctx._impl._stream[].record_event[
            EVENT_FLAG_CPU_VISIBLE
        ]()
        self._impl = Self._HAL(
            ctx.into_unknown_target(), ArcPointer(Optional(hal_event^))
        )

    def __init__(out self, *, var _impl: Self._Impl):
        """Wraps an already-constructed backend event.

        Args:
            _impl: The backend event to take ownership of.
        """
        self._impl = _impl^

    def unwrap_legacy(self) -> Self._Legacy:
        """Returns the underlying legacy event.

        Returns:
            The legacy event this wrapper holds.
        """
        comptime assert (
            use_legacy_backend()
        ), "unwrap_legacy is only available in a legacy DeviceContext build"
        return self._impl

    def unwrap_hal(self) -> Self._HAL:
        """Returns the underlying HAL event.

        Returns:
            The HAL event this wrapper holds.
        """
        comptime assert (
            not use_legacy_backend()
        ), "unwrap_hal is only available in a HAL DeviceContext build"
        return self._impl

    def synchronize(self) raises:
        """Blocks the calling CPU thread until this event completes.

        This function waits until the event has been recorded and all
        operations before the event in the stream have completed.

        Raises:
            If synchronization fails.
        """
        comptime if use_legacy_backend():
            self._impl.synchronize()
            return
        comptime assert not use_legacy_backend()  # TODO(MOCO-4206)

        # An event that was never recorded has nothing to wait on.
        if case .Some(recorded) = self._impl._event[]:
            recorded.synchronize()


@fieldwise_init
struct _HALCompletionFlag(ImplicitlyCopyable):
    """The state of a `CompletionFlag` on the HAL backend.

    Shares the plugin flag that backs a C++ `M::Driver::CompletionFlag`, so a
    standalone Mojo program needs no C++ runtime to use one.
    """

    var _inner: ArcPointer[_CompletionFlagBox]


struct CompletionFlag(ImplicitlyCopyable):
    """A host-visible completion flag a host or device can wait on.

    On the HAL backend, this shares the plugin flag that backs a C++
    `M::Driver::CompletionFlag`, so a standalone Mojo program needs no C++
    runtime to use one.
    """

    comptime _Legacy = LegacyCompletionFlag
    comptime _HAL = _HALCompletionFlag
    comptime _Impl = Self._Legacy if use_legacy_backend() else Self._HAL

    var _impl: Self._Impl

    def __init__(out self, *, var _impl: Self._Impl):
        """Wraps an already-constructed backend flag.

        Args:
            _impl: The backend flag to take ownership of.
        """
        self._impl = _impl^

    def __init__(out self, ctx: DeviceContext) raises:
        """Allocates a flag, initialized to 0, on the device of `ctx`.

        Args:
            ctx: The device context to allocate the flag on.

        Raises:
            If the flag cannot be allocated.
        """
        comptime assert (
            use_legacy_backend()
        ), "this constructor is only available on the legacy DeviceContext"
        self._impl = Self._Legacy(ctx._impl)

    def __init__(out self, *, handle: _LegacyCompletionFlagPtr[mut=True]):
        """Wraps a native completion flag handle.

        Args:
            handle: The native handle of the flag.
        """
        comptime assert (
            use_legacy_backend()
        ), "this constructor is only available on the legacy DeviceContext"
        self._impl = Self._Legacy(handle=handle)

    def __init__(out self, *, unsafe_from_address: Int):
        """Wraps the C++ flag at an integer address, taking a reference.

        Intended for graph-op execute methods that extract a packed pointer
        from a payload buffer (mirroring how `runtime.launch_host_func`
        rebuilds its trampoline and user-data pointers). The caller asserts
        that `unsafe_from_address` points to a valid
        `M::Driver::CompletionFlag` that outlives every wait enqueued through
        this flag.

        Args:
            unsafe_from_address: Raw address of an `M::Driver::CompletionFlag`
                (as packed into a graph payload buffer by the producer side).
        """
        comptime if use_legacy_backend():
            self._impl = Self._Legacy(unsafe_from_address=unsafe_from_address)
            return
        comptime assert not use_legacy_backend()  # TODO(MOCO-4206)

        # The runtime library built this box, and a graph kernel reads its
        # layout here, so both must come from the same source build.
        self._impl = Self._HAL(
            external_call[
                "AsyncRT_CompletionFlag_mojoBox",
                Pointer[ArcPointer[_CompletionFlagBox], MutUntrackedOrigin],
            ](
                OpaquePointer[MutUntrackedOrigin](
                    unsafe_from_address=unsafe_from_address
                )
            )[]
        )

    def unwrap_legacy(self) -> Self._Legacy:
        """Returns the underlying legacy flag.

        Returns:
            The legacy flag this wrapper holds.
        """
        comptime assert (
            use_legacy_backend()
        ), "unwrap_legacy is only available in a legacy DeviceContext build"
        return self._impl

    def unwrap_hal(self) -> Self._HAL:
        """Returns the underlying HAL flag.

        Returns:
            The HAL flag this wrapper holds.
        """
        comptime assert (
            not use_legacy_backend()
        ), "unwrap_hal is only available in a HAL DeviceContext build"
        return self._impl

    def signal(self, value: UInt64):
        """Stores `value` in the flag with release ordering.

        Don't signal a lower value than the last one without a reset in
        between. Aborts if the plugin fails.

        Args:
            value: The value to store.
        """
        comptime if use_legacy_backend():
            self._impl.signal(value)
            return
        comptime assert not use_legacy_backend()  # TODO(MOCO-4206)

        # The legacy signature can't fail, so a plugin error aborts rather than
        # leaving a wait that never passes. `load` and `reset` do the same.
        try:
            self._impl._inner[].signal(value)
        except e:
            abort(t"CompletionFlag.signal: {e}")

    def load(self) -> UInt64:
        """Loads the flag's value with acquire ordering.

        Aborts if the plugin fails.

        Returns:
            The value the flag holds.
        """
        comptime if use_legacy_backend():
            return self._impl.load()
        comptime assert not use_legacy_backend()  # TODO(MOCO-4206)

        try:
            return self._impl._inner[].load()
        except e:
            abort(t"CompletionFlag.load: {e}")

    def reset(self):
        """Sets the flag back to 0.

        Don't reset while a wait on the flag is pending; doing so is undefined
        behavior. Aborts if the plugin fails.
        """
        comptime if use_legacy_backend():
            self._impl.reset()
            return
        comptime assert not use_legacy_backend()  # TODO(MOCO-4206)

        try:
            self._impl._inner[].reset()
        except e:
            abort(t"CompletionFlag.reset: {e}")


struct Device[target: CompilationTarget, origin: ImmOrigin]:
    """A borrowed view of one accelerator.

    Parameters:
        target: The accelerator targeted by this view.
        origin: The origin of the borrow.
    """

    comptime _Legacy = LegacyDevice[Self.target, Self.origin]
    comptime _HAL = HALDevice[Self.target, Self.origin]
    comptime _Impl = Self._Legacy if use_legacy_backend() else Self._HAL

    var _impl: Self._Impl

    var id: Int64
    """The ID of this device among the devices of its API."""

    def __init__(out self, *, var _impl: Self._Impl):
        """Wraps an already-constructed backend view.

        Args:
            _impl: The backend view to take ownership of.
        """
        comptime if use_legacy_backend():
            self.id = _impl.id
        else:
            self.id = _impl.id
        self._impl = _impl

    def unwrap_legacy(self) -> Self._Legacy:
        """Returns the underlying legacy view.

        Returns:
            The legacy device view this wrapper holds.
        """
        comptime assert (
            use_legacy_backend()
        ), "unwrap_legacy is only available in a legacy DeviceContext build"
        return self._impl

    def unwrap_hal(self) -> Self._HAL:
        """Returns the underlying HAL view.

        Returns:
            The HAL device view this wrapper holds.
        """
        comptime assert (
            not use_legacy_backend()
        ), "unwrap_hal is only available in a HAL DeviceContext build"
        return self._impl

    def get_name(self) raises -> String:
        """Returns the device's name.

        Returns:
            The device name.

        Raises:
            If the query fails.
        """
        comptime if use_legacy_backend():
            return self._impl.get_name()
        else:
            return self._impl.get_name()

    def get_attribute(self, attr: DeviceAttribute) raises -> Int:
        """Returns the value of a device attribute.

        Args:
            attr: The attribute to query.

        Returns:
            The attribute's value.

        Raises:
            If the query fails.
        """
        comptime if use_legacy_backend():
            return self._impl.get_attribute(attr)
        else:
            return self._impl.get_attribute(attr)

    def multiprocessor_count(self) raises -> Int:
        """Returns the device's multiprocessor count.

        Returns:
            The number of multiprocessors.

        Raises:
            If the query fails.
        """
        comptime if use_legacy_backend():
            return self._impl.multiprocessor_count()
        else:
            return self._impl.multiprocessor_count()

    def max_threads_per_block(self) raises -> Int:
        """Returns the maximum number of threads in a block.

        Returns:
            The maximum thread count.

        Raises:
            If the query fails.
        """
        comptime if use_legacy_backend():
            return self._impl.max_threads_per_block()
        else:
            return self._impl.max_threads_per_block()

    def max_shared_memory_per_block(self) raises -> Int:
        """Returns the maximum shared memory per block.

        Returns:
            The maximum shared memory in bytes.

        Raises:
            If the query fails.
        """
        comptime if use_legacy_backend():
            return self._impl.max_shared_memory_per_block()
        else:
            return self._impl.max_shared_memory_per_block()

    def max_grid_dim_x(self) raises -> Int:
        """Returns the maximum grid dimension along x.

        Returns:
            The maximum x dimension.

        Raises:
            If the query fails.
        """
        comptime if use_legacy_backend():
            return self._impl.max_grid_dim_x()
        else:
            return self._impl.max_grid_dim_x()

    def max_grid_dim_y(self) raises -> Int:
        """Returns the maximum grid dimension along y.

        Returns:
            The maximum y dimension.

        Raises:
            If the query fails.
        """
        comptime if use_legacy_backend():
            return self._impl.max_grid_dim_y()
        else:
            return self._impl.max_grid_dim_y()

    def max_grid_dim_z(self) raises -> Int:
        """Returns the maximum grid dimension along z.

        Returns:
            The maximum z dimension.

        Raises:
            If the query fails.
        """
        comptime if use_legacy_backend():
            return self._impl.max_grid_dim_z()
        else:
            return self._impl.max_grid_dim_z()

    def max_grid_dim(self) raises -> Dim:
        """Returns the maximum grid dimensions.

        Returns:
            The maximum grid dimensions.

        Raises:
            If the query fails.
        """
        comptime if use_legacy_backend():
            return self._impl.max_grid_dim()
        else:
            return self._impl.max_grid_dim()

    def is_host_unified(self) raises -> Bool:
        """Returns whether this device shares memory with the host.

        Returns:
            True when host and device memory are unified.

        Raises:
            If the query fails.
        """
        comptime if use_legacy_backend():
            return self._impl.is_host_unified()
        else:
            return self._impl.is_host_unified()

    def get_arch(self) raises -> String:
        """Returns the device architecture name.

        Returns:
            The architecture name.

        Raises:
            If the query fails.
        """
        comptime if use_legacy_backend():
            return self._impl.get_arch()
        else:
            return self._impl.get_arch()

    def can_access_peer[
        peer_target: CompilationTarget, peer_origin: ImmOrigin, //
    ](self, peer: Device[peer_target, peer_origin]) raises -> Bool:
        """Returns whether this device can access `peer`'s memory.

        Parameters:
            peer_target: The peer's target.
            peer_origin: The peer's origin.

        Args:
            peer: The peer device.

        Returns:
            True if peer access is available.

        Raises:
            If the query fails.
        """
        comptime if use_legacy_backend():
            return self._impl.can_access_peer(peer._impl)
        else:
            return self._impl.can_access_peer(peer._impl)


struct DeviceContextArray[
    length: Int, target: CompilationTarget = default_accelerator()
](Copyable, Sized):
    """A fixed-length array of device contexts, one per device.

    Used by multi-device custom-op `execute` methods to receive one
    `DeviceContext` per participating device. The graph compiler recognizes
    this type and synthesizes it from the per-device contexts discovered on the
    operation, so kernels can index into it like a homogeneous array without the
    compiler having to introspect a generic `Array` parameter.

    Parameters:
        length: The number of `DeviceContext` values in the collection.
        target: The target compilation accelerator.
    """

    @deprecated(
        "`DeviceContextArray.size` is deprecated, use"
        " `DeviceContextArray.length`."
    )
    comptime size = Self.length
    """Deprecated alias for `length`."""

    var device_contexts: Array[DeviceContext[Self.target], Self.length]
    """The per-device contexts."""

    def __init__(out self) raises:
        """Creates one context per available device.

        Raises:
            If creating any device context fails.
        """
        var scratch = Array[Optional[DeviceContext[Self.target]], Self.length]()
        for i in range(Self.length):
            scratch[i] = DeviceContext[Self.target](device_id=i)
        self.device_contexts = Array[DeviceContext[Self.target], Self.length](
            fill_with=lambda (i: Int) {ref} -> DeviceContext[
                Self.target
            ]: scratch[i].take()
        )

    @inline(.always)
    def __init__(
        out self,
        var device_contexts: Array[DeviceContext[Self.target], Self.length],
    ):
        """Creates the array from existing contexts.

        Args:
            device_contexts: The per-device contexts to store.
        """
        self.device_contexts = device_contexts^

    @inline(.always)
    def __init__[
        *, __literal_size__: Int
    ](
        out self: DeviceContextArray[__literal_size__, Self.target],
        var *device_contexts: DeviceContext[Self.target],
        __list_literal__: NoneType = None,
    ):
        """Creates the array from a list of contexts.

        The graph compiler's multi-device lowering path uses this constructor:
        it synthesizes `DeviceContextArray[length=N](ctx0, ctx1, ..., ctxN-1)`
        directly from the per-device contexts attached to the kernel.

        Parameters:
            __literal_size__: The number of contexts in the literal, inferred
                from the number of elements given.

        Args:
            device_contexts: One `DeviceContext` per device, exactly `length`
                of them.
            __list_literal__: Marker that lets this constructor accept
                list-literal syntax.
        """
        self.device_contexts = {uninitialized = True}
        var ptr = self.device_contexts.unsafe_ptr()
        comptime for i in range(__literal_size__):
            ptr.unsafe_offset(i).unsafe_write_move_from(
                Pointer(to=device_contexts[i])
            )
        # Do not destroy the elements when their backing storage goes away.
        device_contexts^._annihilate()

    def __getitem_param__[index: Int](self) -> DeviceContext[Self.target]:
        """Returns the context at a compile-time index.

        Parameters:
            index: The index to read.

        Returns:
            The context at `index`.
        """
        return self.device_contexts[index]

    def __getitem__[I: Indexer, //](self, idx: I) -> DeviceContext[Self.target]:
        """Returns the context at `idx`.

        Parameters:
            I: The index type.

        Args:
            idx: The index to read.

        Returns:
            The context at `idx`.
        """
        return self.device_contexts[idx]

    def __len__(self) -> Int:
        """Returns the number of contexts.

        Returns:
            The array length.
        """
        return Self.length

    def filter_gpu_contexts[
        num_gpu_devices: Int
    ](self) raises -> Array[DeviceContext[Self.target], num_gpu_devices]:
        """Returns the contexts that run on a GPU, in their original order.

        Some kernels receive a `DeviceContextArray` that mixes GPU contexts with
        CPU contexts carrying host-side pointers.

        Parameters:
            num_gpu_devices: The expected number of GPU contexts.

        Returns:
            The GPU contexts.

        Raises:
            If the number of GPU contexts is not `num_gpu_devices`.
        """
        var gpu_count = 0
        for i in range(Self.length):
            if self[i].api() != "cpu":
                gpu_count += 1
        if gpu_count != num_gpu_devices:
            raise Error("Invalid number of GPU device contexts")

        var gpu_contexts = Array[DeviceContext[Self.target], num_gpu_devices](
            uninitialized=True
        )
        var ptr = gpu_contexts.unsafe_ptr()
        var dev_idx = 0
        for i in range(Self.length):
            if self[i].api() != "cpu":
                ptr.unsafe_offset(dev_idx).unsafe_write(self[i])
                dev_idx += 1
        return gpu_contexts^


struct DeviceMulticastBuffer[dtype: DType](Movable):
    """A buffer whose memory is visible to several devices at once.

    Parameters:
        dtype: The element dtype.
    """

    comptime _Legacy = LegacyDeviceMulticastBuffer[Self.dtype]
    comptime _HAL = HALDeviceMulticastBuffer[Self.dtype]
    comptime _Impl = Self._Legacy if use_legacy_backend() else Self._HAL

    var _impl: Self._Impl

    def __init__(
        out self, var contexts: List[DeviceContext[]], size: Int
    ) raises:
        """Allocates a multicast buffer across `contexts`.

        Args:
            contexts: The contexts that share the buffer.
            size: The number of elements to allocate.

        Raises:
            If the allocation fails or multicast is unsupported.
        """
        comptime if use_legacy_backend():
            var legacy = List[LegacyDeviceContext[]]()
            for ctx in contexts:
                legacy.append(ctx._impl)
            self._impl = Self._Legacy(legacy^, size=size)
        else:
            var hal = List[HALDeviceContext[]]()
            for ctx in contexts:
                hal.append(ctx._impl)
            self._impl = Self._HAL(hal^, size=size)

    def unwrap_legacy(deinit self) -> Self._Legacy:
        """Returns the underlying legacy buffer.

        Returns:
            The legacy buffer this wrapper holds.
        """
        comptime assert (
            use_legacy_backend()
        ), "unwrap_legacy is only available in a legacy DeviceContext build"
        return self._impl^

    def unwrap_hal(deinit self) -> Self._HAL:
        """Returns the underlying HAL buffer.

        Returns:
            The HAL buffer this wrapper holds.
        """
        comptime assert (
            not use_legacy_backend()
        ), "unwrap_hal is only available in a HAL DeviceContext build"
        return self._impl^

    def unicast_buffer_for(
        self, ctx: DeviceContext[]
    ) raises -> DeviceBuffer[Self.dtype]:
        """Returns a buffer addressing this memory from only `ctx`.

        Args:
            ctx: The context the view is for.

        Returns:
            A per-device view of the buffer.

        Raises:
            If `ctx` is not one of the sharing contexts.
        """
        comptime if use_legacy_backend():
            return DeviceBuffer[Self.dtype](
                _impl=self._impl.unicast_buffer_for(ctx._impl)
            )
        else:
            return DeviceBuffer[Self.dtype](
                _impl=self._impl.unicast_buffer_for(ctx._impl)
            )

    def multicast_buffer_for(
        self, ctx: DeviceContext[]
    ) raises -> DeviceBuffer[Self.dtype]:
        """Returns a buffer addressing this memory from every sharing device.

        Args:
            ctx: The context issuing the accesses.

        Returns:
            A multicast view of the buffer.

        Raises:
            If `ctx` is not one of the sharing contexts.
        """
        comptime if use_legacy_backend():
            return DeviceBuffer[Self.dtype](
                _impl=self._impl.multicast_buffer_for(ctx._impl)
            )
        else:
            return DeviceBuffer[Self.dtype](
                _impl=self._impl.multicast_buffer_for(ctx._impl)
            )
