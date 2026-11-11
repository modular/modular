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

from std.format import Writer
from std.sys.info import CompilationTarget

from max._driver.max_license.device_context_hal.device_context_hal import (
    HALCompletionFlag,
    HALDevice,
    HALDeviceBuffer,
    HALDeviceContext,
    HALDeviceEvent,
    HALDeviceMulticastBuffer,
    HALDeviceStreamBase,
    HALHostBuffer,
)
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
from max.gpu.host._legacy.device_attribute import DeviceAttribute
from max.gpu.host._legacy.dim import Dim
from max.gpu.host.info import default_accelerator
from max._driver.max_license.device_context_dispatch.function import (
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
            self._impl = rebind_var[Self._Impl](
                Self._Legacy(
                    rebind[DeviceContext[Self.target]._Legacy](ctx._impl)
                )
            )
        else:
            self._impl = rebind_var[Self._Impl](
                Self._HAL(rebind[DeviceContext[Self.target]._HAL](ctx._impl))
            )

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
        return rebind[Self._Legacy](self._impl)

    def unwrap_hal(self) -> Self._HAL:
        """Returns the underlying HAL stream.

        Returns:
            The HAL stream this wrapper holds.
        """
        comptime assert (
            not use_legacy_backend()
        ), "unwrap_hal is only available in a HAL DeviceContext build"
        return rebind[Self._HAL](self._impl)

    # ===-------------------------------------------------------------------===#
    # Function enqueuer
    # ===-------------------------------------------------------------------===#

    comptime _LegacyEnqueuer = LegacyDeviceStream
    comptime _HALEnqueuer = HALDeviceStreamBase[Self.target]

    def _legacy_enqueuer(self) -> Self._LegacyEnqueuer:
        comptime assert (
            use_legacy_backend()
        ), "this stream is not on the legacy backend"
        return rebind[Self._Legacy](self._impl)

    def _hal_enqueuer(self) -> Self._HALEnqueuer:
        comptime assert (
            not use_legacy_backend()
        ), "this stream is not on the HAL backend"
        return rebind[Self._HAL](self._impl)

    # ===-------------------------------------------------------------------===#
    # Methods
    # ===-------------------------------------------------------------------===#

    def synchronize(self) raises:
        """Blocks until all work enqueued on this stream has completed.

        Raises:
            If the synchronization fails.
        """
        comptime if use_legacy_backend():
            rebind[Self._Legacy](self._impl).synchronize()
        else:
            rebind[Self._HAL](self._impl).synchronize()

    def enqueue_wait_for(self, event: DeviceEvent) raises:
        """Enqueues a wait for `event` on this stream.

        Args:
            event: The event to wait for.

        Raises:
            If the operation fails.
        """
        comptime if use_legacy_backend():
            rebind[Self._Legacy](self._impl).enqueue_wait_for(
                rebind[DeviceEvent._Legacy](event._impl)
            )
        else:
            rebind[Self._HAL](self._impl).enqueue_wait_for(
                rebind[DeviceEvent._HAL](event._impl)
            )

    def record_event(self, event: DeviceEvent) raises:
        """Records `event` on this stream.

        Args:
            event: The event to record.

        Raises:
            If the operation fails.
        """
        comptime if use_legacy_backend():
            rebind[Self._Legacy](self._impl).record_event(
                rebind[DeviceEvent._Legacy](event._impl)
            )
        else:
            rebind[Self._HAL](self._impl).record_event(
                rebind[DeviceEvent._HAL](event._impl)
            )

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
            rebind[Self._Legacy](self._impl).enqueue_host_func(func, user_data)
        else:
            rebind[Self._HAL](self._impl).enqueue_host_func(func, user_data)

    def wait_for_flag(self, flag: CompletionFlag, value: UInt64) raises:
        """Blocks until `flag` reaches `value` on the device.

        Args:
            flag: The completion flag to poll.
            value: The value to wait for.

        Raises:
            If the operation fails.
        """
        comptime if use_legacy_backend():
            rebind[Self._Legacy](self._impl).wait_for_flag(
                rebind[CompletionFlag._Legacy](flag._impl), value
            )
        else:
            rebind[Self._HAL](self._impl).wait_for_flag(
                rebind[CompletionFlag._HAL](flag._impl), value
            )

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


comptime DeviceStream = DeviceStreamBase[]
"""The stream type on the default accelerator."""


struct DeviceEvent(ImplicitlyCopyable):
    """An event used to synchronize work across streams."""

    comptime _Legacy = LegacyDeviceEvent
    comptime _HAL = HALDeviceEvent
    comptime _Impl = Self._Legacy if use_legacy_backend() else Self._HAL

    var _impl: Self._Impl

    def __init__(out self, ctx: DeviceContext[default_accelerator()]) raises:
        """Creates an event on the given device context's device.

        Args:
            ctx: The device context that owns the event.

        Raises:
            If the event cannot be created.
        """
        comptime if use_legacy_backend():
            self._impl = rebind_var[Self._Impl](
                Self._Legacy(
                    rebind[DeviceContext[default_accelerator()]._Legacy](
                        ctx._impl
                    )
                )
            )
        else:
            self._impl = rebind_var[Self._Impl](
                Self._HAL(
                    rebind[DeviceContext[default_accelerator()]._HAL](ctx._impl)
                )
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
        return rebind[Self._Legacy](self._impl)

    def unwrap_hal(self) -> Self._HAL:
        """Returns the underlying HAL event.

        Returns:
            The HAL event this wrapper holds.
        """
        comptime assert (
            not use_legacy_backend()
        ), "unwrap_hal is only available in a HAL DeviceContext build"
        return rebind[Self._HAL](self._impl)

    def synchronize(self) raises:
        """Blocks until this event's work has completed.

        Raises:
            If the synchronization fails.
        """
        comptime if use_legacy_backend():
            rebind[Self._Legacy](self._impl).synchronize()
        else:
            rebind[Self._HAL](self._impl).synchronize()


struct CompletionFlag(ImplicitlyCopyable):
    """A device-side counter a host or device can wait on."""

    comptime _Legacy = LegacyCompletionFlag
    comptime _HAL = HALCompletionFlag
    comptime _Impl = Self._Legacy if use_legacy_backend() else Self._HAL

    var _impl: Self._Impl

    def __init__(out self, *, var _impl: Self._Impl):
        """Wraps an already-constructed backend flag.

        Args:
            _impl: The backend flag to take ownership of.
        """
        self._impl = _impl^

    def __init__(out self, ctx: DeviceContext) raises:
        """Allocates a flag on the device of `ctx`.

        Args:
            ctx: The device context to allocate the flag on.

        Raises:
            If the flag cannot be allocated.
        """
        comptime assert (
            use_legacy_backend()
        ), "this constructor is only available on the legacy DeviceContext"
        self._impl = rebind_var[Self._Impl](
            Self._Legacy(
                rebind[DeviceContext[default_accelerator()]._Legacy](ctx._impl)
            )
        )

    def __init__(out self, *, handle: _LegacyCompletionFlagPtr[mut=True]):
        """Wraps a native completion flag handle.

        Args:
            handle: The native handle of the flag.
        """
        comptime assert (
            use_legacy_backend()
        ), "this constructor is only available on the legacy DeviceContext"
        self._impl = rebind_var[Self._Impl](Self._Legacy(handle=handle))

    def __init__(out self, *, unsafe_from_address: Int):
        """Wraps the flag stored at a raw address.

        Args:
            unsafe_from_address: The address of the flag.
        """
        comptime if use_legacy_backend():
            self._impl = rebind_var[Self._Impl](
                Self._Legacy(unsafe_from_address=unsafe_from_address)
            )
        else:
            self._impl = rebind_var[Self._Impl](
                Self._HAL(unsafe_from_address=unsafe_from_address)
            )

    def unwrap_legacy(self) -> Self._Legacy:
        """Returns the underlying legacy flag.

        Returns:
            The legacy flag this wrapper holds.
        """
        comptime assert (
            use_legacy_backend()
        ), "unwrap_legacy is only available in a legacy DeviceContext build"
        return rebind[Self._Legacy](self._impl)

    def unwrap_hal(self) -> Self._HAL:
        """Returns the underlying HAL flag.

        Returns:
            The HAL flag this wrapper holds.
        """
        comptime assert (
            not use_legacy_backend()
        ), "unwrap_hal is only available in a HAL DeviceContext build"
        return rebind[Self._HAL](self._impl)

    def signal(self, value: UInt64):
        """Sets the flag to `value`.

        Args:
            value: The value to publish.
        """
        comptime if use_legacy_backend():
            rebind[Self._Legacy](self._impl).signal(value)
        else:
            rebind[Self._HAL](self._impl).signal(value)

    def load(self) -> UInt64:
        """Returns the flag's current value.

        Returns:
            The flag's value.
        """
        comptime if use_legacy_backend():
            return rebind[Self._Legacy](self._impl).load()
        else:
            return rebind[Self._HAL](self._impl).load()

    def reset(self):
        """Resets the flag to zero."""
        comptime if use_legacy_backend():
            rebind[Self._Legacy](self._impl).reset()
        else:
            rebind[Self._HAL](self._impl).reset()


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
            self.id = rebind[Self._Legacy](_impl).id
        else:
            self.id = rebind[Self._HAL](_impl).id
        self._impl = _impl

    def unwrap_legacy(self) -> Self._Legacy:
        """Returns the underlying legacy view.

        Returns:
            The legacy device view this wrapper holds.
        """
        comptime assert (
            use_legacy_backend()
        ), "unwrap_legacy is only available in a legacy DeviceContext build"
        return rebind[Self._Legacy](self._impl)

    def unwrap_hal(self) -> Self._HAL:
        """Returns the underlying HAL view.

        Returns:
            The HAL device view this wrapper holds.
        """
        comptime assert (
            not use_legacy_backend()
        ), "unwrap_hal is only available in a HAL DeviceContext build"
        return rebind[Self._HAL](self._impl)

    def get_name(self) raises -> String:
        """Returns the device's name.

        Returns:
            The device name.

        Raises:
            If the query fails.
        """
        comptime if use_legacy_backend():
            return rebind[Self._Legacy](self._impl).get_name()
        else:
            return rebind[Self._HAL](self._impl).get_name()

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
            return rebind[Self._Legacy](self._impl).get_attribute(attr)
        else:
            return rebind[Self._HAL](self._impl).get_attribute(attr)

    def multiprocessor_count(self) raises -> Int:
        """Returns the device's multiprocessor count.

        Returns:
            The number of multiprocessors.

        Raises:
            If the query fails.
        """
        comptime if use_legacy_backend():
            return rebind[Self._Legacy](self._impl).multiprocessor_count()
        else:
            return rebind[Self._HAL](self._impl).multiprocessor_count()

    def max_threads_per_block(self) raises -> Int:
        """Returns the maximum number of threads in a block.

        Returns:
            The maximum thread count.

        Raises:
            If the query fails.
        """
        comptime if use_legacy_backend():
            return rebind[Self._Legacy](self._impl).max_threads_per_block()
        else:
            return rebind[Self._HAL](self._impl).max_threads_per_block()

    def max_shared_memory_per_block(self) raises -> Int:
        """Returns the maximum shared memory per block.

        Returns:
            The maximum shared memory in bytes.

        Raises:
            If the query fails.
        """
        comptime if use_legacy_backend():
            return rebind[Self._Legacy](
                self._impl
            ).max_shared_memory_per_block()
        else:
            return rebind[Self._HAL](self._impl).max_shared_memory_per_block()

    def max_grid_dim_x(self) raises -> Int:
        """Returns the maximum grid dimension along x.

        Returns:
            The maximum x dimension.

        Raises:
            If the query fails.
        """
        comptime if use_legacy_backend():
            return rebind[Self._Legacy](self._impl).max_grid_dim_x()
        else:
            return rebind[Self._HAL](self._impl).max_grid_dim_x()

    def max_grid_dim_y(self) raises -> Int:
        """Returns the maximum grid dimension along y.

        Returns:
            The maximum y dimension.

        Raises:
            If the query fails.
        """
        comptime if use_legacy_backend():
            return rebind[Self._Legacy](self._impl).max_grid_dim_y()
        else:
            return rebind[Self._HAL](self._impl).max_grid_dim_y()

    def max_grid_dim_z(self) raises -> Int:
        """Returns the maximum grid dimension along z.

        Returns:
            The maximum z dimension.

        Raises:
            If the query fails.
        """
        comptime if use_legacy_backend():
            return rebind[Self._Legacy](self._impl).max_grid_dim_z()
        else:
            return rebind[Self._HAL](self._impl).max_grid_dim_z()

    def max_grid_dim(self) raises -> Dim:
        """Returns the maximum grid dimensions.

        Returns:
            The maximum grid dimensions.

        Raises:
            If the query fails.
        """
        comptime if use_legacy_backend():
            return rebind[Self._Legacy](self._impl).max_grid_dim()
        else:
            return rebind[Self._HAL](self._impl).max_grid_dim()

    def is_host_unified(self) raises -> Bool:
        """Returns whether this device shares memory with the host.

        Returns:
            True when host and device memory are unified.

        Raises:
            If the query fails.
        """
        comptime if use_legacy_backend():
            return rebind[Self._Legacy](self._impl).is_host_unified()
        else:
            return rebind[Self._HAL](self._impl).is_host_unified()

    def get_arch(self) raises -> String:
        """Returns the device architecture name.

        Returns:
            The architecture name.

        Raises:
            If the query fails.
        """
        comptime if use_legacy_backend():
            return rebind[Self._Legacy](self._impl).get_arch()
        else:
            return rebind[Self._HAL](self._impl).get_arch()

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
            return rebind[Self._Legacy](self._impl).can_access_peer(
                rebind[LegacyDevice[peer_target, peer_origin]](peer._impl)
            )
        else:
            return rebind[Self._HAL](self._impl).can_access_peer(
                rebind[HALDevice[peer_target, peer_origin]](peer._impl)
            )


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
                legacy.append(rebind[LegacyDeviceContext[]](ctx._impl))
            self._impl = rebind_var[Self._Impl](
                Self._Legacy(legacy^, size=size)
            )
        else:
            var hal = List[HALDeviceContext[]]()
            for ctx in contexts:
                hal.append(rebind[HALDeviceContext[]](ctx._impl))
            self._impl = rebind_var[Self._Impl](Self._HAL(hal^, size=size))

    def unwrap_legacy(deinit self) -> Self._Legacy:
        """Returns the underlying legacy buffer.

        Returns:
            The legacy buffer this wrapper holds.
        """
        comptime assert (
            use_legacy_backend()
        ), "unwrap_legacy is only available in a legacy DeviceContext build"
        return rebind_var[Self._Legacy](self._impl^)

    def unwrap_hal(deinit self) -> Self._HAL:
        """Returns the underlying HAL buffer.

        Returns:
            The HAL buffer this wrapper holds.
        """
        comptime assert (
            not use_legacy_backend()
        ), "unwrap_hal is only available in a HAL DeviceContext build"
        return rebind_var[Self._HAL](self._impl^)

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
                _impl=rebind_var[DeviceBuffer[Self.dtype]._Impl](
                    rebind[Self._Legacy](self._impl).unicast_buffer_for(
                        rebind[LegacyDeviceContext[]](ctx._impl)
                    )
                )
            )
        else:
            return DeviceBuffer[Self.dtype](
                _impl=rebind_var[DeviceBuffer[Self.dtype]._Impl](
                    rebind[Self._HAL](self._impl).unicast_buffer_for(
                        rebind[HALDeviceContext[]](ctx._impl)
                    )
                )
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
                _impl=rebind_var[DeviceBuffer[Self.dtype]._Impl](
                    rebind[Self._Legacy](self._impl).multicast_buffer_for(
                        rebind[LegacyDeviceContext[]](ctx._impl)
                    )
                )
            )
        else:
            return DeviceBuffer[Self.dtype](
                _impl=rebind_var[DeviceBuffer[Self.dtype]._Impl](
                    rebind[Self._HAL](self._impl).multicast_buffer_for(
                        rebind[HALDeviceContext[]](ctx._impl)
                    )
                )
            )
