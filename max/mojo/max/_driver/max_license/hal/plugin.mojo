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
"""HAL plugin loader.

Loads a HAL plugin shared library and resolves all required function pointers
from the M_driver_* C API.
"""

from std.collections import OptionalReg

from std.ffi import (
    OwnedDLHandle,
    RTLD,
    CStringSpan,
    UnsafeUnion,
    c_char,
)

from std.memory import (
    ImmPointer,
    MutOpaquePointer,
    MutPointer,
    OpaquePointer,
    Pointer,
    MaybeUninit,
)

from .status import (
    STATUS_SUCCESS,
    STATUS_UNKNOWN_ERROR,
    STATUS_UNKNOWN_PROPERTY_NAME,
    HALError,
)

from .plugins import *


@fieldwise_init
struct RawDriver(Movable):
    var _raw: RawPlugin

    # Set by `Driver.create` after `M_driver_create` succeeds.
    # None until the driver handle is bound.
    # Used to avoid every HAL struct needing to carry around
    # a DriverHandle or indirect through the ownership hierarchy
    # to obtain it.
    var _driver_handle: DriverHandle

    @staticmethod
    def load(plugin_spec: String) raises HALError -> Self:
        """Load a plugin from a spec string of the form 'name@/path/to/lib.so'.

        If no '@' is present, the entire string is used as both name and path.
        """
        var name: String
        var so_path: String
        var parts = plugin_spec.split("@", maxsplit=1)
        if len(parts) == 2:
            name = String(parts[0])
            so_path = String(parts[1])
        else:
            name = plugin_spec
            so_path = plugin_spec

        var raw: RawPlugin
        var driver_handle: DriverHandle
        try:
            var lib = OwnedDLHandle(so_path, RTLD.NOW)
            raw = RawPlugin(lib^, so_path)
            raw.name = name
            raw.so_path = so_path

            driver_handle = raw._create_init_driver()
        except e:
            raise HALError(
                STATUS_UNKNOWN_ERROR,
                message=String(t"failed to load plugin '{so_path}': {e}"),
            )

        return RawDriver(raw^, driver_handle)

    def __deinit__(deinit self):
        # Move `_plugin` into a local so its `OwnedDLHandle` stays alive
        # until after the `destroy` call returns — ASAP destruction would
        # otherwise `dlclose` the `.so` (unmapping the code page) before
        # we call through the function pointer.
        var plugin = self._raw^
        var status = plugin.base.destroy.get()(self._driver_handle)
        if status != STATUS_SUCCESS:
            print("warning: driver destroy failed with status:", status)
        _ = plugin^

    # TODO: `HALFunction` should, eventually, be able to
    # do the vast majority of this automagically with a _partially_ bound
    # thin fn type that can ensure safety at call site, and a rebind
    # to some concrete type internally.
    # See DRIV-6, MOCO-3661

    def get_device_count(mut self) raises HALError -> Int64:
        var num_devices = MaybeUninit(Int64(0))
        var status = self._raw.base.device_count.get()(
            self._driver_handle, OutParam[Int64](to=num_devices)
        )

        if status != STATUS_SUCCESS:
            var err = self.get_status_message(status)
            raise err^

        return num_devices.unsafe_assume_init()

    def get_device_attribute(
        self, device: DeviceHandle, attribute: Int32
    ) raises HALError -> Int32:
        var value = MaybeUninit[Int32]()
        var status = self._raw.base.device_attribute.get()(
            device, attribute, OutParam[Int32](to=value)
        )
        if status != STATUS_SUCCESS:
            var err = self.get_status_message(status)
            raise HALError(
                err.status,
                message=String(
                    t"failed to get device attribute {attribute}: {err.message}"
                ),
            )
        return value.unsafe_assume_init()

    # ===-------------------------------------------------------------------===#
    # Queue operations
    # ===-------------------------------------------------------------------===#

    def create_queue(
        self, context: ContextHandle
    ) raises HALError -> QueueHandle:
        var queue = MaybeUninit[QueueHandle]()
        var status = self._raw.base.queue_create.get()(
            context, OutParam[QueueHandle](to=queue)
        )
        if status != STATUS_SUCCESS:
            var err = self.get_status_message(status)
            raise HALError(
                err.status,
                message=String(t"failed to create queue: {err.message}"),
            )
        return queue.unsafe_assume_init()

    def destroy_queue[
        ctx_origin: Origin, //
    ](
        self,
        context: Pointer[ContextHandle.T, ctx_origin],
        queue: QueueHandle,
    ) raises HALError:
        var status = self._raw.base.queue_destroy.get()(
            rebind[ContextHandle](context), queue
        )
        if status != STATUS_SUCCESS:
            var err = self.get_status_message(status)
            raise HALError(
                err.status,
                message=String(t"failed to destroy queue: {err.message}"),
            )

    # ===-------------------------------------------------------------------===#
    # Context lifecycle
    # ===-------------------------------------------------------------------===#

    def destroy_context(self, context: ContextHandle) raises HALError:
        var status = self._raw.base.context_destroy.get()(context)
        if status != STATUS_SUCCESS:
            var err = self.get_status_message(status)
            raise HALError(
                err.status,
                message=String(t"failed to destroy context: {err.message}"),
            )

    def set_context_current(self, context: ContextHandle) raises HALError:
        var status = self._raw.base.context_set_current.get()(context)
        if status != STATUS_SUCCESS:
            var err = self.get_status_message(status)
            raise HALError(
                err.status,
                message=String(t"failed to set context current: {err.message}"),
            )

    def get_current_driver_context(
        self, context: ContextHandle
    ) raises HALError -> Int:
        var driver_context = MaybeUninit[Int]()
        var status = self._raw.base.context_get_current_driver_context.get()(
            context, OutParam[Int](to=driver_context)
        )
        if status != STATUS_SUCCESS:
            var err = self.get_status_message(status)
            raise HALError(
                err.status,
                message=String(
                    t"failed to get current driver context: {err.message}"
                ),
            )
        return driver_context.unsafe_assume_init()

    def set_current_driver_context(
        self, context: ContextHandle, driver_context: Int
    ) raises HALError:
        var status = self._raw.base.context_set_current_driver_context.get()(
            context, driver_context
        )
        if status != STATUS_SUCCESS:
            var err = self.get_status_message(status)
            raise HALError(
                err.status,
                message=String(
                    t"failed to set current driver context: {err.message}"
                ),
            )

    def get_context_memory_info(
        self, context: ContextHandle
    ) raises HALError -> Tuple[UInt64, UInt64]:
        var free = MaybeUninit[UInt64]()
        var total = MaybeUninit[UInt64]()
        var status = self._raw.base.context_memory_info.get()(
            context, OutParam[UInt64](to=free), OutParam[UInt64](to=total)
        )
        if status != STATUS_SUCCESS:
            var err = self.get_status_message(status)
            raise HALError(
                err.status,
                message=String(t"failed to get memory info: {err.message}"),
            )
        return (
            free.unsafe_assume_init(),
            total.unsafe_assume_init(),
        )

    # ===-------------------------------------------------------------------===#
    # Bundle lifecycle
    # ===-------------------------------------------------------------------===#

    def unload_bundle(
        self, context: ContextHandle, bundle: RuntimeBundleHandle
    ) raises HALError:
        var status = self._raw.base.bundle_unload.get()(context, bundle)
        if status != STATUS_SUCCESS:
            var err = self.get_status_message(status)
            raise HALError(
                err.status,
                message=String(t"failed to unload bundle: {err.message}"),
            )

    # ===-------------------------------------------------------------------===#
    # Memory operations
    # ===-------------------------------------------------------------------===#

    def alloc_sync(
        self, context: ContextHandle, byte_size: UInt64
    ) raises HALError -> MemoryHandle:
        var mem = MaybeUninit[MemoryHandle]()
        var status = self._raw.base.memory_alloc_sync.get()(
            context, byte_size, OutParam[MemoryHandle](to=mem)
        )
        if status != STATUS_SUCCESS:
            var err = self.get_status_message(status)
            raise HALError(
                err.status,
                message=String(t"failed to alloc_sync: {err.message}"),
            )
        return mem.unsafe_assume_init()

    def free_sync(
        self,
        context: ContextHandle,
        mem: MemoryHandle,
    ) raises HALError:
        var status = self._raw.base.memory_free_sync.get()(context, mem)
        if status != STATUS_SUCCESS:
            var err = self.get_status_message(status)
            raise HALError(
                err.status,
                message=String(t"failed to free_sync: {err.message}"),
            )

    def wrap_memory(
        self,
        context: ContextHandle,
        address: UInt64,
        byte_size: UInt64,
        owning: Bool,
    ) raises HALError -> MemoryHandle:
        var mem = MaybeUninit[MemoryHandle]()
        var status = self._raw.base.memory_wrap.get()(
            context, address, byte_size, owning, OutParam[MemoryHandle](to=mem)
        )
        if status != STATUS_SUCCESS:
            var err = self.get_status_message(status)
            raise HALError(
                err.status,
                message=String(t"failed to wrap memory: {err.message}"),
            )
        return mem.unsafe_assume_init()

    def unwrap_memory(
        self, context: ContextHandle, mem: MemoryHandle
    ) raises HALError -> UInt64:
        var address = MaybeUninit[UInt64]()
        var status = self._raw.base.memory_unwrap.get()(
            context, mem, OutParam[UInt64](to=address)
        )
        if status != STATUS_SUCCESS:
            var err = self.get_status_message(status)
            raise HALError(
                err.status,
                message=String(t"failed to unwrap memory: {err.message}"),
            )
        return address.unsafe_assume_init()

    def alloc_pinned(
        self, context: ContextHandle, byte_size: UInt64
    ) raises HALError -> MemoryHandle:
        var mem = MaybeUninit[MemoryHandle]()
        var status = self._raw.base.memory_alloc_pinned.get()(
            context, byte_size, OutParam[MemoryHandle](to=mem)
        )
        if status != STATUS_SUCCESS:
            var err = self.get_status_message(status)
            raise HALError(
                err.status,
                message=String(t"failed to alloc_pinned: {err.message}"),
            )
        return mem.unsafe_assume_init()

    def free_pinned(
        self,
        context: ContextHandle,
        mem: MemoryHandle,
    ) raises HALError:
        var status = self._raw.base.memory_free_pinned.get()(context, mem)
        if status != STATUS_SUCCESS:
            var err = self.get_status_message(status)
            raise HALError(
                err.status,
                message=String(t"failed to free_pinned: {err.message}"),
            )

    def get_memory_property[
        name: StringLiteral, T: TrivialRegisterPassable
    ](self, mem: MemoryHandle) raises HALError -> T:
        """Query a named property of a device memory allocation."""
        var value = MaybeUninit[T]()
        var status = self._raw.base.memory_property.get()(
            mem,
            name.as_c_string_span(),
            OutParam[T](to=value).unsafe_bitcast[NoneType](),
        )
        if status != STATUS_SUCCESS:
            var err = self.get_status_message(status)
            raise HALError(
                err.status,
                message=String(
                    t"failed to get memory property '{name}': {err.message}"
                ),
            )
        return value.unsafe_assume_init()

    def get_queue_property[
        name: StringLiteral, T: TrivialRegisterPassable
    ](self, queue: QueueHandle) raises HALError -> T:
        """Query a named property of a queue."""
        var value = MaybeUninit[T]()
        var status = self._raw.base.queue_property.get()(
            queue,
            name.as_c_string_span(),
            OutParam[T](to=value).unsafe_bitcast[NoneType](),
        )
        if status != STATUS_SUCCESS:
            var err = self.get_status_message(status)
            raise HALError(
                err.status,
                message=String(
                    t"failed to get queue property '{name}': {err.message}"
                ),
            )
        return value.unsafe_assume_init()

    def get_optional_queue_property[
        name: StringLiteral, T: TrivialRegisterPassable
    ](self, queue: QueueHandle) raises HALError -> OptionalReg[T]:
        """Queries a queue property, returning None if the plugin does not
        expose it (`UNKNOWN_PROPERTY_NAME`) instead of raising. Any other
        failure still raises."""
        var value = MaybeUninit[T]()
        var status = self._raw.base.queue_property.get()(
            queue,
            name.as_c_string_span(),
            OutParam[T](to=value).unsafe_bitcast[NoneType](),
        )
        if status == STATUS_UNKNOWN_PROPERTY_NAME:
            return OptionalReg[T]()
        if status != STATUS_SUCCESS:
            var err = self.get_status_message(status)
            raise HALError(
                err.status,
                message=String(
                    t"failed to get queue property '{name}': {err.message}"
                ),
            )
        return OptionalReg[T](value.unsafe_assume_init())

    def get_device_property[
        name: StringLiteral, T: TrivialRegisterPassable
    ](self, device: DeviceHandle) raises HALError -> T:
        """Query a named property of a device."""
        var value = MaybeUninit[T]()
        var status = self._raw.base.device_property.get()(
            device,
            name.as_c_string_span(),
            OutParam[T](to=value).unsafe_bitcast[NoneType](),
        )
        if status != STATUS_SUCCESS:
            var err = self.get_status_message(status)
            raise HALError(
                err.status,
                message=String(
                    t"failed to get device property '{name}': {err.message}"
                ),
            )
        return value.unsafe_assume_init()

    def get_optional_device_property[
        name: StaticString, T: TrivialRegisterPassable
    ](self, device: DeviceHandle) raises HALError -> OptionalReg[T]:
        """Like `get_device_property`, but returns None when the plugin does
        not expose `name`."""
        var value = MaybeUninit[T]()
        var status = self._raw.base.device_property.get()(
            device,
            name.as_c_string_span(),
            OutParam[T](to=value).unsafe_bitcast[NoneType](),
        )
        if status == STATUS_UNKNOWN_PROPERTY_NAME:
            return OptionalReg[T]()
        if status != STATUS_SUCCESS:
            var err = self.get_status_message(status)
            raise HALError(
                err.status,
                message=String(
                    t"failed to get device property '{name}': {err.message}"
                ),
            )
        return OptionalReg[T](value.unsafe_assume_init())

    def get_device_property_string[
        origin: MutOrigin, //, name: StringLiteral
    ](self, device: DeviceHandle, buf: Pointer[Int8, origin]) raises HALError:
        """Queries a string-valued device property into a caller-supplied
        buffer of at least 256 bytes."""
        var status = self._raw.base.device_property.get()(
            device, name.as_c_string_span(), buf.unsafe_bitcast[NoneType]()
        )
        if status != STATUS_SUCCESS:
            var err = self.get_status_message(status)
            raise HALError(
                err.status,
                message=String(
                    t"failed to get device property '{name}': {err.message}"
                ),
            )

    # ===-------------------------------------------------------------------===#
    # Copy operations
    # ===-------------------------------------------------------------------===#

    def copy_to_device(
        self,
        queue: QueueHandle,
        dst: M_driver_memory_view,
        src: Pointer[mut=False, UInt8, _],
    ) raises HALError:
        var status = self._raw.base.queue_copy_to_device.get()(
            queue,
            ImmPointer(to=dst),
            src,
        )
        if status != STATUS_SUCCESS:
            var err = self.get_status_message(status)
            raise HALError(
                err.status,
                message=String(t"failed to copy to device: {err.message}"),
            )

    def copy_from_device(
        self,
        queue: QueueHandle,
        dst: Pointer[mut=True, UInt8, _],
        src: M_driver_memory_view,
    ) raises HALError:
        var status = self._raw.base.queue_copy_from_device.get()(
            queue,
            dst,
            ImmPointer(to=src),
        )
        if status != STATUS_SUCCESS:
            var err = self.get_status_message(status)
            raise HALError(
                err.status,
                message=String(t"failed to copy from device: {err.message}"),
            )

    def copy_intra_device(
        self,
        queue: QueueHandle,
        dst: M_driver_memory_view,
        src: M_driver_memory_view,
    ) raises HALError:
        var status = self._raw.base.queue_copy_intra_device.get()(
            queue,
            ImmPointer(to=dst),
            ImmPointer(to=src),
        )
        if status != STATUS_SUCCESS:
            var err = self.get_status_message(status)
            raise HALError(
                err.status,
                message=String(t"failed to copy intra-device: {err.message}"),
            )

    def copy_inter_device(
        self,
        queue: QueueHandle,
        dst: M_driver_memory_view,
        src: M_driver_memory_view,
        src_context: ContextHandle,
    ) raises HALError:
        var status = self._raw.base.queue_copy_inter_device.get()(
            queue,
            ImmPointer(to=dst),
            ImmPointer(to=src),
            src_context,
        )
        if status != STATUS_SUCCESS:
            var err = self.get_status_message(status)
            raise HALError(
                err.status,
                message=String(t"failed to copy inter-device: {err.message}"),
            )

    def copy_to_device_sync(
        self,
        context: ContextHandle,
        dst: M_driver_memory_view,
        src: Pointer[mut=False, UInt8, _],
    ) raises HALError:
        var status = self._raw.base.copy_to_device_sync.get()(
            context, ImmPointer(to=dst), src
        )
        if status != STATUS_SUCCESS:
            var err = self.get_status_message(status)
            raise HALError(
                err.status,
                message=String(
                    t"failed to copy to device (sync): {err.message}"
                ),
            )

    def copy_from_device_sync(
        self,
        context: ContextHandle,
        dst: Pointer[mut=True, UInt8, _],
        src: M_driver_memory_view,
    ) raises HALError:
        var status = self._raw.base.copy_from_device_sync.get()(
            context, dst, ImmPointer(to=src)
        )
        if status != STATUS_SUCCESS:
            var err = self.get_status_message(status)
            raise HALError(
                err.status,
                message=String(
                    t"failed to copy from device (sync): {err.message}"
                ),
            )

    def copy_intra_device_sync(
        self,
        context: ContextHandle,
        dst: M_driver_memory_view,
        src: M_driver_memory_view,
    ) raises HALError:
        var status = self._raw.base.copy_intra_device_sync.get()(
            context, ImmPointer(to=dst), ImmPointer(to=src)
        )
        if status != STATUS_SUCCESS:
            var err = self.get_status_message(status)
            raise HALError(
                err.status,
                message=String(
                    t"failed to copy intra-device (sync): {err.message}"
                ),
            )

    def copy_inter_device_sync(
        self,
        context: ContextHandle,
        dst: M_driver_memory_view,
        src: M_driver_memory_view,
        src_context: ContextHandle,
    ) raises HALError:
        var status = self._raw.base.copy_inter_device_sync.get()(
            context, ImmPointer(to=dst), ImmPointer(to=src), src_context
        )
        if status != STATUS_SUCCESS:
            var err = self.get_status_message(status)
            raise HALError(
                err.status,
                message=String(
                    t"failed to copy inter-device (sync): {err.message}"
                ),
            )

    def set_memory_sync(
        self,
        context: ContextHandle,
        dst: M_driver_memory_view,
        value: UInt8,
    ) raises HALError:
        var status = self._raw.base.set_memory_sync.get()(
            context, ImmPointer(to=dst), value
        )
        if status != STATUS_SUCCESS:
            var err = self.get_status_message(status)
            raise HALError(
                err.status,
                message=String(t"failed to set memory (sync): {err.message}"),
            )

    def fill_sync(
        self,
        context: ContextHandle,
        dst: M_driver_memory_view,
        value: UInt64,
        value_size: UInt64,
    ) raises HALError:
        var status = self._raw.base.fill_sync.get()(
            context, ImmPointer(to=dst), value, value_size
        )
        if status != STATUS_SUCCESS:
            var err = self.get_status_message(status)
            raise HALError(
                err.status,
                message=String(t"failed to fill memory (sync): {err.message}"),
            )

    def set_memory(
        self,
        queue: QueueHandle,
        dst: M_driver_memory_view,
        value: UInt8,
    ) raises HALError:
        var status = self._raw.base.queue_set_memory.get()(
            queue,
            ImmPointer(to=dst),
            value,
        )
        if status != STATUS_SUCCESS:
            var err = self.get_status_message(status)
            raise HALError(
                err.status,
                message=String(t"failed to set memory: {err.message}"),
            )

    def fill(
        self,
        queue: QueueHandle,
        dst: M_driver_memory_view,
        value: UInt64,
        value_size: UInt64,
    ) raises HALError:
        var status = self._raw.base.queue_fill.get()(
            queue,
            ImmPointer(to=dst),
            value,
            value_size,
        )
        if status != STATUS_SUCCESS:
            var err = self.get_status_message(status)
            raise HALError(
                err.status,
                message=String(t"failed to fill memory: {err.message}"),
            )

    def queue_wait_value64(
        self, queue: QueueHandle, device_address: UInt64, value: UInt64
    ) raises HALError:
        """Enqueues a wait until the 64-bit slot at `device_address` equals
        `value` (e.g. cuStreamWaitValue64 with EQ semantics)."""
        var status = self._raw.base.queue_wait_value64.get()(
            queue, device_address, value
        )
        if status != STATUS_SUCCESS:
            var err = self.get_status_message(status)
            raise HALError(
                err.status,
                message=String(t"failed to wait on value: {err.message}"),
            )

    def queue_launch_host_func[
        origin: MutOrigin
    ](
        self,
        queue: QueueHandle,
        func: def(OpaquePointer[origin]) thin -> None,
        user_data: OpaquePointer[origin],
    ) raises HALError:
        """Enqueues a host function callback on the queue (e.g.
        cuLaunchHostFunc)."""
        var status = self._raw.base.queue_launch_host_func.get()(
            queue, func, user_data
        )
        if status != STATUS_SUCCESS:
            var err = self.get_status_message(status)
            raise HALError(
                err.status,
                message=String(
                    t"failed to enqueue host function: {err.message}"
                ),
            )

    def synchronize_queue(self, queue: QueueHandle) raises HALError:
        var status = self._raw.base.queue_synchronize.get()(queue)
        if status != STATUS_SUCCESS:
            var err = self.get_status_message(status)
            raise HALError(
                err.status,
                message=String(t"failed to synchronize queue: {err.message}"),
            )

    # ===-------------------------------------------------------------------===#
    # Event operations
    # ===-------------------------------------------------------------------===#

    def create_event(
        self, context: ContextHandle, flags: UInt32
    ) raises HALError -> EventHandle:
        var event = MaybeUninit[EventHandle]()
        var status = self._raw.base.event_create.get()(
            context, flags, OutParam[EventHandle](to=event)
        )
        if status != STATUS_SUCCESS:
            var err = self.get_status_message(status)
            raise HALError(
                err.status,
                message=String(t"failed to create event: {err.message}"),
            )
        return event.unsafe_assume_init()

    def destroy_event(
        self, context: ContextHandle, event: EventHandle
    ) raises HALError:
        var status = self._raw.base.event_destroy.get()(context, event)
        if status != STATUS_SUCCESS:
            var err = self.get_status_message(status)
            raise HALError(
                err.status,
                message=String(t"failed to destroy event: {err.message}"),
            )

    def synchronize_event(
        self, context: ContextHandle, event: EventHandle
    ) raises HALError:
        var status = self._raw.base.event_synchronize.get()(context, event)
        if status != STATUS_SUCCESS:
            var err = self.get_status_message(status)
            raise HALError(
                err.status,
                message=String(t"failed to synchronize event: {err.message}"),
            )

    def is_event_ready(
        self, context: ContextHandle, event: EventHandle
    ) raises HALError -> Bool:
        var result = MaybeUninit[Bool]()
        var status = self._raw.base.is_event_ready.get()(
            context, event, OutParam[Bool](to=result)
        )
        if status != STATUS_SUCCESS:
            var err = self.get_status_message(status)
            raise HALError(
                err.status,
                message=String(
                    t"failed to query event readiness: {err.message}"
                ),
            )
        return result.unsafe_assume_init()

    def record_event(
        self, queue: QueueHandle, event: EventHandle
    ) raises HALError:
        var status = self._raw.base.queue_record_event.get()(queue, event)
        if status != STATUS_SUCCESS:
            var err = self.get_status_message(status)
            raise HALError(
                err.status,
                message=String(t"failed to record event: {err.message}"),
            )

    def wait_for_events(
        self,
        queue: QueueHandle,
        handles: Pointer[mut=True, EventHandle, _],
        num_events: UInt32,
    ) raises HALError:
        var status = self._raw.base.queue_wait_for_events.get()(
            queue, handles, num_events
        )
        if status != STATUS_SUCCESS:
            var err = self.get_status_message(status)
            raise HALError(
                err.status,
                message=String(
                    t"failed to wait for events on queue: {err.message}"
                ),
            )

    # ===-------------------------------------------------------------------===#
    # Function execution
    # ===-------------------------------------------------------------------===#

    def load_function(
        self,
        context: ContextHandle,
        bundle: RuntimeBundleHandle,
        mut name: String,
    ) raises HALError -> FunctionHandle:
        var func = MaybeUninit[FunctionHandle]()
        var status = self._raw.base.function_load.get()(
            context,
            bundle,
            name.as_c_string_span(),
            UInt64(name.byte_length()),
            OutParam[FunctionHandle](to=func),
        )
        if status != STATUS_SUCCESS:
            var err = self.get_status_message(status)
            raise HALError(
                err.status,
                message=String(
                    t"failed to load function '{name}': {err.message}"
                ),
            )
        return func.unsafe_assume_init()

    def set_function_attribute(
        self,
        context: ContextHandle,
        function: FunctionHandle,
        attribute: Int32,
        value: Int32,
    ) raises HALError:
        """Sets a backend function attribute (e.g. the dynamic shared-memory
        cap) on a loaded function."""
        var status = self._raw.base.function_set_attribute.get()(
            context, function, attribute, value
        )
        if status != STATUS_SUCCESS:
            var err = self.get_status_message(status)
            raise HALError(
                err.status,
                message=String(
                    t"failed to set function attribute: {err.message}"
                ),
            )

    def function_occupancy_max_active_blocks(
        self,
        context: ContextHandle,
        func: FunctionHandle,
        block_size: Int32,
        dynamic_shared_memory_bytes: UInt64,
    ) raises HALError -> Int32:
        var num_blocks = MaybeUninit[Int32]()
        var status = self._raw.base.function_occupancy_max_active_blocks.get()(
            context,
            func,
            block_size,
            dynamic_shared_memory_bytes,
            OutParam[Int32](to=num_blocks),
        )
        if status != STATUS_SUCCESS:
            var err = self.get_status_message(status)
            raise HALError(
                err.status,
                message=String(
                    t"failed to query function occupancy: {err.message}"
                ),
            )
        return num_blocks.unsafe_assume_init()

    def tensor_map_encode_tiled(
        self,
        context: ContextHandle,
        tensor_map: MutOpaquePointer[_],
        data_type: Int32,
        rank: Int32,
        global_device_address: UInt64,
        global_dim: Pointer[mut=False, UInt64, _],
        global_strides: Pointer[mut=False, UInt64, _],
        box_dim: Pointer[mut=False, UInt32, _],
        element_strides: Pointer[mut=False, UInt32, _],
        interleave: Int32,
        swizzle: Int32,
        l2_promotion: Int32,
        oob_fill: Int32,
    ) raises HALError:
        """Encodes a tiled TMA descriptor into `tensor_map` (128 host bytes)."""
        var status = self._raw.base.tensor_map_encode_tiled.get()(
            context,
            tensor_map,
            data_type,
            rank,
            global_device_address,
            global_dim,
            global_strides,
            box_dim,
            element_strides,
            interleave,
            swizzle,
            l2_promotion,
            oob_fill,
        )
        if status != STATUS_SUCCESS:
            var err = self.get_status_message(status)
            raise HALError(
                err.status,
                message=String(
                    t"failed to encode tiled tensor map: {err.message}"
                ),
            )

    def tensor_map_encode_im2col(
        self,
        context: ContextHandle,
        tensor_map: MutOpaquePointer[_],
        data_type: Int32,
        rank: Int32,
        global_device_address: UInt64,
        global_dim: Pointer[mut=False, UInt64, _],
        global_strides: Pointer[mut=False, UInt64, _],
        pixel_box_lower_corner: Pointer[mut=False, Int32, _],
        pixel_box_upper_corner: Pointer[mut=False, Int32, _],
        channels_per_pixel: Int32,
        pixels_per_column: Int32,
        element_strides: Pointer[mut=False, UInt32, _],
        interleave: Int32,
        swizzle: Int32,
        l2_promotion: Int32,
        oob_fill: Int32,
    ) raises HALError:
        """Encodes an im2col TMA descriptor into `tensor_map`."""
        var status = self._raw.base.tensor_map_encode_im2col.get()(
            context,
            tensor_map,
            data_type,
            rank,
            global_device_address,
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
        if status != STATUS_SUCCESS:
            var err = self.get_status_message(status)
            raise HALError(
                err.status,
                message=String(
                    t"failed to encode im2col tensor map: {err.message}"
                ),
            )

    def unload_function(
        self,
        context: ContextHandle,
        func: FunctionHandle,
    ) raises HALError:
        var status = self._raw.base.function_unload.get()(context, func)
        if status != STATUS_SUCCESS:
            var err = self.get_status_message(status)
            raise HALError(
                err.status,
                message=String(t"failed to unload function: {err.message}"),
            )

    def execute_function(
        self,
        queue: QueueHandle,
        func: FunctionHandle,
        grid: Tuple[UInt32, UInt32, UInt32],
        block: Tuple[UInt32, UInt32, UInt32],
        args: Pointer[mut=True, OpaquePointer[MutUntrackedOrigin], _],
        arg_sizes: Pointer[mut=True, UInt64, _],
        num_args: UInt32,
        shared_mem_bytes: UInt32 = 0,
        attributes: OptionalReg[OpaquePointer[MutUntrackedOrigin]] = None,
        num_attributes: UInt32 = 0,
    ) raises HALError:
        var config = M_driver_queue_execute_config[MutUntrackedOrigin](
            mode=M_driver_queue_execute_mode.GPU,
            config=UnsafeUnion[
                M_driver_queue_execute_config_gpu[MutUntrackedOrigin]
            ](
                M_driver_queue_execute_config_gpu[MutUntrackedOrigin](
                    grid=M_driver_dim(x=grid[0], y=grid[1], z=grid[2]),
                    block=M_driver_dim(x=block[0], y=block[1], z=block[2]),
                    shared_mem_bytes=shared_mem_bytes,
                    attributes=attributes,
                    num_attributes=num_attributes,
                )
            ),
        )
        var status = self._raw.base.queue_execute.get()(
            queue,
            func,
            Pointer(to=config),
            args,
            arg_sizes,
            num_args,
        )
        if status != STATUS_SUCCESS:
            var err = self.get_status_message(status)
            raise HALError(
                err.status,
                message=String(t"failed to execute function: {err.message}"),
            )

    def get_api_name(self) raises HALError -> String:
        """Retrieve the plugin-reported API name (e.g. "CUDA", "Metal")."""
        var value = MaybeUninit[OpaquePointer[ImmUntrackedOrigin]]()
        var status = self._raw.base.property.get()(
            self._driver_handle,
            "api".as_c_string_span(),
            OutParam[OpaquePointer[ImmUntrackedOrigin]](to=value),
        )
        if status != STATUS_SUCCESS:
            var err = self.get_status_message(status)
            raise HALError(
                err.status,
                message=String(
                    t"failed to get driver 'api' property: {err.message}"
                ),
            )
        return String(
            CStringSpan(
                unsafe_from_ptr=value.unsafe_assume_init().unsafe_bitcast[
                    c_char
                ]()
            )
        )

    # ===-------------------------------------------------------------------===#
    # Status
    # ===-------------------------------------------------------------------===#

    def get_status_message(self, status: Int64) raises HALError -> HALError:
        """Retrieve the human-readable message associated with a status code."""
        var buf = List[Int8](length=1024, fill=0)
        var ret = self._raw.base.status_message.get()(
            self._driver_handle,
            status,
            MutPointer(to=buf.unsafe_ptr()[]),
            Int64(len(buf)),
        )

        if ret == STATUS_SUCCESS:
            try:
                return HALError(
                    status,
                    message=CStringSpan(
                        unsafe_from_ptr=Pointer(to=buf[0]).unsafe_bitcast[
                            c_char
                        ]()
                    ),
                )
            except:
                return HALError(
                    status, message="(failed to decode status message)"
                )
        else:
            return HALError(
                status,
                message="(problem retrieving status message)",
            )
