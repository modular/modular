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
"""HAL plugin loader.

Loads a HAL plugin shared library and resolves all required function pointers
from the M_driver_* C API.
"""

from std.collections import OptionalReg

from std.ffi import (
    _DLHandle,
    OwnedDLHandle,
    RTLD,
    CStringSlice,
    UnsafeUnion,
    c_uchar,
    c_char,
    c_uint,
    c_int,
)

from std.memory import (
    alloc,
    ImmutPointer,
    MutPointer,
    OpaquePointer,
    UnsafePointer,
    UnsafeMaybeUninit,
)

from .status import STATUS_SUCCESS, STATUS_UNKNOWN_ERROR, HALError
from .event import Event
from .device import DeviceSpec

# ===-----------------------------------------------------------------------===#
# Shared plugin structs across FFI
# ===-----------------------------------------------------------------------===#


@fieldwise_init
struct M_driver_slice(TrivialRegisterPassable):
    var data: ImmutPointer[UInt8, ImmutAnyOrigin]
    var size: UInt64


@fieldwise_init
struct M_driver_static_bundle(TrivialRegisterPassable):
    var mapped_data: M_driver_slice
    var file_type: ImmutPointer[Int8, ImmutAnyOrigin]
    var file_type_len: UInt64


@fieldwise_init
struct M_driver_dim(TrivialRegisterPassable):
    var x: UInt32
    var y: UInt32
    var z: UInt32


@fieldwise_init
struct M_driver_queue_execute_config_gpu(TrivialRegisterPassable):
    var grid: M_driver_dim
    var block: M_driver_dim
    var shared_mem_bytes: UInt32
    var attributes: OptionalReg[OpaquePointer[MutExternalOrigin]]
    var num_attributes: UInt32


@fieldwise_init
struct M_driver_queue_execute_mode(TrivialRegisterPassable):
    var value: Int32

    comptime GPU = Self(value=0)


@fieldwise_init
struct M_driver_queue_execute_config:
    var mode: M_driver_queue_execute_mode
    var config: UnsafeUnion[M_driver_queue_execute_config_gpu]


@fieldwise_init
struct M_driver_bundle_compilation_options(TrivialRegisterPassable):
    var debug_level: ImmutPointer[Int8, ImmutAnyOrigin]
    var debug_level_len: UInt64
    var optimization_level: Int32


# ===-----------------------------------------------------------------------===#
# Opaque plugin structs for handles.
# ===-----------------------------------------------------------------------===#


struct M_driver_driver:
    pass


struct M_driver_device:
    pass


struct M_driver_context:
    pass


struct M_driver_queue:
    pass


struct M_driver_event:
    pass


struct M_driver_function:
    pass


struct M_driver_memory:
    pass


struct M_driver_runtime_bundle:
    pass


# ===-----------------------------------------------------------------------===#
# Version negotiation struct — must match C layout of M_driver_version.
# ===-----------------------------------------------------------------------===#

comptime M_DRIVER_INTERFACE_VERSION_MAJOR: UInt32 = 0
comptime M_DRIVER_INTERFACE_VERSION_MINOR: UInt32 = 1
comptime M_DRIVER_INTERFACE_VERSION_PATCH: UInt32 = 0


@fieldwise_init
struct DriverVersion(TrivialRegisterPassable):
    """Mirrors C `M_driver_version { uint32_t major, minor, patch; }`."""

    var major: UInt32
    var minor: UInt32
    var patch: UInt32


# ===-----------------------------------------------------------------------===#
# C API function pointer types
# ===-----------------------------------------------------------------------===#

comptime Handle[T: AnyType] = UnsafePointer[T, MutExternalOrigin]
comptime OutParam[T: TrivialRegisterPassable] = UnsafePointer[
    UnsafeMaybeUninit[T], MutAnyOrigin
]

comptime DriverHandle = Handle[M_driver_driver]
comptime DeviceHandle = Handle[M_driver_device]
comptime ContextHandle = Handle[M_driver_context]
comptime QueueHandle = Handle[M_driver_queue]
comptime EventHandle = Handle[M_driver_event]
comptime FunctionHandle = Handle[M_driver_function]
comptime MemoryHandle = Handle[M_driver_memory]
comptime StaticBundleHandle = Handle[M_driver_static_bundle]
comptime RuntimeBundleHandle = Handle[M_driver_runtime_bundle]
comptime ExecuteConfigHandle = Handle[M_driver_queue_execute_config]
comptime CompilationOptionsHandle = Handle[M_driver_bundle_compilation_options]

comptime PluginResultCode = Int64


struct HALFunction[name: StaticString, fn_type: TrivialRegisterPassable](
    Copyable, Movable
):
    var f: Self.fn_type

    def __init__(out self, lib: _DLHandle, so_path: StringSlice):
        self.f = lib._get_function[Self.name, Self.fn_type]()

    # TODO: figure out if we can do fn type reflection to be able to do
    # typewise variadic call through to the fn ptr
    # @always_inline
    # def __call__(self, *args: *_) -> _:
    #    return self.f(*args)


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

    def __del__(deinit self):
        # Move `_plugin` into a local so its `OwnedDLHandle` stays alive
        # until after the `destroy` call returns — ASAP destruction would
        # otherwise `dlclose` the `.so` (unmapping the code page) before
        # we call through the function pointer.
        var plugin = self._raw^
        var status = plugin.destroy.f(self._driver_handle)
        if status != STATUS_SUCCESS:
            print("warning: driver destroy failed with status:", status)
        _ = plugin^

    # TODO: `HALFunction` should, eventually, be able to
    # do the vast majority of this automagically with a _partially_ bound
    # thin fn type that can ensure safety at call site, and a rebind
    # to some concrete type internally.
    # See DRIV-6, MOCO-3661

    def get_device_count(mut self) raises HALError -> Int64:
        var num_devices = UnsafeMaybeUninit(Int64(0))
        var status = self._raw.device_count.f(
            self._driver_handle, OutParam[Int64](to=num_devices)
        )

        if status != STATUS_SUCCESS:
            var err = self.get_status_message(status)
            raise err^

        return num_devices.unsafe_assume_init_ref()

    # ===-------------------------------------------------------------------===#
    # Queue operations
    # ===-------------------------------------------------------------------===#

    def create_queue(
        self, context: ContextHandle
    ) raises HALError -> QueueHandle:
        var queue = UnsafeMaybeUninit[QueueHandle]()
        var status = self._raw.queue_create.f(
            context, OutParam[QueueHandle](to=queue)
        )
        if status != STATUS_SUCCESS:
            var err = self.get_status_message(status)
            raise HALError(
                err.status,
                message=String(t"failed to create queue: {err.message}"),
            )
        return queue.unsafe_assume_init_ref()

    def destroy_queue(
        self,
        context: ContextHandle,
        queue: QueueHandle,
    ) raises HALError:
        var status = self._raw.queue_destroy.f(context, queue)
        if status != STATUS_SUCCESS:
            var err = self.get_status_message(status)
            raise HALError(
                err.status,
                message=String(t"failed to destroy queue: {err.message}"),
            )

    # ===-------------------------------------------------------------------===#
    # Memory operations
    # ===-------------------------------------------------------------------===#

    def alloc_sync(
        self, context: ContextHandle, byte_size: UInt64
    ) raises HALError -> MemoryHandle:
        var mem = UnsafeMaybeUninit[MemoryHandle]()
        var status = self._raw.memory_alloc_sync.f(
            context, byte_size, OutParam[MemoryHandle](to=mem)
        )
        if status != STATUS_SUCCESS:
            var err = self.get_status_message(status)
            raise HALError(
                err.status,
                message=String(t"failed to alloc_sync: {err.message}"),
            )
        return mem.unsafe_assume_init_ref()

    def free_sync(
        self,
        context: ContextHandle,
        mem: MemoryHandle,
    ) raises HALError:
        var status = self._raw.memory_free_sync.f(context, mem)
        if status != STATUS_SUCCESS:
            var err = self.get_status_message(status)
            raise HALError(
                err.status,
                message=String(t"failed to free_sync: {err.message}"),
            )

    def get_memory_property[
        name: StringLiteral, T: TrivialRegisterPassable
    ](self, mem: MemoryHandle) raises HALError -> T:
        """Query a named property of a device memory allocation."""
        var value = UnsafeMaybeUninit[T]()
        var status = self._raw.memory_property.f(
            mem,
            rebind[CStringSlice[ImmutAnyOrigin]](name.as_c_string_slice()),
            rebind[OpaquePointer[MutAnyOrigin]](OutParam[T](to=value)),
        )
        if status != STATUS_SUCCESS:
            var err = self.get_status_message(status)
            raise HALError(
                err.status,
                message=String(
                    t"failed to get memory property '{name}': {err.message}"
                ),
            )
        return value.unsafe_assume_init_ref()

    # ===-------------------------------------------------------------------===#
    # Copy operations
    # ===-------------------------------------------------------------------===#

    def copy_to_device(
        self,
        queue: QueueHandle,
        dst: MemoryHandle,
        src: UnsafePointer[UInt8, MutAnyOrigin],
        size: UInt64,
    ) raises HALError:
        var status = self._raw.queue_copy_to_device.f(queue, dst, src, size)
        if status != STATUS_SUCCESS:
            var err = self.get_status_message(status)
            raise HALError(
                err.status,
                message=String(t"failed to copy to device: {err.message}"),
            )

    def copy_from_device(
        self,
        queue: QueueHandle,
        dst: UnsafePointer[UInt8, MutAnyOrigin],
        src: MemoryHandle,
        size: UInt64,
    ) raises HALError:
        var status = self._raw.queue_copy_from_device.f(queue, dst, src, size)
        if status != STATUS_SUCCESS:
            var err = self.get_status_message(status)
            raise HALError(
                err.status,
                message=String(t"failed to copy from device: {err.message}"),
            )

    def copy_device_to_device(
        self,
        queue: QueueHandle,
        dst: MemoryHandle,
        src: MemoryHandle,
        size: UInt64,
    ) raises HALError:
        var status = self._raw.queue_copy_device_to_device.f(
            queue, dst, src, size
        )
        if status != STATUS_SUCCESS:
            var err = self.get_status_message(status)
            raise HALError(
                err.status,
                message=String(
                    t"failed to copy device to device: {err.message}"
                ),
            )

    def synchronize_queue(self, queue: QueueHandle) raises HALError:
        var status = self._raw.queue_synchronize.f(queue)
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
        self, context: ContextHandle
    ) raises HALError -> EventHandle:
        var event = UnsafeMaybeUninit[EventHandle]()
        var status = self._raw.event_create.f(
            context, OutParam[EventHandle](to=event)
        )
        if status != STATUS_SUCCESS:
            var err = self.get_status_message(status)
            raise HALError(
                err.status,
                message=String(t"failed to create event: {err.message}"),
            )
        return event.unsafe_assume_init_ref()

    def destroy_event(
        self, context: ContextHandle, event: EventHandle
    ) raises HALError:
        var status = self._raw.event_destroy.f(context, event)
        if status != STATUS_SUCCESS:
            var err = self.get_status_message(status)
            raise HALError(
                err.status,
                message=String(t"failed to destroy event: {err.message}"),
            )

    def synchronize_event(
        self, context: ContextHandle, event: EventHandle
    ) raises HALError:
        var status = self._raw.event_synchronize.f(context, event)
        if status != STATUS_SUCCESS:
            var err = self.get_status_message(status)
            raise HALError(
                err.status,
                message=String(t"failed to synchronize event: {err.message}"),
            )

    def is_event_ready(
        self, context: ContextHandle, event: EventHandle
    ) raises HALError -> Bool:
        var result = UnsafeMaybeUninit[Bool]()
        var status = self._raw.is_event_ready.f(
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
        return result.unsafe_assume_init_ref()

    def record_event(
        self, queue: QueueHandle, event: EventHandle
    ) raises HALError:
        var status = self._raw.queue_record_event.f(queue, event)
        if status != STATUS_SUCCESS:
            var err = self.get_status_message(status)
            raise HALError(
                err.status,
                message=String(t"failed to record event: {err.message}"),
            )

    def wait_for_events[
        origin: ImmutOrigin
    ](
        self,
        queue: QueueHandle,
        read events: List[Event[origin]],
    ) raises HALError:
        var handles = List[EventHandle](capacity=len(events))
        for ref event in events:
            handles.append(event._inner[]._handle)
        var status = self._raw.queue_wait_for_events.f(
            queue, handles.unsafe_ptr(), UInt32(len(handles))
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
        var func = UnsafeMaybeUninit[FunctionHandle]()
        var status = self._raw.function_load.f(
            context,
            bundle,
            rebind[CStringSlice[ImmutAnyOrigin]](name.as_c_string_slice()),
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
        return func.unsafe_assume_init_ref()

    def unload_function(
        self,
        context: ContextHandle,
        func: FunctionHandle,
    ) raises HALError:
        var status = self._raw.function_unload.f(context, func)
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
        args: UnsafePointer[OpaquePointer[MutExternalOrigin], MutAnyOrigin],
        arg_sizes: UnsafePointer[UInt64, MutAnyOrigin],
        num_args: UInt32,
    ) raises HALError:
        var config = M_driver_queue_execute_config(
            mode=M_driver_queue_execute_mode.GPU,
            config=UnsafeUnion[M_driver_queue_execute_config_gpu](
                M_driver_queue_execute_config_gpu(
                    grid=M_driver_dim(x=grid[0], y=grid[1], z=grid[2]),
                    block=M_driver_dim(x=block[0], y=block[1], z=block[2]),
                    shared_mem_bytes=UInt32(0),
                    attributes={},
                    num_attributes=UInt32(0),
                )
            ),
        )
        var status = self._raw.queue_execute.f(
            queue,
            func,
            rebind[ExecuteConfigHandle](UnsafePointer(to=config)),
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

    # ===-------------------------------------------------------------------===#
    # Status
    # ===-------------------------------------------------------------------===#

    def get_status_message(self, status: Int64) raises HALError -> HALError:
        """Retrieve the human-readable message associated with a status code."""
        var buf = List[Int8](length=1024, fill=0)
        var ret = self._raw.status_message.f(
            self._driver_handle,
            status,
            MutPointer(
                to=UnsafePointer[Int8, MutAnyOrigin](buf.unsafe_ptr())[]
            ),
            Int64(len(buf)),
        )

        if ret == STATUS_SUCCESS:
            try:
                return HALError(
                    status,
                    message=CStringSlice(
                        unsafe_from_ptr=UnsafePointer(to=buf[0])
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


@fieldwise_init
struct RawPlugin(Movable):
    """Loaded HAL plugin with resolved function pointers.

    Owns the dlopen'd shared library handle and all resolved symbols.
    Dropping this struct dlcloses the library.
    """

    var _lib: OwnedDLHandle
    var name: String
    var so_path: String

    # Resolved function pointers.
    var status_message: HALFunction[
        "M_driver_status_message",
        def(
            driver: DriverHandle,
            status: Int64,
            message_buffer: MutPointer[Int8, MutAnyOrigin],
            message_buffer_size: Int64,
        ) thin -> PluginResultCode,
    ]
    var create: HALFunction[
        "M_driver_create",
        def(
            version: ImmutPointer[DriverVersion, ImmutAnyOrigin],
            driver: OutParam[DriverHandle],
        ) thin -> PluginResultCode,
    ]
    var destroy: HALFunction[
        "M_driver_destroy",
        def(driver: DriverHandle) thin -> PluginResultCode,
    ]
    var property: HALFunction[
        "M_driver_property",
        def(
            handle: DriverHandle,
            property_name: CStringSlice[ImmutAnyOrigin],
            value: OutParam[OpaquePointer[ImmutExternalOrigin]],
        ) thin -> PluginResultCode,
    ]
    var device_count: HALFunction[
        "M_driver_device_count",
        def(
            driver: DriverHandle, count: OutParam[Int64]
        ) thin -> PluginResultCode,
    ]
    var device_get: HALFunction[
        "M_driver_device_get",
        def(
            driver: DriverHandle,
            device_id: Int64,
            device: OutParam[DeviceHandle],
        ) thin -> PluginResultCode,
    ]
    var context_create: HALFunction[
        "M_driver_context_create",
        def(
            device: DeviceHandle, context: OutParam[ContextHandle]
        ) thin -> PluginResultCode,
    ]
    var context_destroy: HALFunction[
        "M_driver_context_destroy",
        def(context: ContextHandle) thin -> PluginResultCode,
    ]
    var memory_alloc_pinned: HALFunction[
        "M_driver_memory_alloc_pinned",
        def(
            context: ContextHandle,
            size: UInt64,
            memory: OutParam[MemoryHandle],
        ) thin -> PluginResultCode,
    ]
    var memory_free_pinned: HALFunction[
        "M_driver_memory_free_pinned",
        def(
            context: ContextHandle,
            memory: MemoryHandle,
        ) thin -> PluginResultCode,
    ]
    var memory_alloc_sync: HALFunction[
        "M_driver_memory_alloc_sync",
        def(
            context: ContextHandle,
            size: UInt64,
            memory: OutParam[MemoryHandle],
        ) thin -> PluginResultCode,
    ]
    var memory_free_sync: HALFunction[
        "M_driver_memory_free_sync",
        def(
            context: ContextHandle,
            memory: MemoryHandle,
        ) thin -> PluginResultCode,
    ]
    var memory_alloc_async: HALFunction[
        "M_driver_memory_alloc_async",
        def(
            queue: QueueHandle,
            size: UInt64,
            memory: OutParam[MemoryHandle],
        ) thin -> PluginResultCode,
    ]
    var memory_free_async: HALFunction[
        "M_driver_memory_free_async",
        def(
            queue: QueueHandle,
            memory: MemoryHandle,
        ) thin -> PluginResultCode,
    ]
    var queue_create: HALFunction[
        "M_driver_queue_create",
        def(
            context: ContextHandle, queue: OutParam[QueueHandle]
        ) thin -> PluginResultCode,
    ]
    var queue_destroy: HALFunction[
        "M_driver_queue_destroy",
        def(
            context: ContextHandle, queue: QueueHandle
        ) thin -> PluginResultCode,
    ]
    var queue_copy_to_device: HALFunction[
        "M_driver_queue_copy_to_device",
        def(
            queue: QueueHandle,
            dst: MemoryHandle,
            src: UnsafePointer[UInt8, MutAnyOrigin],
            size: UInt64,
        ) thin -> PluginResultCode,
    ]
    var queue_copy_from_device: HALFunction[
        "M_driver_queue_copy_from_device",
        def(
            queue: QueueHandle,
            dst: UnsafePointer[UInt8, MutAnyOrigin],
            src: MemoryHandle,
            size: UInt64,
        ) thin -> PluginResultCode,
    ]
    var queue_copy_device_to_device: HALFunction[
        "M_driver_queue_copy_device_to_device",
        def(
            queue: QueueHandle,
            dst: MemoryHandle,
            src: MemoryHandle,
            size: UInt64,
        ) thin -> PluginResultCode,
    ]
    var queue_set_memory: HALFunction[
        "M_driver_queue_set_memory",
        def(
            queue: QueueHandle,
            dst: MemoryHandle,
            size: UInt64,
            value: UInt64,
            value_size: UInt64,
        ) thin -> PluginResultCode,
    ]
    var event_create: HALFunction[
        "M_driver_event_create",
        def(
            context: ContextHandle, event: OutParam[EventHandle]
        ) thin -> PluginResultCode,
    ]
    var event_destroy: HALFunction[
        "M_driver_event_destroy",
        def(
            context: ContextHandle, event: EventHandle
        ) thin -> PluginResultCode,
    ]
    var event_synchronize: HALFunction[
        "M_driver_event_synchronize",
        def(
            context: ContextHandle, event: EventHandle
        ) thin -> PluginResultCode,
    ]
    var is_event_ready: HALFunction[
        "M_driver_is_event_ready",
        def(
            context: ContextHandle,
            event: EventHandle,
            is_ready: OutParam[Bool],
        ) thin -> PluginResultCode,
    ]
    var function_load: HALFunction[
        "M_driver_function_load",
        def(
            context: ContextHandle,
            bundle: RuntimeBundleHandle,
            function_name: CStringSlice[ImmutAnyOrigin],
            function_name_len: UInt64,
            function: OutParam[FunctionHandle],
        ) thin -> PluginResultCode,
    ]
    var function_unload: HALFunction[
        "M_driver_function_unload",
        def(
            context: ContextHandle, function: FunctionHandle
        ) thin -> PluginResultCode,
    ]
    var device_property: HALFunction[
        "M_driver_device_property",
        def(
            device: DeviceHandle,
            property_name: CStringSlice[ImmutAnyOrigin],
            value: OpaquePointer[MutAnyOrigin],
        ) thin -> PluginResultCode,
    ]
    var memory_property: HALFunction[
        "M_driver_memory_property",
        def(
            memory: MemoryHandle,
            property_name: CStringSlice[ImmutAnyOrigin],
            value: OpaquePointer[MutAnyOrigin],
        ) thin -> PluginResultCode,
    ]
    var queue_execute: HALFunction[
        "M_driver_queue_execute",
        def(
            queue: QueueHandle,
            function: FunctionHandle,
            config: ExecuteConfigHandle,
            args: UnsafePointer[OpaquePointer[MutExternalOrigin], MutAnyOrigin],
            arg_sizes: UnsafePointer[UInt64, MutAnyOrigin],
            num_args: UInt32,
        ) thin -> PluginResultCode,
    ]
    var queue_record_event: HALFunction[
        "M_driver_queue_record_event",
        def(queue: QueueHandle, event: EventHandle) thin -> PluginResultCode,
    ]
    var queue_wait_for_events: HALFunction[
        "M_driver_queue_wait_for_events",
        def(
            queue: QueueHandle,
            events: UnsafePointer[EventHandle, MutAnyOrigin],
            num_events: UInt32,
        ) thin -> PluginResultCode,
    ]
    var queue_synchronize: HALFunction[
        "M_driver_queue_synchronize",
        def(queue: QueueHandle) thin -> PluginResultCode,
    ]
    var queue_is_stream: HALFunction[
        "M_driver_queue_is_stream",
        def(
            queue: QueueHandle, is_stream: OutParam[Bool]
        ) thin -> PluginResultCode,
    ]
    var bundle_load: HALFunction[
        "M_driver_bundle_load",
        def(
            context: ContextHandle,
            bundle: StaticBundleHandle,
            opts: CompilationOptionsHandle,
            runtime_bundle: OutParam[RuntimeBundleHandle],
        ) thin -> PluginResultCode,
    ]
    var bundle_unload: HALFunction[
        "M_driver_bundle_unload",
        def(
            context: ContextHandle,
            bundle: RuntimeBundleHandle,
        ) thin -> PluginResultCode,
    ]

    def __init__(
        out self, var lib: OwnedDLHandle, so_path: String
    ) raises HALError:
        self._lib = lib^
        self.name = so_path
        self.so_path = so_path
        var handle = self._lib.borrow()
        self.status_message = type_of(self.status_message)(handle, so_path)
        self.create = type_of(self.create)(handle, so_path)
        self.destroy = type_of(self.destroy)(handle, so_path)
        self.property = type_of(self.property)(handle, so_path)
        self.device_count = type_of(self.device_count)(handle, so_path)
        self.device_get = type_of(self.device_get)(handle, so_path)
        self.context_create = type_of(self.context_create)(handle, so_path)
        self.context_destroy = type_of(self.context_destroy)(handle, so_path)
        self.memory_alloc_pinned = type_of(self.memory_alloc_pinned)(
            handle, so_path
        )
        self.memory_free_pinned = type_of(self.memory_free_pinned)(
            handle, so_path
        )
        self.memory_alloc_sync = type_of(self.memory_alloc_sync)(
            handle, so_path
        )
        self.memory_free_sync = type_of(self.memory_free_sync)(handle, so_path)
        self.memory_alloc_async = type_of(self.memory_alloc_async)(
            handle, so_path
        )
        self.memory_free_async = type_of(self.memory_free_async)(
            handle, so_path
        )
        self.queue_create = type_of(self.queue_create)(handle, so_path)
        self.queue_destroy = type_of(self.queue_destroy)(handle, so_path)
        self.queue_copy_to_device = type_of(self.queue_copy_to_device)(
            handle, so_path
        )
        self.queue_copy_from_device = type_of(self.queue_copy_from_device)(
            handle, so_path
        )
        self.queue_copy_device_to_device = type_of(
            self.queue_copy_device_to_device
        )(handle, so_path)
        self.queue_set_memory = type_of(self.queue_set_memory)(handle, so_path)
        self.event_create = type_of(self.event_create)(handle, so_path)
        self.event_destroy = type_of(self.event_destroy)(handle, so_path)
        self.event_synchronize = type_of(self.event_synchronize)(
            handle, so_path
        )
        self.is_event_ready = type_of(self.is_event_ready)(handle, so_path)
        self.function_load = type_of(self.function_load)(handle, so_path)
        self.function_unload = type_of(self.function_unload)(handle, so_path)
        self.device_property = type_of(self.device_property)(handle, so_path)
        self.memory_property = type_of(self.memory_property)(handle, so_path)
        self.queue_execute = type_of(self.queue_execute)(handle, so_path)
        self.queue_record_event = type_of(self.queue_record_event)(
            handle, so_path
        )
        self.queue_wait_for_events = type_of(self.queue_wait_for_events)(
            handle, so_path
        )
        self.queue_synchronize = type_of(self.queue_synchronize)(
            handle, so_path
        )
        self.queue_is_stream = type_of(self.queue_is_stream)(handle, so_path)
        self.bundle_load = type_of(self.bundle_load)(handle, so_path)
        self.bundle_unload = type_of(self.bundle_unload)(handle, so_path)

    def _create_init_driver(mut self) raises HALError -> DriverHandle:
        var handle = UnsafeMaybeUninit[DriverHandle]()
        var version = DriverVersion(
            major=M_DRIVER_INTERFACE_VERSION_MAJOR,
            minor=M_DRIVER_INTERFACE_VERSION_MINOR,
            patch=M_DRIVER_INTERFACE_VERSION_PATCH,
        )

        var status = self.create.f(
            ImmutPointer(
                to=UnsafePointer[DriverVersion, ImmutAnyOrigin](
                    UnsafePointer(to=version)
                )[]
            ),
            OutParam[DriverHandle](to=handle),
        )
        if status != STATUS_SUCCESS:
            raise HALError(
                status,
                message=String(
                    t"failed to initialise driver plugin from {self.so_path}"
                ),
            )

        var driver_handle = handle.unsafe_assume_init_ref()

        return driver_handle
