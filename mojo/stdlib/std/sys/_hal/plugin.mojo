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

from std.ffi import _DLHandle, OwnedDLHandle, RTLD, CStringSlice

from std.memory import (
    alloc,
    ImmutPointer,
    MutPointer,
    UnsafePointer,
    UnsafeMaybeUninit,
)
from std.memory._nonnull import NonNullUnsafePointer

from .status import STATUS_SUCCESS, STATUS_UNKNOWN_ERROR, HALError


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


struct M_driver_bundle:
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
comptime OutParam[T: TrivialRegisterPassable] = NonNullUnsafePointer[
    UnsafeMaybeUninit[T], MutAnyOrigin
]

comptime DriverHandle = Handle[M_driver_driver]
comptime DeviceHandle = Handle[M_driver_device]
comptime ContextHandle = Handle[M_driver_context]
comptime QueueHandle = Handle[M_driver_queue]
comptime EventHandle = Handle[M_driver_event]
comptime FunctionHandle = Handle[M_driver_function]
comptime MemoryHandle = Handle[M_driver_memory]
comptime BundleHandle = Handle[M_driver_bundle]

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


# ===-----------------------------------------------------------------------===#
# Plugin
# ===-----------------------------------------------------------------------===#


@fieldwise_init
struct Plugin(Movable):
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
            status: Int64,
            message_buffer: MutPointer[Int8, MutAnyOrigin],
            message_buffer_size: Int64,
        ) -> PluginResultCode,
    ]
    var create: HALFunction[
        "M_driver_create",
        def(
            version: ImmutPointer[DriverVersion, ImmutAnyOrigin],
            driver: OutParam[DriverHandle],
        ) -> PluginResultCode,
    ]
    var destroy: HALFunction[
        "M_driver_destroy",
        def(driver: DriverHandle) -> PluginResultCode,
    ]
    var property: HALFunction[
        "M_driver_property",
        def(
            handle: DriverHandle,
            property_name: CStringSlice,
            value: OutParam[OpaquePointer[ImmutExternalOrigin]],
        ) -> PluginResultCode,
    ]
    var device_count: HALFunction[
        "M_driver_device_count",
        def(driver: DriverHandle, count: OutParam[Int64]) -> PluginResultCode,
    ]
    var device_get: HALFunction[
        "M_driver_device_get",
        def(
            driver: DriverHandle,
            device_id: Int64,
            device: OutParam[DeviceHandle],
        ) -> PluginResultCode,
    ]
    var context_create: HALFunction[
        "M_driver_context_create",
        def(
            device: DeviceHandle, context: OutParam[ContextHandle]
        ) -> PluginResultCode,
    ]
    var context_destroy: HALFunction[
        "M_driver_context_destroy",
        def(context: ContextHandle) -> PluginResultCode,
    ]
    var memory_alloc_pinned: HALFunction[
        "M_driver_memory_alloc_pinned",
        def(
            context: ContextHandle,
            size: UInt64,
            memory: OutParam[MemoryHandle],
        ) -> PluginResultCode,
    ]
    var memory_free_pinned: HALFunction[
        "M_driver_memory_free_pinned",
        def(
            context: ContextHandle,
            memory: MemoryHandle,
        ) -> PluginResultCode,
    ]
    var memory_alloc_sync: HALFunction[
        "M_driver_memory_alloc_sync",
        def(
            context: ContextHandle,
            size: UInt64,
            memory: OutParam[MemoryHandle],
        ) -> PluginResultCode,
    ]
    var memory_free_sync: HALFunction[
        "M_driver_memory_free_sync",
        def(
            context: ContextHandle,
            memory: MemoryHandle,
        ) -> PluginResultCode,
    ]
    var memory_alloc_async: HALFunction[
        "M_driver_memory_alloc_async",
        def(
            queue: QueueHandle,
            size: UInt64,
            memory: OutParam[MemoryHandle],
        ) -> PluginResultCode,
    ]
    var memory_free_async: HALFunction[
        "M_driver_memory_free_async",
        def(
            queue: QueueHandle,
            memory: MemoryHandle,
        ) -> PluginResultCode,
    ]
    var queue_create: HALFunction[
        "M_driver_queue_create",
        def(
            context: ContextHandle, queue: OutParam[QueueHandle]
        ) -> PluginResultCode,
    ]
    var queue_destroy: HALFunction[
        "M_driver_queue_destroy",
        def(context: ContextHandle, queue: QueueHandle) -> PluginResultCode,
    ]
    var queue_copy_to_device: HALFunction[
        "M_driver_queue_copy_to_device",
        def(
            queue: QueueHandle,
            dst: MemoryHandle,
            src: UnsafePointer[UInt8, MutAnyOrigin],
            size: UInt64,
        ) -> PluginResultCode,
    ]
    var queue_copy_from_device: HALFunction[
        "M_driver_queue_copy_from_device",
        def(
            queue: QueueHandle,
            dst: UnsafePointer[UInt8, MutAnyOrigin],
            src: UnsafePointer[UInt8, MutAnyOrigin],
            size: UInt64,
        ) -> PluginResultCode,
    ]
    var queue_set_memory: HALFunction[
        "M_driver_queue_set_memory",
        def(
            queue: QueueHandle,
            dst: UnsafePointer[UInt8, MutAnyOrigin],
            size: UInt64,
            value: UInt64,
            value_size: UInt64,
        ) -> PluginResultCode,
    ]
    var event_create: HALFunction[
        "M_driver_event_create",
        def(
            context: ContextHandle, event: OutParam[EventHandle]
        ) -> PluginResultCode,
    ]
    var event_destroy: HALFunction[
        "M_driver_event_destroy",
        def(context: ContextHandle, event: EventHandle) -> PluginResultCode,
    ]
    var event_synchronize: HALFunction[
        "M_driver_event_synchronize",
        def(context: ContextHandle, event: EventHandle) -> PluginResultCode,
    ]
    var is_event_ready: HALFunction[
        "M_driver_is_event_ready",
        def(
            context: ContextHandle,
            event: EventHandle,
            is_ready: OutParam[Bool],
        ) -> PluginResultCode,
    ]
    var function_load: HALFunction[
        "M_driver_function_load",
        def(
            context: ContextHandle,
            bundle: BundleHandle,
            function_name: CStringSlice,
            function_name_len: UInt64,
            function: OutParam[FunctionHandle],
        ) -> PluginResultCode,
    ]
    var function_unload: HALFunction[
        "M_driver_function_unload",
        def(
            context: ContextHandle, function: FunctionHandle
        ) -> PluginResultCode,
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
        self.queue_set_memory = type_of(self.queue_set_memory)(handle, so_path)
        self.event_create = type_of(self.event_create)(handle, so_path)
        self.event_destroy = type_of(self.event_destroy)(handle, so_path)
        self.event_synchronize = type_of(self.event_synchronize)(
            handle, so_path
        )
        self.is_event_ready = type_of(self.is_event_ready)(handle, so_path)
        self.function_load = type_of(self.function_load)(handle, so_path)
        self.function_unload = type_of(self.function_unload)(handle, so_path)

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

        try:
            var lib = OwnedDLHandle(so_path, RTLD.NOW)
            var plugin = Plugin(lib^, so_path)
            plugin.name = name
            plugin.so_path = so_path
            return plugin^
        except e:
            raise HALError(
                STATUS_UNKNOWN_ERROR,
                message=String(t"Failed to load plugin '{so_path}': {e}"),
            )

    def get_status_message(self, status: Int64) raises HALError -> HALError:
        """Retrieve the human-readable message associated with a status code."""
        var buf = List[Int8](length=1024, fill=0)
        var ret = self.status_message.f(
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
                STATUS_UNKNOWN_ERROR,
                message="(problem retrieving status message)",
            )
