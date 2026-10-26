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
from . import *


struct RawPluginBase[required: Bool = True](RawPluginComponent):
    # Resolved function pointers.
    var status_message: HALFunction[
        "M_driver_status_message",
        def(
            driver: DriverHandle,
            status: Int64,
            message_buffer: MutPointer[Int8, _],
            message_buffer_size: Int64,
        ) thin abi("C") -> PluginResultCode,
        required=Self.required,
    ]
    var create: HALFunction[
        "M_driver_create",
        def(
            version: ImmPointer[DriverVersion, _],
            driver: OutParam[DriverHandle, _],
        ) thin abi("C") -> PluginResultCode,
        required=Self.required,
    ]
    var destroy: HALFunction[
        "M_driver_destroy",
        def(driver: DriverHandle) thin abi("C") -> PluginResultCode,
        required=Self.required,
    ]
    var property: HALFunction[
        "M_driver_property",
        def(
            handle: DriverHandle,
            property_name: CStringSpan[_],
            value: OutParam[OpaquePointer[ImmUntrackedOrigin], _],
        ) thin abi("C") -> PluginResultCode,
        required=Self.required,
    ]
    var device_count: HALFunction[
        "M_driver_device_count",
        def(
            driver: DriverHandle, count: OutParam[Int64, _]
        ) thin abi("C") -> PluginResultCode,
        required=Self.required,
    ]
    var device_get: HALFunction[
        "M_driver_device_get",
        def(
            driver: DriverHandle,
            device_id: Int64,
            device: OutParam[DeviceHandle, _],
        ) thin abi("C") -> PluginResultCode,
        required=Self.required,
    ]
    var context_create: HALFunction[
        "M_driver_context_create",
        def(
            device: DeviceHandle, context: OutParam[ContextHandle, _]
        ) thin abi("C") -> PluginResultCode,
        required=Self.required,
    ]
    var context_destroy: HALFunction[
        "M_driver_context_destroy",
        def(context: ContextHandle) thin abi("C") -> PluginResultCode,
        required=Self.required,
    ]
    var context_set_current: HALFunction[
        "M_driver_context_set_current",
        def(context: ContextHandle) thin abi("C") -> PluginResultCode,
        required=Self.required,
    ]
    var context_get_current_driver_context: HALFunction[
        "M_driver_context_get_current_driver_context",
        def(
            context: ContextHandle, driver_context: OutParam[Int, _]
        ) thin abi("C") -> PluginResultCode,
        required=Self.required,
    ]
    var context_set_current_driver_context: HALFunction[
        "M_driver_context_set_current_driver_context",
        def(
            context: ContextHandle, driver_context: Int
        ) thin abi("C") -> PluginResultCode,
        required=Self.required,
    ]
    var context_memory_info: HALFunction[
        "M_driver_context_memory_info",
        def(
            context: ContextHandle,
            free: OutParam[UInt64, _],
            total: OutParam[UInt64, _],
        ) thin abi("C") -> PluginResultCode,
        required=Self.required,
    ]
    var memory_alloc_pinned: HALFunction[
        "M_driver_memory_alloc_pinned",
        def(
            context: ContextHandle,
            size: UInt64,
            memory: OutParam[MemoryHandle, _],
        ) thin abi("C") -> PluginResultCode,
        required=Self.required,
    ]
    var memory_free_pinned: HALFunction[
        "M_driver_memory_free_pinned",
        def(
            context: ContextHandle,
            memory: MemoryHandle,
        ) thin abi("C") -> PluginResultCode,
        required=Self.required,
    ]
    var memory_alloc_sync: HALFunction[
        "M_driver_memory_alloc_sync",
        def(
            context: ContextHandle,
            size: UInt64,
            memory: OutParam[MemoryHandle, _],
        ) thin abi("C") -> PluginResultCode,
        required=Self.required,
    ]
    var memory_free_sync: HALFunction[
        "M_driver_memory_free_sync",
        def(
            context: ContextHandle,
            memory: MemoryHandle,
        ) thin abi("C") -> PluginResultCode,
        required=Self.required,
    ]
    var memory_wrap: HALFunction[
        "M_driver_memory_wrap",
        def(
            context: ContextHandle,
            address: UInt64,
            size: UInt64,
            owning: Bool,
            memory: OutParam[MemoryHandle, _],
        ) thin abi("C") -> PluginResultCode,
        required=Self.required,
    ]
    var memory_unwrap: HALFunction[
        "M_driver_memory_unwrap",
        def(
            context: ContextHandle,
            memory: MemoryHandle,
            address: OutParam[UInt64, _],
        ) thin abi("C") -> PluginResultCode,
        required=Self.required,
    ]
    var memory_alloc_async: HALFunction[
        "M_driver_memory_alloc_async",
        def(
            queue: QueueHandle,
            size: UInt64,
            memory: OutParam[MemoryHandle, _],
        ) thin abi("C") -> PluginResultCode,
        required=Self.required,
    ]
    var memory_free_async: HALFunction[
        "M_driver_memory_free_async",
        def(
            queue: QueueHandle,
            memory: MemoryHandle,
        ) thin abi("C") -> PluginResultCode,
        required=Self.required,
    ]
    var queue_create: HALFunction[
        "M_driver_queue_create",
        def(
            context: ContextHandle, queue: OutParam[QueueHandle, _]
        ) thin abi("C") -> PluginResultCode,
        required=Self.required,
    ]
    var queue_destroy: HALFunction[
        "M_driver_queue_destroy",
        def(
            context: ContextHandle, queue: QueueHandle
        ) thin abi("C") -> PluginResultCode,
        required=Self.required,
    ]
    var queue_copy_to_device: HALFunction[
        "M_driver_queue_copy_to_device",
        def[
            dst_origin: ImmOrigin, src_origin: ImmOrigin
        ](
            queue: QueueHandle,
            dst: MemoryViewHandle[dst_origin],
            src: Pointer[UInt8, src_origin],
        ) thin abi("C") -> PluginResultCode,
        required=Self.required,
    ]
    var queue_copy_from_device: HALFunction[
        "M_driver_queue_copy_from_device",
        def[
            dst_origin: MutOrigin, src_origin: ImmOrigin
        ](
            queue: QueueHandle,
            dst: Pointer[UInt8, dst_origin],
            src: MemoryViewHandle[src_origin],
        ) thin abi("C") -> PluginResultCode,
        required=Self.required,
    ]
    var queue_copy_intra_device: HALFunction[
        "M_driver_queue_copy_intra_device",
        def[
            dst_origin: ImmOrigin, src_origin: ImmOrigin
        ](
            queue: QueueHandle,
            dst: MemoryViewHandle[dst_origin],
            src: MemoryViewHandle[src_origin],
        ) thin abi("C") -> PluginResultCode,
        required=Self.required,
    ]
    var queue_copy_inter_device: HALFunction[
        "M_driver_queue_copy_inter_device",
        def[
            dst_origin: ImmOrigin, src_origin: ImmOrigin
        ](
            queue: QueueHandle,
            dst: MemoryViewHandle[dst_origin],
            src: MemoryViewHandle[src_origin],
            src_context: ContextHandle,
        ) thin abi("C") -> PluginResultCode,
        required=Self.required,
    ]
    var copy_to_device_sync: HALFunction[
        "M_driver_copy_to_device_sync",
        def[
            dst_origin: ImmOrigin, src_origin: ImmOrigin
        ](
            context: ContextHandle,
            dst: MemoryViewHandle[dst_origin],
            src: Pointer[UInt8, src_origin],
        ) thin abi("C") -> PluginResultCode,
        required=Self.required,
    ]
    var copy_from_device_sync: HALFunction[
        "M_driver_copy_from_device_sync",
        def[
            dst_origin: MutOrigin, src_origin: ImmOrigin
        ](
            context: ContextHandle,
            dst: Pointer[UInt8, dst_origin],
            src: MemoryViewHandle[src_origin],
        ) thin abi("C") -> PluginResultCode,
        required=Self.required,
    ]
    var copy_intra_device_sync: HALFunction[
        "M_driver_copy_intra_device_sync",
        def[
            dst_origin: ImmOrigin, src_origin: ImmOrigin
        ](
            context: ContextHandle,
            dst: MemoryViewHandle[dst_origin],
            src: MemoryViewHandle[src_origin],
        ) thin abi("C") -> PluginResultCode,
        required=Self.required,
    ]
    var copy_inter_device_sync: HALFunction[
        "M_driver_copy_inter_device_sync",
        def[
            dst_origin: ImmOrigin, src_origin: ImmOrigin
        ](
            context: ContextHandle,
            dst: MemoryViewHandle[dst_origin],
            src: MemoryViewHandle[src_origin],
            src_context: ContextHandle,
        ) thin abi("C") -> PluginResultCode,
        required=Self.required,
    ]
    var set_memory_sync: HALFunction[
        "M_driver_set_memory_sync",
        def[
            dst_origin: ImmOrigin
        ](
            context: ContextHandle,
            dst: MemoryViewHandle[dst_origin],
            value: UInt8,
        ) thin abi("C") -> PluginResultCode,
        required=Self.required,
    ]
    var fill_sync: HALFunction[
        "M_driver_fill_sync",
        def[
            dst_origin: ImmOrigin
        ](
            context: ContextHandle,
            dst: MemoryViewHandle[dst_origin],
            value: UInt64,
            value_size: UInt64,
        ) thin abi("C") -> PluginResultCode,
        required=Self.required,
    ]
    var queue_set_memory: HALFunction[
        "M_driver_queue_set_memory",
        def[
            dst_origin: ImmOrigin
        ](
            queue: QueueHandle,
            dst: MemoryViewHandle[dst_origin],
            value: UInt8,
        ) thin abi("C") -> PluginResultCode,
        required=Self.required,
    ]
    var queue_fill: HALFunction[
        "M_driver_queue_fill",
        def[
            dst_origin: ImmOrigin
        ](
            queue: QueueHandle,
            dst: MemoryViewHandle[dst_origin],
            value: UInt64,
            value_size: UInt64,
        ) thin abi("C") -> PluginResultCode,
        required=Self.required,
    ]
    var event_create: HALFunction[
        "M_driver_event_create",
        def(
            context: ContextHandle,
            flags: UInt32,
            event: OutParam[EventHandle, _],
        ) thin abi("C") -> PluginResultCode,
        required=Self.required,
    ]
    var event_destroy: HALFunction[
        "M_driver_event_destroy",
        def(
            context: ContextHandle, event: EventHandle
        ) thin abi("C") -> PluginResultCode,
        required=Self.required,
    ]
    var event_synchronize: HALFunction[
        "M_driver_event_synchronize",
        def(
            context: ContextHandle, event: EventHandle
        ) thin abi("C") -> PluginResultCode,
        required=Self.required,
    ]
    var is_event_ready: HALFunction[
        "M_driver_is_event_ready",
        def(
            context: ContextHandle,
            event: EventHandle,
            is_ready: OutParam[Bool, _],
        ) thin abi("C") -> PluginResultCode,
        required=Self.required,
    ]
    var function_load: HALFunction[
        "M_driver_function_load",
        def(
            context: ContextHandle,
            bundle: RuntimeBundleHandle,
            function_name: CStringSpan[_],
            function_name_len: UInt64,
            function: OutParam[FunctionHandle, _],
        ) thin abi("C") -> PluginResultCode,
        required=Self.required,
    ]
    var function_unload: HALFunction[
        "M_driver_function_unload",
        def(
            context: ContextHandle, function: FunctionHandle
        ) thin abi("C") -> PluginResultCode,
        required=Self.required,
    ]
    var function_set_attribute: HALFunction[
        "M_driver_function_set_attribute",
        def(
            context: ContextHandle,
            function: FunctionHandle,
            attribute: Int32,
            value: Int32,
        ) thin abi("C") -> PluginResultCode,
        required=Self.required,
    ]
    var function_occupancy_max_active_blocks: HALFunction[
        "M_driver_function_occupancy_max_active_blocks",
        def(
            context: ContextHandle,
            function: FunctionHandle,
            block_size: Int32,
            dynamic_shared_memory_bytes: UInt64,
            num_blocks: OutParam[Int32, _],
        ) thin abi("C") -> PluginResultCode,
        required=Self.required,
    ]
    # The array parameters mirror the C header's `const uint64_t *` /
    # `const uint32_t *` types; the device address is a `uint64_t` like the
    # other device-address parameters in the ABI.
    var tensor_map_encode_tiled: HALFunction[
        "M_driver_tensor_map_encode_tiled",
        def[
            tensor_map_origin: MutOrigin,
            global_dim_origin: ImmOrigin,
            global_strides_origin: ImmOrigin,
            box_dim_origin: ImmOrigin,
            element_strides_origin: ImmOrigin,
        ](
            context: ContextHandle,
            tensor_map: OpaquePointer[tensor_map_origin],
            data_type: Int32,
            rank: Int32,
            global_device_address: UInt64,
            global_dim: Pointer[UInt64, global_dim_origin],
            global_strides: Pointer[UInt64, global_strides_origin],
            box_dim: Pointer[UInt32, box_dim_origin],
            element_strides: Pointer[UInt32, element_strides_origin],
            interleave: Int32,
            swizzle: Int32,
            l2_promotion: Int32,
            oob_fill: Int32,
        ) thin abi("C") -> PluginResultCode,
        required=Self.required,
    ]
    var tensor_map_encode_im2col: HALFunction[
        "M_driver_tensor_map_encode_im2col",
        def[
            tensor_map_origin: MutOrigin,
            global_dim_origin: ImmOrigin,
            global_strides_origin: ImmOrigin,
            lower_corner_origin: ImmOrigin,
            upper_corner_origin: ImmOrigin,
            element_strides_origin: ImmOrigin,
        ](
            context: ContextHandle,
            tensor_map: OpaquePointer[tensor_map_origin],
            data_type: Int32,
            rank: Int32,
            global_device_address: UInt64,
            global_dim: Pointer[UInt64, global_dim_origin],
            global_strides: Pointer[UInt64, global_strides_origin],
            pixel_box_lower_corner: Pointer[Int32, lower_corner_origin],
            pixel_box_upper_corner: Pointer[Int32, upper_corner_origin],
            channels_per_pixel: Int32,
            pixels_per_column: Int32,
            element_strides: Pointer[UInt32, element_strides_origin],
            interleave: Int32,
            swizzle: Int32,
            l2_promotion: Int32,
            oob_fill: Int32,
        ) thin abi("C") -> PluginResultCode,
        required=Self.required,
    ]
    var device_property: HALFunction[
        "M_driver_device_property",
        def[
            value_origin: MutOrigin
        ](
            device: DeviceHandle,
            property_name: CStringSpan[_],
            value: OpaquePointer[value_origin],
        ) thin abi("C") -> PluginResultCode,
        required=Self.required,
    ]
    var device_attribute: HALFunction[
        "M_driver_device_attribute",
        def(
            device: DeviceHandle,
            attribute: Int32,
            value: OutParam[Int32, _],
        ) thin abi("C") -> PluginResultCode,
        required=Self.required,
    ]
    var memory_property: HALFunction[
        "M_driver_memory_property",
        def[
            value_origin: MutOrigin
        ](
            memory: MemoryHandle,
            property_name: CStringSpan[_],
            value: OpaquePointer[value_origin],
        ) thin abi("C") -> PluginResultCode,
        required=Self.required,
    ]
    var queue_execute: HALFunction[
        "M_driver_queue_execute",
        def[
            attr_origin: MutOrigin,
            config_origin: MutOrigin,
            args_origin: MutOrigin,
            arg_sizes_origin: MutOrigin,
        ](
            queue: QueueHandle,
            function: FunctionHandle,
            config: InternalHandle[
                M_driver_queue_execute_config[attr_origin], config_origin
            ],
            args: Pointer[OpaquePointer[MutUntrackedOrigin], args_origin],
            arg_sizes: Pointer[UInt64, arg_sizes_origin],
            num_args: UInt32,
        ) thin abi("C") -> PluginResultCode,
        required=Self.required,
    ]
    var queue_record_event: HALFunction[
        "M_driver_queue_record_event",
        def(
            queue: QueueHandle, event: EventHandle
        ) thin abi("C") -> PluginResultCode,
        required=Self.required,
    ]
    var queue_wait_for_events: HALFunction[
        "M_driver_queue_wait_for_events",
        def[
            events_origin: MutOrigin
        ](
            queue: QueueHandle,
            events: Pointer[EventHandle, events_origin],
            num_events: UInt32,
        ) thin abi("C") -> PluginResultCode,
        required=Self.required,
    ]
    var queue_synchronize: HALFunction[
        "M_driver_queue_synchronize",
        def(queue: QueueHandle) thin abi("C") -> PluginResultCode,
        required=Self.required,
    ]
    var queue_is_stream: HALFunction[
        "M_driver_queue_is_stream",
        def(
            queue: QueueHandle, is_stream: OutParam[Bool, _]
        ) thin abi("C") -> PluginResultCode,
        required=Self.required,
    ]
    var queue_wait_value64: HALFunction[
        "M_driver_queue_wait_value64",
        def(
            queue: QueueHandle,
            device_address: UInt64,
            value: UInt64,
        ) thin abi("C") -> PluginResultCode,
        required=Self.required,
    ]
    var queue_property: HALFunction[
        "M_driver_queue_property",
        def[
            value_origin: MutOrigin
        ](
            queue: QueueHandle,
            property_name: CStringSpan[_],
            value: OpaquePointer[value_origin],
        ) thin abi("C") -> PluginResultCode,
        required=Self.required,
    ]
    var queue_launch_host_func: HALFunction[
        "M_driver_queue_launch_host_func",
        def[
            origin: MutOrigin
        ](
            queue: QueueHandle,
            func: def(OpaquePointer[origin]) thin -> None,
            user_data: OpaquePointer[origin],
        ) thin abi("C") -> PluginResultCode,
        required=Self.required,
    ]
    var bundle_load: HALFunction[
        "M_driver_bundle_load",
        def[
            bundle_origin: MutOrigin,
            bundle_data_origin: ImmOrigin,
            opts_origin: MutOrigin,
        ](
            context: ContextHandle,
            bundle: InternalHandle[
                M_driver_static_bundle[bundle_data_origin], bundle_origin
            ],
            opts: InternalHandle[
                M_driver_bundle_compilation_options, opts_origin
            ],
            runtime_bundle: OutParam[RuntimeBundleHandle, _],
        ) thin abi("C") -> PluginResultCode,
        required=Self.required,
    ]
    var bundle_unload: HALFunction[
        "M_driver_bundle_unload",
        def(
            context: ContextHandle,
            bundle: RuntimeBundleHandle,
        ) thin abi("C") -> PluginResultCode,
        required=Self.required,
    ]
