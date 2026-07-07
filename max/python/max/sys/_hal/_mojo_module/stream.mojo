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
"""Python projection of HAL ``Stream``."""

from std.collections import List
from std.memory import ArcPointer, OpaquePointer, UnsafePointer
from std.os import abort
from std.python import PythonObject
from _hal.device import get_device_spec
from _hal.event import EVENT_FLAG_CPU_VISIBLE
from _hal.plugin import EventHandle, FunctionHandle, RawDriver
from _hal.stream import Stream as HALStream

from .buffer import Buffer, BufferView
from .event import Event
from .function import Function


@fieldwise_init
struct Stream(Movable, Writable):
    """Python projection of HAL ``Stream``."""

    # TODO: generalize to multi-device — currently hardcoded to device 0.
    comptime device_spec = get_device_spec[0]()
    var _arc: ArcPointer[HALStream[Self.device_spec]]
    # Cached for `wait_for_events`, which drops below the HAL API
    # (HAL's `wait_for_events` is comptime-variadic, can't be built
    # from a runtime Python tuple). Other ops route through HALStream
    # methods so this cache isn't on their hot path.
    var _raw: ArcPointer[RawDriver]

    @staticmethod
    def _self_ptr(
        py_self: PythonObject,
    ) -> UnsafePointer[Self, MutAnyOrigin]:
        try:
            return py_self.downcast_value_ptr[Self]()
        except e:
            abort(String("Stream method receiver was not a Stream: ", e))

    @staticmethod
    def synchronize(py_self: PythonObject) raises:
        var self_ptr = Self._self_ptr(py_self)
        self_ptr[]._arc[].synchronize()

    @staticmethod
    def record_event(py_self: PythonObject) raises -> PythonObject:
        var self_ptr = Self._self_ptr(py_self)
        var hal_evt = self_ptr[]._arc[].record_event[EVENT_FLAG_CPU_VISIBLE]()
        var ctx_arc = self_ptr[]._arc[]._queue[]._context
        return PythonObject(alloc=Event(_hal=hal_evt^, _ctx_keepalive=ctx_arc))

    @staticmethod
    def copy_to_device(
        py_self: PythonObject,
        dst_obj: PythonObject,
        src_addr_obj: PythonObject,
    ) raises:
        var self_ptr = Self._self_ptr(py_self)
        var dst_view = dst_obj.downcast_value_ptr[BufferView]()
        var src_ptr = UnsafePointer[UInt8, ImmutAnyOrigin](
            unsafe_from_address=Int(py=src_addr_obj)
        )
        # Route through HALStream so the chain-wait / chain-signal logic
        # (needed to preserve in-order semantics on non-stream queues)
        # runs around the copy.
        self_ptr[]._arc[].copy_to_device(dst_view[]._hal, src_ptr)

    @staticmethod
    def copy_from_device(
        py_self: PythonObject,
        dst_addr_obj: PythonObject,
        src_obj: PythonObject,
    ) raises:
        var self_ptr = Self._self_ptr(py_self)
        var src_view = src_obj.downcast_value_ptr[BufferView]()
        var dst_ptr = UnsafePointer[UInt8, MutAnyOrigin](
            unsafe_from_address=Int(py=dst_addr_obj)
        )
        self_ptr[]._arc[].copy_from_device(dst_ptr, src_view[]._hal)

    @staticmethod
    def copy_intra_device(
        py_self: PythonObject,
        dst_obj: PythonObject,
        src_obj: PythonObject,
    ) raises:
        var self_ptr = Self._self_ptr(py_self)
        var dst_view = dst_obj.downcast_value_ptr[BufferView]()
        var src_view = src_obj.downcast_value_ptr[BufferView]()
        self_ptr[]._arc[].copy_intra_device(dst_view[]._hal, src_view[]._hal)

    @staticmethod
    def set_memory(
        py_self: PythonObject,
        dst_obj: PythonObject,
        value_obj: PythonObject,
        size_obj: PythonObject,
    ) raises:
        var self_ptr = Self._self_ptr(py_self)
        var dst_ptr = dst_obj.downcast_value_ptr[Buffer]()
        var size = UInt64(Int(py=size_obj))
        var value = UInt8(Int(py=value_obj))
        self_ptr[]._arc[].set_memory(
            dst_ptr[]._hal.view(byte_offset=0, byte_size=size), value
        )

    @staticmethod
    def fill(
        py_self: PythonObject,
        dst_obj: PythonObject,
        value_obj: PythonObject,
        value_size_obj: PythonObject,
        size_obj: PythonObject,
    ) raises:
        var self_ptr = Self._self_ptr(py_self)
        var dst_ptr = dst_obj.downcast_value_ptr[Buffer]()
        var size = UInt64(Int(py=size_obj))
        var value = UInt64(Int(py=value_obj))
        var value_size = UInt64(Int(py=value_size_obj))
        self_ptr[]._arc[].fill(
            dst_ptr[]._hal.view(byte_offset=0, byte_size=size),
            value,
            value_size,
        )

    @staticmethod
    def wait_for_events(py_self: PythonObject, events_obj: PythonObject) raises:
        var self_ptr = Self._self_ptr(py_self)
        var n = len(events_obj)
        if n == 0:
            return
        var handles = List[EventHandle](capacity=n)
        for i in range(n):
            var evt_ptr = events_obj[i].downcast_value_ptr[Event]()
            handles.append(evt_ptr[]._hal._handle())
        # Mirror HALStream.wait_for_events: enqueue the wait on the
        # underlying queue, then chain-signal so subsequent stream ops
        # are ordered after this wait.
        ref hal_stream = self_ptr[]._arc[]
        self_ptr[]._raw[].wait_for_events(
            hal_stream._queue[]._handle, handles.unsafe_ptr(), UInt32(n)
        )
        hal_stream._chain_signal()

    @staticmethod
    def execute(
        py_self: PythonObject,
        func_obj: PythonObject,
        grid_obj: PythonObject,
        block_obj: PythonObject,
        args_obj: PythonObject,
        arg_sizes_obj: PythonObject,
        shared_mem_bytes_obj: PythonObject,
    ) raises:
        var self_ptr = Self._self_ptr(py_self)
        var func_ptr = func_obj.downcast_value_ptr[Function]()
        var grid = (
            UInt32(Int(py=grid_obj[0])),
            UInt32(Int(py=grid_obj[1])),
            UInt32(Int(py=grid_obj[2])),
        )
        var block = (
            UInt32(Int(py=block_obj[0])),
            UInt32(Int(py=block_obj[1])),
            UInt32(Int(py=block_obj[2])),
        )
        var n = len(args_obj)
        var args = List[OpaquePointer[MutUntrackedOrigin]](capacity=n)
        var arg_sizes = List[UInt64](capacity=n)
        for i in range(n):
            args.append(
                OpaquePointer[MutUntrackedOrigin](
                    unsafe_from_address=Int(py=args_obj[i])
                )
            )
            arg_sizes.append(UInt64(Int(py=arg_sizes_obj[i])))
        var shared_mem_bytes = UInt32(Int(py=shared_mem_bytes_obj))
        self_ptr[]._arc[].execute(
            func_ptr[]._handle,
            grid,
            block,
            args.unsafe_ptr(),
            arg_sizes.unsafe_ptr(),
            UInt32(n),
            shared_mem_bytes=shared_mem_bytes,
        )

    def write_to(self, mut writer: Some[Writer]):
        writer.write("Stream()")

    def write_repr_to(self, mut writer: Some[Writer]):
        writer.write("Stream()")
