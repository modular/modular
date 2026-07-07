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
"""Python projection of HAL ``Context``."""

from std.memory import ArcPointer, UnsafePointer
from std.os import abort
from std.python import PythonObject
from _hal.buffer import Buffer as HALBuffer
from _hal.context import Context as HALContext
from _hal.device import get_device_spec
from _hal.queue import Queue as HALQueue
from _hal.stream import Stream as HALStream

from .buffer import Buffer
from .bundle import Bundle
from .function import Function
from .queue import Queue
from .stream import Stream


@fieldwise_init
struct Context(Movable, Writable):
    """Python projection of HAL ``Context``."""

    # TODO: generalize to multi-device — currently hardcoded to device 0.
    comptime device_spec = get_device_spec[0]()
    var _arc: ArcPointer[HALContext[Self.device_spec]]

    @staticmethod
    def _self_ptr(
        py_self: PythonObject,
    ) -> UnsafePointer[Self, MutAnyOrigin]:
        try:
            return py_self.downcast_value_ptr[Self]()
        except e:
            abort(String("Context method receiver was not a Context: ", e))

    @staticmethod
    def create_queue(py_self: PythonObject) raises -> PythonObject:
        var self_ptr = Self._self_ptr(py_self)
        var queue_arc = HALQueue[Self.device_spec]._create(self_ptr[]._arc[])
        var raw_arc = queue_arc[]._raw
        return PythonObject(alloc=Queue(_arc=queue_arc^, _raw=raw_arc^))

    @staticmethod
    def create_stream(py_self: PythonObject) raises -> PythonObject:
        var self_ptr = Self._self_ptr(py_self)
        var stream_arc = HALStream[Self.device_spec]._create(self_ptr[]._arc[])
        var raw_arc = stream_arc[]._queue[]._raw
        return PythonObject(alloc=Stream(_arc=stream_arc^, _raw=raw_arc^))

    @staticmethod
    def alloc_sync(
        py_self: PythonObject, size_obj: PythonObject
    ) raises -> PythonObject:
        var self_ptr = Self._self_ptr(py_self)
        var byte_size = UInt64(Int(py=size_obj))
        var hal_buf = self_ptr[]._arc[].alloc_sync(byte_size)
        var ctx_arc = self_ptr[]._arc
        return PythonObject(
            alloc=Buffer(
                _hal=hal_buf^,
                _ctx=ctx_arc^,
                _is_pinned=False,
            )
        )

    @staticmethod
    def alloc_host_pinned(
        py_self: PythonObject, size_obj: PythonObject
    ) raises -> PythonObject:
        var self_ptr = Self._self_ptr(py_self)
        var byte_size = UInt64(Int(py=size_obj))
        var hal_buf = self_ptr[]._arc[].alloc_host_pinned(byte_size)
        var ctx_arc = self_ptr[]._arc
        return PythonObject(
            alloc=Buffer(
                _hal=hal_buf^,
                _ctx=ctx_arc^,
                _is_pinned=True,
            )
        )

    @staticmethod
    def memory_get_address(
        py_self: PythonObject, buf_obj: PythonObject
    ) raises -> PythonObject:
        var self_ptr = Self._self_ptr(py_self)
        var buf_ptr = buf_obj.downcast_value_ptr[Buffer]()
        return PythonObject(
            Int(self_ptr[]._arc[].memory_get_address(buf_ptr[]._hal))
        )

    @staticmethod
    def load_function(
        py_self: PythonObject,
        bundle_obj: PythonObject,
        name_obj: PythonObject,
    ) raises -> PythonObject:
        var self_ptr = Self._self_ptr(py_self)
        var bundle_ptr = bundle_obj.downcast_value_ptr[Bundle]()
        var name = String(py=name_obj)
        var func_handle = (
            self_ptr[]._arc[].load_function(bundle_ptr[]._arc[], name)
        )
        var ctx_arc = self_ptr[]._arc
        var bundle_arc = bundle_ptr[]._arc
        return PythonObject(
            alloc=Function(
                _ctx=ctx_arc^,
                _bundle=bundle_arc^,
                _handle=func_handle,
            )
        )

    def write_to(self, mut writer: Some[Writer]):
        writer.write("Context()")

    def write_repr_to(self, mut writer: Some[Writer]):
        writer.write("Context()")
