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
"""Python extension module backing ``max.sys._hal``."""

from std.os import abort
from std.python import PythonObject
from std.python.bindings import PythonModuleBuilder

from _mojo_module import (
    Buffer,
    BufferView,
    Bundle,
    Context,
    Device,
    Driver,
    Event,
    Function,
    Queue,
    Stream,
)


@export
def PyInit_mojo_module() abi("C") -> PythonObject:
    """Initializes the ``mojo_module`` Python extension."""
    try:
        var b = PythonModuleBuilder("mojo_module")

        _ = (
            b.add_type[Driver]("Driver")
            .def_py_init[Driver.py_init]()
            .def_method[Driver.get_name]("get_name")
            .def_method[Driver.device_count]("device_count")
            .def_method[Driver.get_device]("get_device")
        )

        _ = (
            b.add_type[Device]("Device")
            .def_method[Device.get_id]("get_id")
            .def_method[Device.get_context]("get_context")
        )

        _ = (
            b.add_type[Context]("Context")
            .def_method[Context.create_queue]("create_queue")
            .def_method[Context.create_stream]("create_stream")
            .def_method[Context.alloc_sync]("alloc_sync")
            .def_method[Context.alloc_host_pinned]("alloc_host_pinned")
            .def_method[Context.memory_get_address]("memory_get_address")
            .def_method[Context.load_function]("load_function")
        )

        _ = (
            b.add_type[Buffer]("Buffer")
            .def_method[Buffer.get_byte_size]("get_byte_size")
            .def_method[Buffer.view]("view")
        )

        _ = (
            b.add_type[BufferView]("BufferView")
            .def_method[BufferView.get_byte_offset]("get_byte_offset")
            .def_method[BufferView.get_byte_size]("get_byte_size")
        )

        _ = b.add_type[Bundle]("Bundle").def_method[Bundle.get_function_name](
            "get_function_name"
        )

        _ = b.add_type[Function]("Function")

        _ = (
            b.add_type[Queue]("Queue")
            .def_method[Queue.synchronize]("synchronize")
            .def_method[Queue.record_event]("record_event")
            .def_method[Queue.copy_to_device]("copy_to_device")
            .def_method[Queue.copy_from_device]("copy_from_device")
            .def_method[Queue.copy_intra_device]("copy_intra_device")
            .def_method[Queue.set_memory]("set_memory")
            .def_method[Queue.fill]("fill")
            .def_method[Queue.wait_for_events]("wait_for_events")
            .def_method[Queue.execute]("execute")
        )

        _ = (
            b.add_type[Stream]("Stream")
            .def_method[Stream.synchronize]("synchronize")
            .def_method[Stream.record_event]("record_event")
            .def_method[Stream.copy_to_device]("copy_to_device")
            .def_method[Stream.copy_from_device]("copy_from_device")
            .def_method[Stream.copy_intra_device]("copy_intra_device")
            .def_method[Stream.set_memory]("set_memory")
            .def_method[Stream.fill]("fill")
            .def_method[Stream.wait_for_events]("wait_for_events")
            .def_method[Stream.execute]("execute")
        )

        _ = (
            b.add_type[Event]("Event")
            .def_method[Event.synchronize]("synchronize")
            .def_method[Event.is_ready]("is_ready")
        )

        return b.finalize()
    except e:
        abort(t"failed to create Python module: {e}")
