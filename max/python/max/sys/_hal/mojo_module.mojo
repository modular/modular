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
"""Python extension module backing ``max.sys._hal``.

Projects the Mojo HAL primitives (``Driver``/``Device``/``Context``/
``Queue``/``Stream``/``Event``) into Python with the same names and the
same lifecycle relationships. Scope of this PR: lifecycle chain only —
no buffer allocation, no copies, no kernel launches.
"""

from std.memory import ArcPointer, UnsafePointer
from std.os import abort, getenv
from std.python import PythonObject
from std.python.bindings import (
    PythonModuleBuilder,
    check_arguments_arity,
    check_and_get_or_convert_arg,
)
from std.sys._hal.driver import Driver as HALDriver
from std.sys._hal.device import (
    Device as HALDevice,
    get_device_spec,
)
from std.sys._hal.context import Context as HALContext
from std.sys._hal.queue import Queue as HALQueue
from std.sys._hal.stream import Stream as HALStream
from std.sys._hal.event import (
    Event as HALEvent,
    EVENT_FLAG_CPU_VISIBLE,
)


def _load_driver() raises -> ArcPointer[HALDriver]:
    # TODO: ``MODULAR_DRIVER_PLUGINS`` will become a colon-separated list of
    # plugin specs (one per HAL backend); when that lands, replace this
    # helper with a plural ``_load_drivers()``.
    var spec = getenv("MODULAR_DRIVER_PLUGINS")
    if not spec:
        raise Error("MODULAR_DRIVER_PLUGINS env var not set.")
    return HALDriver.create(spec)


@fieldwise_init
struct Driver(Movable, Writable):
    """Python projection of HAL ``Driver``."""

    var _arc: ArcPointer[HALDriver]

    @staticmethod
    def py_init(
        out self: Self, args: PythonObject, kwargs: PythonObject
    ) raises:
        check_arguments_arity(0, args, "Driver")
        self = Self(_arc=_load_driver())

    @staticmethod
    def _self_ptr(
        py_self: PythonObject,
    ) -> UnsafePointer[Self, MutAnyOrigin]:
        try:
            return py_self.downcast_value_ptr[Self]()
        except e:
            abort(String("Driver method receiver was not a Driver: ", e))

    @staticmethod
    def get_name(py_self: PythonObject) raises -> PythonObject:
        var self_ptr = Self._self_ptr(py_self)
        return PythonObject(self_ptr[]._arc[].get_name())

    @staticmethod
    def device_count(py_self: PythonObject) raises -> PythonObject:
        var self_ptr = Self._self_ptr(py_self)
        return PythonObject(Int(self_ptr[]._arc[].get_device_count()))

    @staticmethod
    def get_device(
        py_self: PythonObject, idx_obj: PythonObject
    ) raises -> PythonObject:
        var self_ptr = Self._self_ptr(py_self)
        var idx = Int64(Int(py=idx_obj))
        var device_arc = HALDevice[Device.device_spec]._create(
            self_ptr[]._arc[], idx
        )
        return PythonObject(alloc=Device(_arc=device_arc^))

    def write_to(self, mut writer: Some[Writer]):
        writer.write("Driver(name=", self._arc[].get_name(), ")")

    def write_repr_to(self, mut writer: Some[Writer]):
        writer.write("Driver(name=", self._arc[].get_name(), ")")


@fieldwise_init
struct Device(Movable, Writable):
    """Python projection of HAL ``Device``."""

    comptime device_spec = get_device_spec[0]()
    var _arc: ArcPointer[HALDevice[Self.device_spec]]

    @staticmethod
    def _self_ptr(
        py_self: PythonObject,
    ) -> UnsafePointer[Self, MutAnyOrigin]:
        try:
            return py_self.downcast_value_ptr[Self]()
        except e:
            abort(String("Device method receiver was not a Device: ", e))

    @staticmethod
    def get_id(py_self: PythonObject) raises -> PythonObject:
        var self_ptr = Self._self_ptr(py_self)
        return PythonObject(Int(self_ptr[]._arc[].id))

    @staticmethod
    def get_context(py_self: PythonObject) raises -> PythonObject:
        var self_ptr = Self._self_ptr(py_self)
        var ctx_arc = HALContext[Self.device_spec]._create(self_ptr[]._arc[])
        return PythonObject(alloc=Context(_arc=ctx_arc^))

    def write_to(self, mut writer: Some[Writer]):
        writer.write("Device(id=", self._arc[].id, ")")

    def write_repr_to(self, mut writer: Some[Writer]):
        writer.write("Device(id=", self._arc[].id, ")")


@fieldwise_init
struct Context(Movable, Writable):
    """Python projection of HAL ``Context``."""

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
        return PythonObject(alloc=Queue(_arc=queue_arc^))

    @staticmethod
    def create_stream(py_self: PythonObject) raises -> PythonObject:
        var self_ptr = Self._self_ptr(py_self)
        var stream_arc = HALStream[Self.device_spec]._create(self_ptr[]._arc[])
        return PythonObject(alloc=Stream(_arc=stream_arc^))

    def write_to(self, mut writer: Some[Writer]):
        writer.write("Context()")

    def write_repr_to(self, mut writer: Some[Writer]):
        writer.write("Context()")


@fieldwise_init
struct Queue(Movable, Writable):
    """Python projection of HAL ``Queue``."""

    comptime device_spec = get_device_spec[0]()
    var _arc: ArcPointer[HALQueue[Self.device_spec]]

    @staticmethod
    def _self_ptr(
        py_self: PythonObject,
    ) -> UnsafePointer[Self, MutAnyOrigin]:
        try:
            return py_self.downcast_value_ptr[Self]()
        except e:
            abort(String("Queue method receiver was not a Queue: ", e))

    @staticmethod
    def synchronize(py_self: PythonObject) raises -> PythonObject:
        var self_ptr = Self._self_ptr(py_self)
        self_ptr[]._arc[].synchronize()
        return PythonObject(None)

    @staticmethod
    def record_event(py_self: PythonObject) raises -> PythonObject:
        var self_ptr = Self._self_ptr(py_self)
        var hal_evt = self_ptr[]._arc[].record_event[EVENT_FLAG_CPU_VISIBLE]()
        var ctx_arc = self_ptr[]._arc[]._context
        return PythonObject(alloc=Event(_hal=hal_evt^, _ctx_keepalive=ctx_arc))

    def write_to(self, mut writer: Some[Writer]):
        writer.write("Queue()")

    def write_repr_to(self, mut writer: Some[Writer]):
        writer.write("Queue()")


@fieldwise_init
struct Stream(Movable, Writable):
    """Python projection of HAL ``Stream``."""

    comptime device_spec = get_device_spec[0]()
    var _arc: ArcPointer[HALStream[Self.device_spec]]

    @staticmethod
    def _self_ptr(
        py_self: PythonObject,
    ) -> UnsafePointer[Self, MutAnyOrigin]:
        try:
            return py_self.downcast_value_ptr[Self]()
        except e:
            abort(String("Stream method receiver was not a Stream: ", e))

    @staticmethod
    def synchronize(py_self: PythonObject) raises -> PythonObject:
        var self_ptr = Self._self_ptr(py_self)
        self_ptr[]._arc[].synchronize()
        return PythonObject(None)

    @staticmethod
    def record_event(py_self: PythonObject) raises -> PythonObject:
        var self_ptr = Self._self_ptr(py_self)
        var hal_evt = self_ptr[]._arc[].record_event[EVENT_FLAG_CPU_VISIBLE]()
        var ctx_arc = self_ptr[]._arc[]._queue[]._context
        return PythonObject(alloc=Event(_hal=hal_evt^, _ctx_keepalive=ctx_arc))

    def write_to(self, mut writer: Some[Writer]):
        writer.write("Stream()")

    def write_repr_to(self, mut writer: Some[Writer]):
        writer.write("Stream()")


@fieldwise_init
struct Event(Movable, Writable):
    """Python projection of HAL ``Event``.

    Always created with ``EVENT_FLAG_CPU_VISIBLE`` so host
    ``synchronize()`` and ``is_ready()`` are always callable.

    Holds an explicit ``ArcPointer[HALContext]`` because HAL's
    ``_EventInner`` keeps the context handle as a raw value, not an
    Arc. If Context were destroyed while an Event is still alive,
    ``synchronize_event`` / ``destroy_event`` would use a stale
    handle — CUDA segfaults on this; Metal tolerates the UAF.
    """

    comptime device_spec = get_device_spec[0]()
    var _hal: HALEvent[EVENT_FLAG_CPU_VISIBLE]
    var _ctx_keepalive: ArcPointer[HALContext[Self.device_spec]]

    @staticmethod
    def _self_ptr(
        py_self: PythonObject,
    ) -> UnsafePointer[Self, MutAnyOrigin]:
        try:
            return py_self.downcast_value_ptr[Self]()
        except e:
            abort(String("Event method receiver was not an Event: ", e))

    @staticmethod
    def synchronize(py_self: PythonObject) raises -> PythonObject:
        var self_ptr = Self._self_ptr(py_self)
        self_ptr[]._hal.synchronize()
        return PythonObject(None)

    @staticmethod
    def is_ready(py_self: PythonObject) raises -> PythonObject:
        var self_ptr = Self._self_ptr(py_self)
        return PythonObject(self_ptr[]._hal.is_ready())

    def write_to(self, mut writer: Some[Writer]):
        writer.write("Event()")

    def write_repr_to(self, mut writer: Some[Writer]):
        writer.write("Event()")


@export
def PyInit_mojo_module() -> PythonObject:
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
        )

        _ = (
            b.add_type[Queue]("Queue")
            .def_method[Queue.synchronize]("synchronize")
            .def_method[Queue.record_event]("record_event")
        )

        _ = (
            b.add_type[Stream]("Stream")
            .def_method[Stream.synchronize]("synchronize")
            .def_method[Stream.record_event]("record_event")
        )

        _ = (
            b.add_type[Event]("Event")
            .def_method[Event.synchronize]("synchronize")
            .def_method[Event.is_ready]("is_ready")
        )

        return b.finalize()
    except e:
        abort(t"failed to create Python module: {e}")
