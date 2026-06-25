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
"""Python projection of HAL ``Event``."""

from std.memory import ArcPointer, UnsafePointer
from std.os import abort
from std.python import PythonObject
from _hal.context import Context as HALContext
from _hal.device import get_device_spec
from _hal.event import (
    Event as HALEvent,
    EVENT_FLAG_CPU_VISIBLE,
)


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

    # TODO: generalize to multi-device — currently hardcoded to device 0.
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
    def synchronize(py_self: PythonObject) raises:
        var self_ptr = Self._self_ptr(py_self)
        self_ptr[]._hal.synchronize()

    @staticmethod
    def is_ready(py_self: PythonObject) raises -> PythonObject:
        var self_ptr = Self._self_ptr(py_self)
        return PythonObject(self_ptr[]._hal.is_ready())

    def write_to(self, mut writer: Some[Writer]):
        writer.write("Event()")

    def write_repr_to(self, mut writer: Some[Writer]):
        writer.write("Event()")
