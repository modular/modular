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
"""Python projection of HAL ``Device``."""

from std.memory import ArcPointer, UnsafePointer
from std.os import abort
from std.python import PythonObject
from _hal.context import Context as HALContext
from _hal.device import Device as HALDevice, get_device_spec

from .context import Context


@fieldwise_init
struct Device(Movable, Writable):
    """Python projection of HAL ``Device``."""

    # TODO: generalize to multi-device — currently hardcoded to device 0.
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
