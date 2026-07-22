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
"""Python projection of HAL ``Driver``."""

from std.memory import ArcPointer, UnsafePointer
from std.os import abort, getenv
from std.python import PythonObject
from std.python.bindings import check_arguments_arity
from _hal.driver import Driver as HALDriver
from _hal.device import Device as HALDevice

from .device import Device


def load_driver() raises -> ArcPointer[HALDriver]:
    # TODO: ``MODULAR_DRIVER_PLUGINS`` will become a colon-separated list of
    # plugin specs (one per HAL backend); when that lands, replace this
    # helper with a plural ``load_drivers()``.
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
        self = Self(_arc=load_driver())

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
