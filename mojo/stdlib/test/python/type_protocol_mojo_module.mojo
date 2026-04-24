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

# ===----------------------------------------------------------------------=== #
# Test for TypeProtocolBuilder.
#
# Exposes a Box type (wrapping Float64) to Python that supports all six
# rich comparison operators via tp_richcompare:
#   - box1 < box2    Py_LT
#   - box1 <= box2   Py_LE
#   - box1 == box2   Py_EQ
#   - box1 != box2   Py_NE
#   - box1 > box2    Py_GT
#   - box1 >= box2   Py_GE
# ===----------------------------------------------------------------------=== #

from std.os import abort
from std.memory import UnsafePointer
from std.python import PythonObject
from std.python.bindings import PythonModuleBuilder

from std.python.builders import TypeProtocolBuilder
from std.python.utils import NotImplementedError, RichCompareOps


struct Box(Defaultable, Movable, Writable):
    var value: Float64

    def __init__(out self):
        self.value = 0.0

    def __init__(out self, value: Float64):
        self.value = value

    @staticmethod
    def _get_self_ptr(
        py_self: PythonObject,
    ) -> UnsafePointer[Self, MutAnyOrigin]:
        try:
            return py_self.downcast_value_ptr[Self]()
        except e:
            abort(String("downcast failed: ", e))

    @staticmethod
    def new(value: PythonObject) raises -> PythonObject:
        return PythonObject(alloc=Box(Float64(py=value)))

    @staticmethod
    def get_value(py_self: PythonObject) raises -> PythonObject:
        return PythonObject(Self._get_self_ptr(py_self)[].value)

    @staticmethod
    def rich_compare(
        self,
        other: PythonObject,
        op: Int,
    ) raises -> Bool:
        var a = self.value
        var b = other.downcast_value_ptr[Self]()[].value
        if op == RichCompareOps.Py_LT:
            return a < b
        if op == RichCompareOps.Py_LE:
            return a <= b
        if op == RichCompareOps.Py_EQ:
            return a == b
        if op == RichCompareOps.Py_NE:
            return a != b
        if op == RichCompareOps.Py_GT:
            return a > b
        if op == RichCompareOps.Py_GE:
            return a >= b
        raise NotImplementedError()

    def write_to(self, mut writer: Some[Writer]):
        writer.write("Box(", self.value, ")")


# BoxV uses a value-receiver rich_compare handler.
struct BoxV(Defaultable, Movable, Writable):
    var value: Float64

    def __init__(out self):
        self.value = 0.0

    def __init__(out self, value: Float64):
        self.value = value

    @staticmethod
    def _get_self_ptr(
        py_self: PythonObject,
    ) -> UnsafePointer[Self, MutAnyOrigin]:
        try:
            return py_self.downcast_value_ptr[Self]()
        except e:
            abort(String("downcast failed: ", e))

    @staticmethod
    def new(value: PythonObject) raises -> PythonObject:
        return PythonObject(alloc=BoxV(Float64(py=value)))

    @staticmethod
    def get_value(py_self: PythonObject) raises -> PythonObject:
        return PythonObject(Self._get_self_ptr(py_self)[].value)

    # Raising value-receiver rich_compare
    def rich_compare(self, other: PythonObject, op: Int) raises -> Bool:
        var a = self.value
        var b = other.downcast_value_ptr[Self]()[].value
        if op == RichCompareOps.Py_LT:
            return a < b
        if op == RichCompareOps.Py_LE:
            return a <= b
        if op == RichCompareOps.Py_EQ:
            return a == b
        if op == RichCompareOps.Py_NE:
            return a != b
        if op == RichCompareOps.Py_GT:
            return a > b
        if op == RichCompareOps.Py_GE:
            return a >= b
        raise NotImplementedError()

    def write_to(self, mut writer: Some[Writer]):
        writer.write("BoxV(", self.value, ")")


@export
def PyInit_type_protocol_mojo_module() -> PythonObject:
    try:
        var b = PythonModuleBuilder("type_protocol_mojo_module")
        ref tb = (
            b.add_type[Box]("Box")
            .def_init_defaultable[Box]()
            .def_staticmethod[Box.new]("new")
            .def_method[Box.get_value]("get_value")
        )
        var tpb = TypeProtocolBuilder[Box](tb)
        _ = tpb.def_richcompare[Box.rich_compare]()
        ref tbv = (
            b.add_type[BoxV]("BoxV")
            .def_init_defaultable[BoxV]()
            .def_staticmethod[BoxV.new]("new")
            .def_method[BoxV.get_value]("get_value")
        )
        var tpbv = TypeProtocolBuilder[BoxV](tbv)
        _ = tpbv.def_richcompare[BoxV.rich_compare]()
        return b.finalize()
    except e:
        abort(String("failed to create Python module: ", e))
