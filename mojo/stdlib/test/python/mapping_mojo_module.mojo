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
# Test for MappingProtocolBuilder.
#
# Exposes a SimpleList type to Python that supports:
#   - len(obj)       via mp_length
#   - obj[i]         via mp_subscript
#   - obj[i] = v     via mp_ass_subscript (assignment)
#   - del obj[i]     via mp_ass_subscript (deletion)
# ===----------------------------------------------------------------------=== #

from std.os import abort
from std.memory import UnsafePointer
from std.utils import Variant
from std.python import PythonObject
from std.python.bindings import PythonModuleBuilder

from std.python.builders import MappingProtocolBuilder


struct SimpleList(Defaultable, Movable, Writable):
    var data: List[Int]

    def __init__(out self):
        self.data = []

    @staticmethod
    def from_list(items: PythonObject) raises -> PythonObject:
        var result = SimpleList()
        for item in items:
            result.data.append(Int(py=item))
        return PythonObject(alloc=result^)

    @staticmethod
    def py__len__(self_ptr: UnsafePointer[Self, MutAnyOrigin]) raises -> Int:
        return len(self_ptr[].data)

    @staticmethod
    def py__getitem__(
        self_ptr: UnsafePointer[Self, MutAnyOrigin], index: PythonObject
    ) raises -> PythonObject:
        var i = Int(py=index)
        if i < 0 or i >= len(self_ptr[].data):
            raise Error("index out of range")
        return PythonObject(self_ptr[].data[i])

    @staticmethod
    def py__setitem__(
        self_ptr: UnsafePointer[Self, MutAnyOrigin],
        index: PythonObject,
        value: Variant[PythonObject, Int],
    ) raises -> None:
        var i = Int(py=index)
        if i < 0 or i >= len(self_ptr[].data):
            raise Error("index out of range")
        if value.isa[PythonObject]():
            self_ptr[].data[i] = Int(py=value[PythonObject])
        else:
            _ = self_ptr[].data.pop(i)

    def write_to(self, mut writer: Some[Writer]):
        writer.write("SimpleList(len=", len(self.data), ")")


# SimpleListV uses value-receiver handlers (def(self: Self, ...) instead of
# def(self_ptr: UnsafePointer[Self, MutAnyOrigin], ...)).
# Read-only handlers use the non-raising overload; mutating ones use raising
# ptr-receiver since they need to modify the Python-owned object in place.
struct SimpleListV(Defaultable, Movable, Writable):
    var data: List[Int]

    def __init__(out self):
        self.data = []

    @staticmethod
    def from_list(items: PythonObject) raises -> PythonObject:
        var result = SimpleListV()
        for item in items:
            result.data.append(Int(py=item))
        return PythonObject(alloc=result^)

    # Non-raising value receiver
    def py__len__(self) -> Int:
        return len(self.data)

    # Raising value receiver
    def py__getitem__(self, index: PythonObject) raises -> PythonObject:
        var i = Int(py=index)
        if i < 0 or i >= len(self.data):
            raise Error("index out of range")
        return PythonObject(self.data[i])

    # Mutation still uses pointer receiver so changes are visible on the Python object
    @staticmethod
    def py__setitem__(
        self_ptr: UnsafePointer[Self, MutAnyOrigin],
        index: PythonObject,
        value: Variant[PythonObject, Int],
    ) raises -> None:
        var i = Int(py=index)
        if i < 0 or i >= len(self_ptr[].data):
            raise Error("index out of range")
        if value.isa[PythonObject]():
            self_ptr[].data[i] = Int(py=value[PythonObject])
        else:
            _ = self_ptr[].data.pop(i)

    def write_to(self, mut writer: Some[Writer]):
        writer.write("SimpleListV(len=", len(self.data), ")")


@export
def PyInit_mapping_mojo_module() -> PythonObject:
    try:
        var b = PythonModuleBuilder("mapping_mojo_module")
        ref tb = (
            b.add_type[SimpleList]("SimpleList")
            .def_init_defaultable[SimpleList]()
            .def_staticmethod[SimpleList.from_list]("from_list")
        )
        var mpb = MappingProtocolBuilder[SimpleList](tb)
        _ = (
            mpb.def_len[SimpleList.py__len__]()
            .def_getitem[SimpleList.py__getitem__]()
            .def_setitem[SimpleList.py__setitem__]()
        )
        ref tbv = (
            b.add_type[SimpleListV]("SimpleListV")
            .def_init_defaultable[SimpleListV]()
            .def_staticmethod[SimpleListV.from_list]("from_list")
        )
        var mpbv = MappingProtocolBuilder[SimpleListV](tbv)
        _ = (
            mpbv.def_len[SimpleListV.py__len__]()
            .def_getitem[SimpleListV.py__getitem__]()
            .def_setitem[SimpleListV.py__setitem__]()
        )
        return b.finalize()
    except e:
        abort(String("failed to create Python module: ", e))
