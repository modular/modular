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
# Test for SequenceProtocolBuilder.
#
# Exposes a Seq type to Python that supports:
#   - len(obj)       via sq_length
#   - obj[i]         via sq_item        (integer index)
#   - obj[i] = v     via sq_ass_item    (assignment)
#   - del obj[i]     via sq_ass_item    (deletion)
#   - v in obj       via sq_contains
#   - obj + other    via sq_concat
#   - obj * n        via sq_repeat
# ===----------------------------------------------------------------------=== #

from std.os import abort
from std.memory import UnsafePointer
from std.utils import Variant
from std.python import PythonObject
from std.python.bindings import PythonModuleBuilder

from std.python.builders import SequenceProtocolBuilder


struct Seq(Defaultable, Movable, Writable):
    var data: List[Int]

    def __init__(out self):
        self.data = []

    @staticmethod
    def from_list(items: PythonObject) raises -> PythonObject:
        var result = Seq()
        for item in items:
            result.data.append(Int(py=item))
        return PythonObject(alloc=result^)

    @staticmethod
    def py__len__(self_ptr: UnsafePointer[Self, MutAnyOrigin]) raises -> Int:
        return len(self_ptr[].data)

    @staticmethod
    def py__getitem__(
        self_ptr: UnsafePointer[Self, MutAnyOrigin], index: Int
    ) raises -> PythonObject:
        if index < 0 or index >= len(self_ptr[].data):
            raise Error("index out of range")
        return PythonObject(self_ptr[].data[index])

    @staticmethod
    def py__setitem__(
        self_ptr: UnsafePointer[Self, MutAnyOrigin],
        index: Int,
        value: Variant[PythonObject, Int],
    ) raises -> None:
        if index < 0 or index >= len(self_ptr[].data):
            raise Error("index out of range")
        if value.isa[PythonObject]():
            self_ptr[].data[index] = Int(py=value[PythonObject])
        else:
            _ = self_ptr[].data.pop(index)

    @staticmethod
    def py__contains__(
        self_ptr: UnsafePointer[Self, MutAnyOrigin], item: PythonObject
    ) raises -> Bool:
        var v = Int(py=item)
        for elem in self_ptr[].data:
            if elem == v:
                return True
        return False

    @staticmethod
    def py__concat__(
        self_ptr: UnsafePointer[Self, MutAnyOrigin], other: PythonObject
    ) raises -> PythonObject:
        var other_ptr = other.downcast_value_ptr[Self]()
        var result = Seq()
        for v in self_ptr[].data:
            result.data.append(v)
        for v in other_ptr[].data:
            result.data.append(v)
        return PythonObject(alloc=result^)

    @staticmethod
    def py__repeat__(
        self_ptr: UnsafePointer[Self, MutAnyOrigin], count: Int
    ) raises -> PythonObject:
        var result = Seq()
        for _ in range(count):
            for v in self_ptr[].data:
                result.data.append(v)
        return PythonObject(alloc=result^)

    def write_to(self, mut writer: Some[Writer]):
        writer.write("Seq(len=", len(self.data), ")")


# SeqV uses value-receiver handlers where no mutation is needed, and pointer
# receivers where the object must be modified in place.
struct SeqV(Defaultable, Movable, Writable):
    var data: List[Int]

    def __init__(out self):
        self.data = []

    @staticmethod
    def from_list(items: PythonObject) raises -> PythonObject:
        var result = SeqV()
        for item in items:
            result.data.append(Int(py=item))
        return PythonObject(alloc=result^)

    # Non-raising value receiver
    def py__len__(self) -> Int:
        return len(self.data)

    # Raising value receiver
    def py__getitem__(self, index: Int) raises -> PythonObject:
        if index < 0 or index >= len(self.data):
            raise Error("index out of range")
        return PythonObject(self.data[index])

    # Mutation uses pointer receiver
    @staticmethod
    def py__setitem__(
        self_ptr: UnsafePointer[Self, MutAnyOrigin],
        index: Int,
        value: Variant[PythonObject, Int],
    ) raises -> None:
        if index < 0 or index >= len(self_ptr[].data):
            raise Error("index out of range")
        if value.isa[PythonObject]():
            self_ptr[].data[index] = Int(py=value[PythonObject])
        else:
            _ = self_ptr[].data.pop(index)

    # Non-raising value receiver for contains
    def py__contains__(self, item: PythonObject) raises -> Bool:
        var v = Int(py=item)
        for elem in self.data:
            if elem == v:
                return True
        return False

    # Raising value receiver for concat
    def py__concat__(self, other: PythonObject) raises -> PythonObject:
        var other_ptr = other.downcast_value_ptr[Self]()
        var result = SeqV()
        for v in self.data:
            result.data.append(v)
        for v in other_ptr[].data:
            result.data.append(v)
        return PythonObject(alloc=result^)

    # Non-raising value receiver for repeat
    def py__repeat__(self, count: Int) raises -> PythonObject:
        var result = SeqV()
        for _ in range(count):
            for v in self.data:
                result.data.append(v)
        return PythonObject(alloc=result^)

    def write_to(self, mut writer: Some[Writer]):
        writer.write("SeqV(len=", len(self.data), ")")


@export
def PyInit_sequence_mojo_module() -> PythonObject:
    try:
        var b = PythonModuleBuilder("sequence_mojo_module")
        ref tb = (
            b.add_type[Seq]("Seq")
            .def_init_defaultable[Seq]()
            .def_staticmethod[Seq.from_list]("from_list")
        )
        var spb = SequenceProtocolBuilder[Seq](tb)
        _ = (
            spb.def_len[Seq.py__len__]()
            .def_getitem[Seq.py__getitem__]()
            .def_setitem[Seq.py__setitem__]()
            .def_contains[Seq.py__contains__]()
            .def_concat[Seq.py__concat__]()
            .def_repeat[Seq.py__repeat__]()
        )
        ref tbv = (
            b.add_type[SeqV]("SeqV")
            .def_init_defaultable[SeqV]()
            .def_staticmethod[SeqV.from_list]("from_list")
        )
        var spbv = SequenceProtocolBuilder[SeqV](tbv)
        _ = (
            spbv.def_len[SeqV.py__len__]()
            .def_getitem[SeqV.py__getitem__]()
            .def_setitem[SeqV.py__setitem__]()
            .def_contains[SeqV.py__contains__]()
            .def_concat[SeqV.py__concat__]()
            .def_repeat[SeqV.py__repeat__]()
        )
        return b.finalize()
    except e:
        abort(String("failed to create Python module: ", e))
