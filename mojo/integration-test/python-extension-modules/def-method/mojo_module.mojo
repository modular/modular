# ===----------------------------------------------------------------------=== #
# Copyright (c) 2025, Modular Inc. All rights reserved.
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

from os import abort

from memory import UnsafePointer
from python import Python, PythonObject
from python.bindings import PythonModuleBuilder


@export
fn PyInit_mojo_module() -> PythonObject:
    try:
        var b = PythonModuleBuilder("mojo_module")

        _ = (
            b.add_type[Person]("Person")
            # def_method with return, raising
            .def_method[Person.get_name]("get_name")
            .def_method[Person.split_name]("split_name")
            .def_method[Person._with]("_with")
            # def_method with return, not raising
            .def_method[Person.get_age]("get_age")
            .def_method[Person._get_birth_year]("_get_birth_year")
            .def_method[Person._with_first_last_name]("_with_first_last_name")
            # # def_method with no return, raising
            .def_method[Person.erase_name]("erase_name")
            .def_method[Person.set_age]("set_age")
            .def_method[Person.set_name_and_age]("set_name_and_age")
            # # def_method with no return, not raising
            .def_method[Person.reset]("reset")
            .def_method[Person.set_name]("set_name")
            .def_method[Person._set_age_from_dates]("_set_age_from_dates")
        )
        return b.finalize()
    except e:
        return abort[PythonObject](
            String("failed to create Python module: ", e)
        )


@fieldwise_init
struct Person(Defaultable, Representable, Copyable, Movable):
    var name: String
    var age: Int

    fn __init__(out self):
        self.name = "John Smith"
        self.age = 123

    fn __repr__(self) -> String:
        return String(
            "Person(",
            repr(self.name),
            ", ",
            repr(self.age),
            ")",
        )

    @staticmethod
    fn _get_self_ptr(py_self: PythonObject) -> UnsafePointer[Self]:
        try:
            return py_self.downcast_value_ptr[Self]()
        except e:
            return abort[UnsafePointer[Self]](
                String(
                    (
                        "Python method receiver object did not have the"
                        " expected type:"
                    ),
                    e,
                )
            )

    @staticmethod
    fn get_name(py_self: PythonObject) raises -> PythonObject:
        # TODO: replace with property once we have them
        var self_ptr = Self._get_self_ptr(py_self)

        var s = Python().evaluate("hasattr(sys.modules[__name__], 'deny_name')")
        if s:
            raise "name cannot be accessed"

        return PythonObject(self_ptr[].name)

    @staticmethod
    fn split_name(
        py_self: PythonObject, sep: PythonObject
    ) raises -> PythonObject:
        var self_ptr = Self._get_self_ptr(py_self)
        return Python.list(self_ptr[].name.split(String(sep)))

    @staticmethod
    fn _with(
        py_self: PythonObject, name: PythonObject, age: PythonObject
    ) raises -> PythonObject:
        Self.set_name_and_age(py_self, name, age)
        return py_self

    @staticmethod
    fn get_age(py_self: PythonObject) -> PythonObject:
        # TODO: replace with property once we have them
        return PythonObject(Self._get_self_ptr(py_self)[].age)

    @staticmethod
    fn _get_birth_year(
        py_self: PythonObject, this_year: PythonObject
    ) -> PythonObject:
        var self_ptr = Self._get_self_ptr(py_self)
        try:
            return PythonObject(Int(this_year) - self_ptr[].age)
        except e:
            return abort[PythonObject](String("failed to get birth year: ", e))

    @staticmethod
    fn _with_first_last_name(
        py_self: PythonObject, first_name: PythonObject, last_name: PythonObject
    ) -> PythonObject:
        var self_ptr = Self._get_self_ptr(py_self)
        try:
            self_ptr[].name = String(first_name) + " " + String(last_name)
        except e:
            return abort[PythonObject](String("failed to set name: ", e))
        return py_self

    @staticmethod
    fn erase_name(py_self: PythonObject) raises:
        var self_ptr = Self._get_self_ptr(py_self)
        if not self_ptr[].name:
            raise "cannot erase name if it's already empty"

        self_ptr[].name = String()

    @staticmethod
    fn set_age(py_self: PythonObject, age: PythonObject) raises:
        var self_ptr = Self._get_self_ptr(py_self)
        try:
            self_ptr[].age = Int(age)
        except e:
            raise "cannot set age to " + String(age)

    @staticmethod
    fn set_name_and_age(
        py_self: PythonObject, name: PythonObject, age: PythonObject
    ) raises:
        var self_ptr = Self._get_self_ptr(py_self)
        self_ptr[].name = String(name)
        Self.set_age(py_self, age)

    @staticmethod
    fn reset(py_self: PythonObject):
        var self_ptr = Self._get_self_ptr(py_self)
        self_ptr[].name = "John Smith"
        self_ptr[].age = 123

    @staticmethod
    fn set_name(py_self: PythonObject, name: PythonObject):
        # TODO: replace with property once we have them
        try:
            Self._get_self_ptr(py_self)[].name = String(name)
        except e:
            abort(String("failed to set name: ", e))

    @staticmethod
    fn _set_age_from_dates(
        py_self: PythonObject, birth_year: PythonObject, this_year: PythonObject
    ):
        var self_ptr = Self._get_self_ptr(py_self)
        try:
            self_ptr[].age = Int(this_year) - Int(birth_year)
        except e:
            abort(String("failed to set age: ", e))
