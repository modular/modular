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

from builtin._pybind import (
    check_and_get_arg,
    check_and_get_or_convert_arg,
    check_arguments_arity,
)
from memory import UnsafePointer
from python import Python, PythonObject
from python.bindings import (
    PyMojoObject,
    PythonModuleBuilder,
    lookup_py_type_object,
)
from python._cpython import PyObjectPtr, PyTypeObject


@export
fn PyInit_mojo_module() -> PythonObject:
    # ----------------------------------
    # Create a Python module
    # ----------------------------------

    # This will initialize the Python interpreter and create
    # an extension module with the provided name.

    try:
        var b = PythonModuleBuilder("mojo_module")
        b.def_py_function[case_return_arg_tuple]("case_return_arg_tuple")
        b.def_function[case_raise_empty_error]("case_raise_empty_error")
        b.def_function[case_raise_string_error]("case_raise_string_error")
        b.def_function[case_mojo_raise]("case_mojo_raise")
        b.def_function[case_mojo_mutate]("case_mojo_mutate")
        b.def_function[case_downcast_unbound_type]("case_downcast_unbound_type")
        b.def_py_function[incr_int__wrapper]("incr_int")
        b.def_py_function[add_to_int__wrapper]("add_to_int")
        b.def_function[create_string]("create_string")

        _ = (
            b.add_type[Person]("Person")
            .def_method[Person.obj_name]("name")
            .def_method[Person.change_name]("change_name")
        )
        _ = b.add_type[Int]("Int")
        _ = b.add_type[String]("String")
        _ = b.add_type[FailToInitialize]("FailToInitialize")
        return b.finalize()
    except e:
        return abort[PythonObject](
            String("failed to create Python module: ", e)
        )


# ===----------------------------------------------------------------------=== #
# Functions
# ===----------------------------------------------------------------------=== #


fn case_return_arg_tuple(
    py_self: PythonObject, args: PythonObject
) -> PythonObject:
    return args


fn case_raise_empty_error() -> PythonObject:
    var cpython = Python().cpython()

    var error_type = cpython.get_error_global("PyExc_ValueError")

    cpython.PyErr_SetNone(error_type)

    return PythonObject(from_owned_ptr=PyObjectPtr())


fn case_raise_string_error() -> PythonObject:
    var cpython = Python().cpython()

    var error_type = cpython.get_error_global("PyExc_ValueError")

    cpython.PyErr_SetString(error_type, "sample value error".unsafe_cstr_ptr())

    return PythonObject(from_owned_ptr=PyObjectPtr())


# Returning New Mojo Values
fn create_string() raises -> PythonObject:
    var result = String("Hello")

    return PythonObject(alloc=result^)


fn case_mojo_raise() raises -> PythonObject:
    raise "Mojo error"


fn case_mojo_mutate(list: PythonObject) raises -> PythonObject:
    # this would work even if args was `read`, but we want just to test that
    # the binding API accepts a function that mutates the argument.
    list[0] += 1

    return PythonObject(None)


struct NonBoundType:
    pass


fn case_downcast_unbound_type(value: PythonObject) raises:
    var _ptr = value.downcast_value_ptr[NonBoundType]()


# ===----------------------------------------------------------------------=== #
# Custom Types
# ===----------------------------------------------------------------------=== #


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
    fn obj_name(self_: PythonObject) raises -> PythonObject:
        var self0 = self_.downcast_value_ptr[Self]()

        return PythonObject(self0[].name)

    @staticmethod
    fn change_name(
        self_: PythonObject, new_name: PythonObject
    ) raises -> PythonObject:
        var self0 = UnsafePointer[Self, **_](
            unchecked_downcast_value=self_
        ).origin_cast[mut=True]()

        if len(new_name) > len(self0[].name.codepoints()):
            raise "cannot make name longer than current name"

        self0[].name = String(new_name)

        return PythonObject(None)


# ===----------------------------------------------------------------------=== #
# Test: Object Creation Behavior
# ===----------------------------------------------------------------------=== #


struct FailToInitialize(Movable, Defaultable, Representable):
    fn __init__(out self):
        pass

    fn __del__(owned self):
        abort("FailToInitialize should never be deinitialized.")

    fn __repr__(self) -> String:
        return "FailToInitialize()"


# ===----------------------------------------------------------------------=== #
# Recipe Book
# ===----------------------------------------------------------------------=== #

# ====================================
# Recipe: Argument: Arity and argument type checking
# ====================================


fn incr_int(mut arg: Int):
    arg += 1


fn add_to_int(mut arg: Int, owned value: Int):
    arg += value


#
# Manual Wrappers
#


fn incr_int__wrapper(
    py_self: PythonObject, py_args: PythonObject
) raises -> PythonObject:
    check_arguments_arity(1, py_args, "incr_int".value)

    var arg_0: UnsafePointer[Int] = check_and_get_arg[Int](
        "incr_int", py_args, 0
    )

    # Note: Pass an `mut` reference to the wrapped function
    incr_int(arg_0[])

    return PythonObject(None)


fn add_to_int__wrapper(
    py_self: PythonObject, py_args: PythonObject
) raises -> PythonObject:
    check_arguments_arity(2, py_args, "add_to_int".value)

    var arg_0: UnsafePointer[Int] = check_and_get_arg[Int](
        "add_to_int", py_args, 0
    )

    var arg_1: UnsafePointer[Int] = check_and_get_or_convert_arg[Int](
        "add_to_int",
        py_args,
        1,
    )

    # Note: Pass an `mut` reference to the wrapped function
    add_to_int(arg_0[], arg_1[])

    return PythonObject(None)
