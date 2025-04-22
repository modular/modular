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
from python import Python, PythonObject, TypedPythonObject, PythonModule
from python._bindings import (
    PyMojoObject,
    py_c_function_wrapper,
    python_type_object,
)
from python._cpython import PyMethodDef, PyObjectPtr, PyTypeObject


@export
fn PyInit_mojo_module() -> PythonObject:
    # ----------------------------------
    # Create a Python module
    # ----------------------------------

    # This will initialize the Python interpreter and create
    # an extension module with the provided name.
    var module: PythonModule

    try:
        module = (
            PythonModule("mojo_module")
            .def_py_function[case_return_arg_tuple]("case_return_arg_tuple")
            .def_py_function[case_raise_empty_error]("case_raise_empty_error")
            .def_py_function[case_raise_string_error]("case_raise_string_error")
            .def_py_function[case_mojo_raise]("case_mojo_raise")
            .def_py_function[incr_int__wrapper]("incr_int")
            .def_py_function[add_to_int__wrapper]("add_to_int")
            .def_py_function[create_string__wrapper]("create_string")
        )
    except e:
        return abort[PythonObject]("failed to create Python module: ", e)

    add_person_type(module)
    add_int_type(module)
    add_string_type(module)

    return module


# ===----------------------------------------------------------------------=== #
# Functions
# ===----------------------------------------------------------------------=== #


@export
fn case_return_arg_tuple(
    py_self: PythonObject,
    args: TypedPythonObject["Tuple"],
) -> PythonObject:
    return args


@export
fn case_raise_empty_error(
    py_self: PythonObject,
    args: TypedPythonObject["Tuple"],
) -> PythonObject:
    var cpython = Python().impl.cpython()

    var error_type = cpython.get_error_global("PyExc_ValueError")

    cpython.PyErr_SetNone(error_type)

    return PythonObject(PyObjectPtr())


@export
fn case_raise_string_error(
    py_self: PythonObject,
    args: TypedPythonObject["Tuple"],
) -> PythonObject:
    var cpython = Python().impl.cpython()

    var error_type = cpython.get_error_global("PyExc_ValueError")

    cpython.PyErr_SetString(error_type, "sample value error".unsafe_cstr_ptr())

    return PythonObject(PyObjectPtr())


# Tests `create_wrapper_function()` of a `raises` function.
@export
fn case_mojo_raise(
    py_self: PythonObject,
    args: TypedPythonObject["Tuple"],
) raises -> PythonObject:
    raise "Mojo error"


# ===----------------------------------------------------------------------=== #
# Custom Types
# ===----------------------------------------------------------------------=== #


@value
struct Person:
    var name: String
    var age: Int

    fn __init__(out self):
        self.name = "John Smith"
        self.age = 123

    # Required by Pythonable
    fn __repr__(self) -> String:
        return String(
            "Person(",
            repr(self.name),
            ", ",
            repr(self.age),
            ")",
        )

    @staticmethod
    fn obj_name(
        self_: PythonObject, args: TypedPythonObject["Tuple"]
    ) -> PythonObject:
        var self0 = self_.unsafe_as_py_object_ptr().unchecked_cast_to_mojo_value[
            Person
        ]()

        return PythonObject(self0[].name).steal_data()


fn add_person_type(mut module: PythonModule):
    # ----------------------------------------------
    # Construct a 'type' object describing `Person`
    # ----------------------------------------------

    var methods = List[PyMethodDef](
        PyMethodDef.function(py_c_function_wrapper[Person.obj_name], "name"),
        # Zeroed item as terminator
        PyMethodDef(),
    )

    # ----------------------------------
    # Register the type in the module
    # ----------------------------------

    try:
        var type_obj = python_type_object[Person]("Person", methods)

        Python.add_object(module, "Person", type_obj)
    except e:
        abort("error adding object: ", e)


# ===----------------------------------------------------------------------=== #
# Recipe Book
# ===----------------------------------------------------------------------=== #

# ====================================
# Recipe: Argument: Arity and argument type checking
# ====================================


fn add_int_type(mut module: PythonModule):
    try:
        var type_obj = python_type_object[Int]("Int")

        Python.add_object(module, "Int", type_obj)
    except e:
        abort("error adding object: ", e)


fn incr_int(mut arg: Int):
    arg += 1


fn add_to_int(mut arg: Int, owned value: Int):
    arg += value


#
# Manual Wrappers
#


fn incr_int__wrapper(
    py_self: PythonObject,
    py_args: TypedPythonObject["Tuple"],
) raises -> PythonObject:
    check_arguments_arity("incr_int".value, 1, py_args)

    var arg_0: UnsafePointer[Int] = check_and_get_arg[Int](
        "incr_int", "Int", py_args, 0
    )

    # Note: Pass an `mut` reference to the wrapped function
    incr_int(arg_0[])

    return PythonObject(None)


fn add_to_int__wrapper(
    py_self: PythonObject,
    py_args: TypedPythonObject["Tuple"],
) raises -> PythonObject:
    check_arguments_arity("add_to_int".value, 2, py_args)

    var arg_0: UnsafePointer[Int] = check_and_get_arg[Int](
        "add_to_int", "Int", py_args, 0
    )

    var arg_1: UnsafePointer[Int] = check_and_get_or_convert_arg[Int](
        "add_to_int",
        "Int",
        py_args,
        1,
    )

    # Note: Pass an `mut` reference to the wrapped function
    add_to_int(arg_0[], arg_1[])

    return PythonObject(None)


# ====================================
# Recipe: Function: Returning New Mojo Values
# ====================================


fn add_string_type(mut module: PythonModule):
    try:
        var type_obj = python_type_object[String]("String")

        Python.add_object(module, "String", type_obj)
    except e:
        abort("error adding object: ", e)


fn create_string() raises -> String:
    return "Hello"


fn create_string__wrapper(
    py_self: PythonObject,
    py_args: TypedPythonObject["Tuple"],
) raises -> PythonObject:
    alias func_name = "create_int"
    alias func_arity = 0

    check_arguments_arity(func_name.value, func_arity, py_args)

    var cpython = Python().impl.cpython()

    var result = create_string()

    # TODO(MSTDL-1018):
    #   Improve how we're looking up the PyTypeObject for a Mojo type.
    # NOTE:
    #   We can't just use python_type_object[String] because that constructs
    #   a _new_ PyTypeObject. We want to reference the existing _singleton_
    #   PyTypeObject that represents a given Mojo type.
    var string_ty = Python.import_module("mojo_module").String

    # SAFETY:
    #   `Int` was added to the module by us, so it should be an instance
    #   of PyTypeObject. (Caveat: This theoretically might not be true if the
    #   user has manually re-assigned the `Int` attribute.)
    var string_ty_ptr = string_ty.unsafe_as_py_object_ptr().unsized_obj_ptr.bitcast[
        PyTypeObject
    ]()

    # Allocate storage to hold a PyMojoObject[String]
    var string_obj_raw_ptr: PyObjectPtr = cpython.PyType_GenericAlloc(
        string_ty_ptr,
        0,
    )

    # Initialize the PyMojoObject[String]
    string_obj_raw_ptr.unchecked_cast_to_mojo_value[String]().init_pointee_move(
        result
    )

    return PythonObject(string_obj_raw_ptr)
