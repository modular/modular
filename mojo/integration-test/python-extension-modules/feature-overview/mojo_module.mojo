# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

import builtin

from sys import exit
from sys.info import sizeof
from sys.ffi import OpaquePointer, c_int
from memory import UnsafePointer

from os import abort

from python import Python, PythonObject, TypedPythonObject
from python._bindings import (
    py_c_function_wrapper,
    PyMojoObject,
)
from builtin._pybind import (
    check_arguments_arity,
    check_argument_type,
    check_and_get_arg,
    check_and_get_or_convert_arg,
)
from python._cpython import (
    PyMethodDef,
    PyObject,
    PyObjectPtr,
)


@export
fn PyInit_mojo_module() -> PythonObject:
    # ----------------------------------
    # Create a Python module
    # ----------------------------------

    # This will initialize the Python interpreter and create
    # an extension module with the provided name.
    var module: TypedPythonObject["Module"]

    try:
        module = Python.create_module("bindings")
    except e:
        return abort[PythonObject]("failed to create Python module: " + str(e))

    # ----------------------------------
    # Populate the Python module
    # ----------------------------------

    var funcs = List[PyMethodDef](
        PyMethodDef.function[
            py_c_function_wrapper[case_return_arg_tuple],
            "case_return_arg_tuple",
        ](),
        PyMethodDef.function[
            py_c_function_wrapper[case_raise_empty_error],
            "case_raise_empty_error",
        ](),
        PyMethodDef.function[
            py_c_function_wrapper[case_raise_string_error],
            "case_raise_string_error",
        ](),
        PyMethodDef.function[
            py_c_function_wrapper[case_mojo_raise],
            "case_mojo_raise",
        ](),
        PyMethodDef.function[
            py_c_function_wrapper[incr_int__wrapper],
            "incr_int",
        ](),
        PyMethodDef.function[
            py_c_function_wrapper[add_to_int__wrapper],
            "add_to_int",
        ](),
    )

    try:
        Python.add_functions(module, funcs)
    except e:
        abort("Error adding functions to PyModule: " + str(e))

    add_person_type(module)
    add_int_type(module)

    return module


# ===----------------------------------------------------------------------=== #
# Functions
# ===----------------------------------------------------------------------=== #


@export
fn case_return_arg_tuple(
    py_self: PythonObject,
    args: TypedPythonObject["Tuple"],
) -> PythonObject:
    var cpython = Python().impl.cpython()

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

    fn __init__(inout self):
        self.name = "John Smith"
        self.age = 123

    # Required by Pythonable
    fn __repr__(self) -> String:
        return String.write(
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


fn add_person_type(inout module: TypedPythonObject["Module"]):
    # ----------------------------------------------
    # Construct a 'type' object describing `Person`
    # ----------------------------------------------

    var methods = List[PyMethodDef](
        PyMethodDef.function[
            py_c_function_wrapper[Person.obj_name],
            "name",
        ](),
        # Zeroed item as terminator
        PyMethodDef(),
    )

    # ----------------------------------
    # Register the type in the module
    # ----------------------------------

    try:
        var type_obj = PyMojoObject[Person].python_type_object["Person"](
            methods
        )

        Python.add_object(module, "Person", type_obj)
    except e:
        abort("error adding object: " + str(e))


# ===----------------------------------------------------------------------=== #
# Recipe Book
# ===----------------------------------------------------------------------=== #

# ====================================
# Recipe: Argument: Arity and argument type checking
# ====================================


fn add_int_type(inout module: TypedPythonObject["Module"]):
    try:
        var type_obj = PyMojoObject[Int].python_type_object["Int"](
            methods=List[PyMethodDef]()
        )

        Python.add_object(module, "Int", type_obj)
    except e:
        abort("error adding object: " + str(e))


fn incr_int(inout arg: Int):
    arg += 1


fn add_to_int(inout arg: Int, owned value: Int):
    arg += value


#
# Manual Wrappers
#


fn incr_int__wrapper(
    py_self: PythonObject,
    py_args: TypedPythonObject["Tuple"],
) raises -> PythonObject:
    check_arguments_arity("incr_int", 1, py_args)

    var arg_0: UnsafePointer[Int] = check_and_get_arg[Int](
        "incr_int", "Int", py_args, 0
    )

    # Note: Pass an `inout` reference to the wrapped function
    incr_int(arg_0[])

    return PythonObject(None)


fn add_to_int__wrapper(
    py_self: PythonObject,
    py_args: TypedPythonObject["Tuple"],
) raises -> PythonObject:
    check_arguments_arity("add_to_int", 2, py_args)

    var arg_0: UnsafePointer[Int] = check_and_get_arg[Int](
        "add_to_int", "Int", py_args, 0
    )

    # Stack space to hold a converted value for this argument, if needed.
    # TODO: It should not be necessary to provide a default value for this.
    var arg_1_owned: Int = 0
    var arg_1: UnsafePointer[Int] = check_and_get_or_convert_arg[Int](
        "add_to_int",
        "Int",
        py_args,
        1,
        UnsafePointer.address_of(arg_1_owned),
    )

    # Note: Pass an `inout` reference to the wrapped function
    add_to_int(arg_0[], arg_1[])

    return PythonObject(None)
