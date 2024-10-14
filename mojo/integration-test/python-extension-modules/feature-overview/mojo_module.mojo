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
    create_wrapper_function,
    PyMojoObject,
    check_arguments_arity,
    check_argument_type,
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
            create_wrapper_function[case_return_arg_tuple](),
            "case_return_arg_tuple",
        ](),
        PyMethodDef.function[
            create_wrapper_function[case_raise_empty_error](),
            "case_raise_empty_error",
        ](),
        PyMethodDef.function[
            create_wrapper_function[case_raise_string_error](),
            "case_raise_string_error",
        ](),
        PyMethodDef.function[
            create_wrapper_function[case_mojo_raise](),
            "case_mojo_raise",
        ](),
        PyMethodDef.function[
            create_wrapper_function[incr_int__wrapper](),
            "incr_int",
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
        return String.format_sequence(
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
            create_wrapper_function[Person.obj_name](),
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


fn incr_int__wrapper(
    py_self: PythonObject,
    py_args: TypedPythonObject["Tuple"],
) raises -> PythonObject:
    check_arguments_arity("incr_int", 1, py_args)

    var arg_0_obj = py_args[0]

    var arg_0: UnsafePointer[Int] = check_argument_type[Int](
        "incr_int",
        "Int",
        arg_0_obj,
    )

    # Note: Pass an `inout` reference to the wrapped function
    incr_int(arg_0[])

    return PythonObject(None)
