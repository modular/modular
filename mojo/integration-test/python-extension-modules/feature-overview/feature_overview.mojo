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
from python._bindings import create_wrapper_function, PyMojoObject
from python._cpython import (
    PyMethodDef,
    PyObject,
    PyObjectPtr,
)


@export
fn PyInit_feature_overview() -> PythonObject:
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
    )

    try:
        Python.add_functions(module, funcs)
    except e:
        abort("Error adding functions to PyModule: " + str(e))

    add_person_type(module)

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

    @staticmethod
    fn obj_init(self: UnsafePointer[Self]):
        # Field ptrs
        # TODO(MSTDL-950): Avoid get field ptrs through uninit reference.
        var name_ptr = UnsafePointer.address_of(self[].name)
        var age_ptr = UnsafePointer.address_of(self[].age)

        name_ptr.init_pointee_move("John Smith")
        age_ptr.init_pointee_move(123)

    @staticmethod
    fn obj_destroy(self: UnsafePointer[Self]):
        # TODO(MSTDL-633):
        #   Is this always safe? Wrap in GIL, because this could
        #   evaluate arbitrary code?
        # Destroy this `Person` instance.
        self.destroy_pointee()

    @staticmethod
    fn obj_name(
        self_: PythonObject, args: TypedPythonObject["Tuple"]
    ) -> PythonObject:
        var self0 = PyMojoObject[Person].unsafe_cast_obj(
            self_.unsafe_as_py_object_ptr()
        )

        return PythonObject(self0[].name).steal_data()


fn add_person_type(inout module: TypedPythonObject["Module"]):
    var cpython = Python().impl.cpython()

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
        var type_obj = PyMojoObject[Person].python_type_object[
            "Person",
            Person.obj_init,
            Person.obj_destroy,
        ](methods)

        Python.add_object(module, "Person", type_obj)
    except e:
        abort("error adding object: " + str(e))
