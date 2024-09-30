# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

import builtin

from sys import exit
from memory import UnsafePointer

from os import abort

from python import Python, PythonObject, TypedPythonObject
from python._cpython import PyMethodDef, PyObjectPtr, create_wrapper_function


@export
fn PyInit_feature_overview() -> PythonObject:
    # Initialize the global runtime (including the memory allocator)
    _ = builtin._startup._init_global_runtime(UnsafePointer[NoneType]())

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
