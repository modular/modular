# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

from os import abort

from python import Python, PythonObject, PythonModule
from python._cpython import PyObjectPtr


@export
fn PyInit_mojo_module() -> PythonObject:
    """Create a Python module with a function binding for `mojo_count_args`."""

    try:
        return PythonModule("mojo_module").def_py_c_function[
            mojo_count_args,
            "mojo_count_args",
            docstring="Count the provided arguments",
        ]()
    except e:
        return abort[PythonObject]("failed to create Python module: ", e)


@export
fn mojo_count_args(py_self: PyObjectPtr, args: PyObjectPtr) -> PyObjectPtr:
    var cpython = Python().impl.cpython()

    return PythonObject(cpython.PyObject_Length(args)).py_object
