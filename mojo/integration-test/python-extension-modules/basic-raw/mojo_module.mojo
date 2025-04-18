# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

from os import abort

import builtin
from python import Python, PythonObject, TypedPythonObject, PythonModule
from python._cpython import PyMethodDef, PyObjectPtr


@export
fn PyInit_mojo_module() -> PythonObject:
    # ----------------------------------
    # Create a Python module
    # ----------------------------------

    # This will initialize the Python interpreter and create
    # an extension module with the provided name.
    var module: PythonModule

    try:
        module = PythonModule("bindings")
    except:
        return abort[PythonObject]("failed to create Python module")

    # ----------------------------------
    # Populate the Python module
    # ----------------------------------

    # Create a function for the `mojo_count_args` below with the right bound args
    # set fn ptr + name and attach to the module above
    var funcs = List[PyMethodDef](
        PyMethodDef.function[
            mojo_count_args,
            "mojo_count_args",
            docstring="Count the provided arguments",
        ](),
    )

    try:
        Python.add_functions(module, funcs)
    except e:
        abort("Error adding functions to PyModule: ", e)

    # end up with a PythonModule with list of functions set on the module
    # (name,args,calling conv,etc.)

    return module


@export
fn mojo_count_args(py_self: PyObjectPtr, args: PyObjectPtr) -> PyObjectPtr:
    var cpython = Python().impl.cpython()

    return PythonObject(cpython.PyObject_Length(args)).py_object
