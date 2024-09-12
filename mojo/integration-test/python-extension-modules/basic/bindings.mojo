# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

import builtin

from sys import exit
from sys.ffi import C_char

from os import abort

from python import Python, PythonObject, TypedPythonObject
from python._cpython import PyMethodDef, PyObjectPtr


@export
fn PyInit_bindings() -> PythonObject:
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
    except:
        return abort[PythonObject]("failed to create Python module")

    # ----------------------------------
    # Populate the Python module
    # ----------------------------------

    alias METHOD_COUNT = 1

    alias METH_VARARGS = 0x1

    # FIXME:
    #   This is an intentional memory leak, because we don't store this
    #   in a global variable (yet).
    var methods = UnsafePointer[PyMethodDef].alloc(METHOD_COUNT + 1)

    # Create a function for the `mojo_count_args` below with the right bound args
    # set fn ptr + name and attach to the module above
    (methods + 0).init_pointee_move(
        PyMethodDef(
            "mojo_count_args".unsafe_cstr_ptr(),
            mojo_count_args,
            METH_VARARGS,
            "docs for mojo_count_args".unsafe_cstr_ptr(),
        )
    )

    # Write a zeroed entry at the end as a terminator.
    (methods + 1).init_pointee_move(PyMethodDef())

    var result = Python.add_methods(module, methods)

    if result != 0:
        print("ERROR: Error adding methods to PyModule:", result)
        exit(-1)

    # end up with a PythonModule with list of functions set on the module
    # (name,args,calling conv,etc.)

    return module


@export
fn mojo_count_args(py_self: PyObjectPtr, args: PyObjectPtr) -> PyObjectPtr:
    var cpython = Python().impl.cpython()

    return PythonObject(cpython.PyObject_Length(args)).py_object
