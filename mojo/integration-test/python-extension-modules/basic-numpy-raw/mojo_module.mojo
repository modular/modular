# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

from os import abort

import builtin
from memory import UnsafePointer
from python import Python, PythonObject, TypedPythonObject
from python._bindings import (
    check_argument_type,
    check_arguments_arity,
    py_c_function_wrapper,
    python_type_object,
)
from python._cpython import PyMethodDef, PyObjectPtr


@export
fn PyInit_mojo_module() -> PythonObject:
    # ----------------------------------
    # Create a Python module
    # ----------------------------------

    # This will initialize the Python interpreter and create
    # an extension module with the provided name.
    var module: TypedPythonObject["Module"]

    try:
        module = TypedPythonObject["Module"]("bindings")
    except:
        return abort[PythonObject]("failed to create Python module")

    # ----------------------------------
    # Populate the Python module
    # ----------------------------------

    # Create a function for the `mojo_incr_np_array` below with the right bound args
    # set fn ptr + name and attach to the module above
    var funcs = List[PyMethodDef](
        PyMethodDef.function[
            py_c_function_wrapper[mojo_incr_np_array],
            "mojo_incr_np_array",
            docstring="Increment the contents of a numpy array by one",
        ](),
    )

    try:
        Python.add_functions(module, funcs)
    except e:
        abort("Error adding functions to PyModule: ", e)

    # end up with a PythonModule with list of functions set on the module
    # (name,args,calling conv,etc.)

    return module


@value
struct PyArrayObject[dtype: DType]:
    """
    Container for a numpy array.

    See: https://numpy.org/doc/2.1/reference/c-api/types-and-structures.html#c.PyArrayObject
    """

    var data: UnsafePointer[Scalar[dtype]]
    var nd: Int
    var dimensions: UnsafePointer[Int]
    var strides: UnsafePointer[Int]
    var base: PyObjectPtr
    var descr: PyObjectPtr
    var flags: Int
    var weakreflist: PyObjectPtr

    # version dependent private members are omitted
    # ...


@export
fn mojo_incr_np_array(
    py_self: PythonObject, py_args: TypedPythonObject["Tuple"]
) raises -> PythonObject:
    alias dtype = DType.int32

    print("Hello from mojo_incr_np_array")
    var arg0: PyObjectPtr = py_args[0].unsafe_as_py_object_ptr()

    var py_array_object_ptr: UnsafePointer[
        PyArrayObject[dtype]
    ] = arg0.unchecked_cast_to_mojo_value[PyArrayObject[dtype]]()

    var nd = py_array_object_ptr[].nd
    var data_ptr = py_array_object_ptr[].data

    # Print each field of the struct
    print()
    print("Numpy Array Struct:")
    print("  data:", String(data_ptr))
    print("  nd:", nd)
    print("  dimensions:", end=" ")
    for i in range(nd):
        print(py_array_object_ptr[].dimensions[i], end=" ")
    print()
    print("  strides:", end=" ")
    for i in range(nd):
        print(py_array_object_ptr[].strides[i], end=" ")
    print()
    print("  descr:", String(py_array_object_ptr[].descr.unsized_obj_ptr))
    print("  flags:", hex(py_array_object_ptr[].flags))
    print(
        "  weakreflist:",
        String(py_array_object_ptr[].weakreflist.unsized_obj_ptr),
    )
    print()

    var num_elts = 1
    for i in range(nd):
        dim = py_array_object_ptr[].dimensions[i]
        num_elts *= dim

    for i in range(num_elts):
        data_ptr[i] += 1

    print("Goodbye from mojo_incr_np_array")
    return PythonObject(None)
