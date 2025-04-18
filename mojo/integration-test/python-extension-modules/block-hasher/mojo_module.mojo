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
from builtin._pybind import check_and_get_or_convert_arg
from python._cpython import PyMethodDef, PyObjectPtr
from sys import sizeof
import time


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

    # Create a function for the `mojo_block_hasher_with_list_output` below with the right bound args
    # set fn ptr + name and attach to the module above
    var funcs = List[PyMethodDef](
        PyMethodDef.function[
            py_c_function_wrapper[mojo_block_hasher_return_list],
            "mojo_block_hasher_return_list",
            docstring="Computes block hashes for a numpy array containing tokens",
        ](),
        PyMethodDef.function[
            py_c_function_wrapper[mojo_block_hasher_inplace],
            "mojo_block_hasher_inplace",
            docstring="Computes block hashes for a numpy array containing tokens",
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

    fn num_elts(self) -> Int:
        var num_elts = 1
        for i in range(self.nd):
            num_elts *= self.dimensions[i]
        return num_elts


fn _mojo_block_hasher[
    dtype: DType, //,
](
    py_array_object_ptr: UnsafePointer[PyArrayObject[dtype]],
    block_size: Int,
    enable_dbg_prints: Bool,
) -> List[Scalar[DType.uint64]]:
    # Compute number of hashes
    var num_elts: Int = py_array_object_ptr[].num_elts()
    var num_hashes: Int = num_elts // block_size
    if enable_dbg_prints:
        print("num_elts:", num_elts)
        print("num_hashes:", num_hashes)

    # Perfoming hashing
    var results: List[Scalar[DType.uint64]] = List[Scalar[DType.uint64]](
        capacity=num_hashes
    )
    var num_bytes = block_size * sizeof[dtype]()
    var hash_ptr_base = py_array_object_ptr[].data
    for block_idx in range(num_hashes):
        var hash_ptr_ints = hash_ptr_base.offset(block_idx * block_size)
        var hash_ptr_bytes = hash_ptr_ints.bitcast[Byte]()
        var hash_val = hash(hash_ptr_bytes, num_bytes)
        results.append(Scalar[DType.uint64](hash_val))

        if enable_dbg_prints:
            print("  hash([", end="")
            for i in range(min(block_size, 10)):
                print(hash_ptr_ints[i], end=" ")
            print("...]) => ", hash_val)

    return results


@export
fn mojo_block_hasher_return_list(
    py_self: PythonObject, py_args: TypedPythonObject["Tuple"]
) raises -> PythonObject:
    # Parse np array tokens input
    # ~660ns
    var arg0: PyObjectPtr = py_args[0].unsafe_as_py_object_ptr()
    var py_array_object_ptr: UnsafePointer[
        PyArrayObject[DType.int32]
    ] = arg0.unchecked_cast_to_mojo_value[PyArrayObject[DType.int32]]()

    # Parse block size
    # ~13us
    var block_size = Int.try_from_python(py_args[1])

    # Parse enable_dbg_prints
    # note: struct 'Bool' does not implement all requirements for 'ConvertibleFromPython'
    # ~13us
    var enable_dbg_prints_int = Int.try_from_python(py_args[2])
    var enable_dbg_prints: Bool = enable_dbg_prints_int != 0

    # Perfoming hashing
    # ~120ns
    var results = _mojo_block_hasher(
        py_array_object_ptr, block_size, enable_dbg_prints
    )

    # ~7.5us
    return PythonObject.list(results)


@export
fn mojo_block_hasher_inplace(
    py_self: PythonObject, py_args: TypedPythonObject["Tuple"]
) raises -> PythonObject:
    # Parse np array tokens input
    # ~660ns
    var arg0: PyObjectPtr = py_args[0].unsafe_as_py_object_ptr()
    var np_array_tokens: UnsafePointer[
        PyArrayObject[DType.int32]
    ] = arg0.unchecked_cast_to_mojo_value[PyArrayObject[DType.int32]]()

    # Parse np array hashes output
    # ~660ns
    var arg1: PyObjectPtr = py_args[1].unsafe_as_py_object_ptr()
    var np_array_hashes: UnsafePointer[
        PyArrayObject[DType.uint64]
    ] = arg1.unchecked_cast_to_mojo_value[PyArrayObject[DType.uint64]]()

    # Parse block size
    # ~13us
    var block_size = Int.try_from_python(py_args[2])

    # Parse enable_dbg_prints
    # note: struct 'Bool' does not implement all requirements for 'ConvertibleFromPython'
    # ~13us
    var enable_dbg_prints_int = Int.try_from_python(py_args[3])
    var enable_dbg_prints: Bool = enable_dbg_prints_int != 0

    # Perform hashing
    # ~120ns
    var results = _mojo_block_hasher(
        np_array_tokens, block_size, enable_dbg_prints
    )

    # Copy results to np_array_hashes
    # ~2us
    for i in range(len(results)):
        np_array_hashes[].data[i] = Scalar[DType.uint64](results[i])

    # ~400ns
    return PythonObject(None)
