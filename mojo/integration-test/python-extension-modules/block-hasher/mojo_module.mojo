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

from memory import UnsafePointer
from python import (
    Python,
    PythonObject,
    TypedPythonObject,
    PythonModule,
)
from python._cpython import PyObjectPtr
from sys import sizeof


@export
fn PyInit_mojo_module() -> PythonObject:
    """Create a Python module with function bindings for
    `mojo_block_hasher_return_list` and `mojo_block_hasher_inplace`.
    """
    try:
        return (
            PythonModule("mojo_module")
            .def_py_function[mojo_block_hasher_return_list](
                "mojo_block_hasher_return_list",
                docstring=(
                    "Computes block hashes for a numpy array containing tokens"
                ),
            )
            .def_py_function[mojo_block_hasher_inplace](
                "mojo_block_hasher_inplace",
                docstring=(
                    "Computes block hashes for a numpy array containing tokens"
                ),
            )
        )
    except e:
        return abort[PythonObject]("failed to create Python module: ", e)


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
) -> PythonObject:
    # Compute number of hashes
    var num_elts: Int = py_array_object_ptr[].num_elts()
    var num_hashes: Int = num_elts // block_size
    if enable_dbg_prints:
        print("num_elts:", num_elts)
        print("num_hashes:", num_hashes)

    var cpython = Python().cpython()

    # Create a list of NULL elements with the size needed to store the hash
    # results.
    var result_py_list = cpython.PyList_New(num_hashes)

    # Perfoming hashing
    var num_bytes = block_size * sizeof[dtype]()
    var hash_ptr_base = py_array_object_ptr[].data
    for block_idx in range(num_hashes):
        var hash_ptr_ints = hash_ptr_base.offset(block_idx * block_size)
        var hash_ptr_bytes = hash_ptr_ints.bitcast[Byte]()
        var hash_val = hash(hash_ptr_bytes, num_bytes)

        # Convert the hash result to a Python object and store it in our
        # uninitialized list.
        var hash_val_obj = cpython.PyLong_FromSsize_t(hash_val)
        _ = cpython.PyList_SetItem(result_py_list, block_idx, hash_val_obj)

        if enable_dbg_prints:
            print("  hash([", end="")
            for i in range(min(block_size, 10)):
                print(hash_ptr_ints[i], end=" ")
            print("...]) => ", hash_val)

    return result_py_list


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
    return results^


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
    var num_elts: Int = np_array_tokens[].num_elts()
    var num_hashes: Int = num_elts // block_size

    # Perfoming hashing and store result in np_array_hashes
    var num_bytes = block_size * sizeof[DType.int32]()
    var hash_ptr_base = np_array_tokens[].data
    for block_idx in range(num_hashes):
        var hash_ptr_ints = hash_ptr_base.offset(block_idx * block_size)
        var hash_ptr_bytes = hash_ptr_ints.bitcast[Byte]()
        var hash_val = hash(hash_ptr_bytes, num_bytes)

        np_array_hashes[].data[block_idx] = UInt64(hash_val)

    # ~400ns
    return PythonObject(None)
