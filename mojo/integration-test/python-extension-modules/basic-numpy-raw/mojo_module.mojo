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
from python import PythonObject, TypedPythonObject, PythonModule
from python._cpython import PyObjectPtr


@export
fn PyInit_mojo_module() -> PythonObject:
    """Create a Python module with a function binding for `mojo_incr_np_array`.
    """

    try:
        return PythonModule("mojo_module").def_py_function[mojo_incr_np_array](
            "mojo_incr_np_array",
            docstring="Increment the contents of a numpy array by one",
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
