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

from python import Python, PythonObject
from memory import memcpy, UnsafePointer
from collections import List

from max.tensor import Tensor, TensorShape


@always_inline
fn numpy_data_pointer[
    type: DType
](numpy_array: PythonObject) raises -> UnsafePointer[Scalar[type]]:
    return numpy_array.__array_interface__["data"][0].unsafe_get_as_pointer[
        type
    ]()


@always_inline
fn memcpy_to_numpy[
    type: DType
](array: PythonObject, tensor: Tensor[type]) raises:
    var dst = numpy_data_pointer[type](array)
    var src = tensor._ptr
    var length = tensor.num_elements()
    memcpy(dst.address, src.address, length)


@always_inline
fn memcpy_from_numpy[
    type: DType
](array: PythonObject, tensor: Tensor[type]) raises:
    var src = numpy_data_pointer[type](array)
    var dst = tensor._ptr
    var length = tensor.num_elements()
    memcpy(dst.address, src.address, length)


@always_inline
fn shape_to_python_list(shape: TensorShape) raises -> PythonObject:
    var python_list = Python.evaluate("list()")
    for i in range(shape.rank()):
        _ = python_list.append(shape[i])
    return python_list^


@always_inline
fn get_np_dtype[type: DType](np: PythonObject) raises -> PythonObject:
    @parameter
    if type is DType.float32:
        return np.float32
    elif type is DType.int32:
        return np.int32
    elif type is DType.int64:
        return np.int64
    elif type is DType.uint8:
        return np.uint8

    raise "Unknown datatype"


@always_inline
fn tensor_to_numpy[
    type: DType
](tensor: Tensor[type], np: PythonObject) raises -> PythonObject:
    var shape = shape_to_python_list(tensor.shape())
    var tensor_as_numpy = np.zeros(shape, get_np_dtype[type](np))
    memcpy_to_numpy(tensor_as_numpy, tensor)
    return tensor_as_numpy^


@always_inline
fn numpy_to_tensor[type: DType](array: PythonObject) raises -> Tensor[type]:
    var shape = List[Int]()
    var array_shape = array.shape
    for dim in array_shape:
        shape.append(dim)
    var out = Tensor[type](shape)
    memcpy_from_numpy(array, out)
    return out^
