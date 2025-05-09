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
# RUN: %mojo %s | FileCheck %s

from collections import List

from max.tensor import TensorShape, TensorSpec


# CHECK: test_tensor_shape
fn test_tensor_shape():
    print("== test_tensor_shape")

    # CHECK: 1
    print(TensorShape(1))

    # CHECK: 1
    print(TensorShape((1,)))

    # CHECK: 1x2
    print(TensorShape(1, 2))

    # CHECK: 1x2x3
    print(TensorShape(1, 2, 3))

    # CHECK: 1x2x3x4
    print(TensorShape(1, 2, 3, 4))

    # CHECK: 1x2x3x4x5x6
    print(TensorShape(1, 2, 3, 4, 5, 6))

    # CHECK: 1x2x3x4x5x6x7
    print(TensorShape(1, 2, 3, 4, 5, 6, 7))

    # CHECK: 1x2x3x4x5x6x7x8x9x10
    print(TensorShape(1, 2, 3, 4, 5, 6, 7, 8, 9, 10))

    # CHECK: 1048576x2147483648x3298534883328
    print(TensorShape(1048576, 2147483648, 3298534883328))

    # CHECK: True
    print(TensorShape(1, 2, 3, 4) == TensorShape(1, 2, 3, 4))

    # CHECK: False
    print(TensorShape(1, 2, 3, 4) == TensorShape(1, 4, 5, 1))

    var shape_vec = List[Int]()
    for i in range(1, 5):
        shape_vec.append(i)

    # CHECK: True
    print(TensorShape(shapes=shape_vec) == TensorShape(1, 2, 3, 4))


# CHECK: test_tensor_spec
fn test_tensor_spec():
    print("== test_tensor_spec")

    # CHECK: 1x2x3xfloat32
    print(String(TensorSpec(DType.float32, 1, 2, 3)))

    # CHECK: 1x2x3x4x5x6xfloat32
    print(String(TensorSpec(DType.float32, 1, 2, 3, 4, 5, 6)))

    # CHECK: True
    print(
        TensorSpec(DType.float32, 1, 2, 3, 4)
        == TensorSpec(DType.float32, 1, 2, 3, 4)
    )

    # CHECK: False
    print(
        TensorSpec(DType.int32, 1, 2, 3, 4)
        == TensorSpec(DType.float32, 1, 2, 3, 4)
    )

    # CHECK: False
    print(
        TensorSpec(DType.float32, 1, 2, 3, 4)
        == TensorSpec(DType.float32, 1, 4, 5, 1)
    )

    var shape_vec = List[Int]()
    for i in range(1, 5):
        shape_vec.append(i)

    # CHECK: 1x2x3x4xfloat32
    print(String(TensorSpec(DType.float32, shape_vec)))

    # Check that dynamic dims work.
    # CHECK: True
    var shape = TensorShape(-9223372036854775808)
    var spec = TensorSpec(DType.float32, shape)
    print(shape == spec.shape)


fn main():
    test_tensor_shape()
    test_tensor_spec()
