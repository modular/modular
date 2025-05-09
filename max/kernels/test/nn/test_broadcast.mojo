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

from buffer import NDBuffer
from buffer.dimlist import DimList
from nn.broadcast import broadcast

from utils.index import IndexList


# CHECK-LABEL: test_broadcast_empty_shape
fn test_broadcast_empty_shape():
    print("== test_broadcast_empty_shape")

    # parameters
    alias input_shape = DimList(1)
    alias output_shape = DimList(0)

    # Create a 1D tensor of shape (1), of the form [1]
    var input_stack = InlineArray[
        Scalar[DType.index], Int(input_shape.product())
    ](uninitialized=True)
    var input = NDBuffer[DType.index, 1, _, input_shape](input_stack)
    input[0] = 1

    # Create a 1D tensor of shape (0)
    var output_stack = InlineArray[
        Scalar[DType.index], Int(output_shape.product())
    ](uninitialized=True)
    var output = NDBuffer[DType.index, 1, _, output_shape](output_stack)

    broadcast(output, input)
    # output tensor will have the form:
    # []

    # CHECK: 1
    print(input[0])

    # test shouldn't crash


# CHECK-LABEL: test_broadcast_same_shape
fn test_broadcast_same_shape():
    print("== test_broadcast_same_shape")

    # parameters
    alias input_shape = DimList(1, 2, 1)
    alias output_shape = DimList(1, 2, 1)

    # Create a 3D tensor of shape (1, 2, 1), of the form
    # [[[1], [2]]]
    var input_stack = InlineArray[
        Scalar[DType.index], Int(input_shape.product())
    ](uninitialized=True)
    var input = NDBuffer[DType.index, 3, _, input_shape](input_stack)
    input[IndexList[3](0, 0, 0)] = 1
    input[IndexList[3](0, 1, 0)] = 2

    # Create a 3D tensor of shape (1, 2, 1)
    var output_stack = InlineArray[
        Scalar[DType.index], Int(output_shape.product())
    ](uninitialized=True)
    var output = NDBuffer[DType.index, 3, _, output_shape](output_stack)
    output.fill(0)

    broadcast(output, input)
    # output tensor will have the form:
    # [[[1], [2]]]

    # CHECK: 1
    print(input[0, 0, 0])
    # CHECK: 2
    print(input[0, 1, 0])

    # CHECK: 1
    print(output[0, 0, 0])
    # CHECK: 2
    print(output[0, 1, 0])


# CHECK-LABEL: test_broadcast_single_axis
fn test_broadcast_single_axis():
    print("== test_broadcast_single_axis")

    # parameters
    alias input_shape = DimList(1, 2)
    alias output_shape = DimList(3, 2)

    # Create a 2D tensor of shape (1, 2), of the form
    # [[1, 2]]
    var input_stack = InlineArray[
        Scalar[DType.index], Int(input_shape.product())
    ](uninitialized=True)
    var input = NDBuffer[DType.index, 2, _, input_shape](input_stack)

    input[IndexList[2](0, 0)] = 1
    input[IndexList[2](0, 1)] = 2

    # Create a 2D tensor of shape (3, 2)
    var output_stack = InlineArray[
        Scalar[DType.index], Int(output_shape.product())
    ](uninitialized=True)
    var output = NDBuffer[DType.index, 2, _, output_shape](output_stack)
    output.fill(0)

    broadcast(output, input)
    # output tensor will have the form:
    # [[1, 2], [1, 2], [1, 2]]

    # CHECK: 1
    print(input[0, 0])
    # CHECK: 2
    print(input[0, 1])

    # CHECK: 1
    print(output[0, 0])
    # CHECK: 2
    print(output[0, 1])
    # CHECK: 1
    print(output[1, 0])
    # CHECK: 2
    print(output[1, 1])
    # CHECK: 1
    print(output[2, 0])
    # CHECK: 2
    print(output[2, 1])


# CHECK-LABEL: test_broadcast_multi_axes
fn test_broadcast_multi_axes():
    print("== test_broadcast_multi_axes")

    # parameters
    alias input_shape = DimList(1, 2, 1)
    alias output_shape = DimList(2, 2, 3)

    # Create a 3D tensor of shape (1, 2, 1), of the form
    # [[[1], [2]]]
    var input_stack = InlineArray[
        Scalar[DType.index], Int(input_shape.product())
    ](uninitialized=True)
    var input = NDBuffer[DType.index, 3, _, input_shape](input_stack)

    input[IndexList[3](0, 0, 0)] = 1
    input[IndexList[3](0, 1, 0)] = 2

    # Create a 3D tensor of shape (2, 2, 3)
    var output_stack = InlineArray[
        Scalar[DType.index], Int(output_shape.product())
    ](uninitialized=True)
    var output = NDBuffer[DType.index, 3, _, output_shape](output_stack)
    output.fill(0)

    broadcast(output, input)
    # output tensor will have the form:
    # [[[1, 1, 1], [2, 2, 2]],
    #  [[1, 1, 1], [2, 2, 2]]]

    # CHECK: 1
    print(input[0, 0, 0])
    # CHECK: 2
    print(input[0, 1, 0])

    # CHECK: 1
    print(output[0, 0, 0])
    # CHECK: 2
    print(output[0, 1, 0])
    # CHECK: 1
    print(output[0, 0, 1])
    # CHECK: 2
    print(output[0, 1, 1])
    # CHECK: 1
    print(output[0, 0, 2])
    # CHECK: 2
    print(output[0, 1, 2])
    # CHECK: 1
    print(output[1, 0, 0])
    # CHECK: 2
    print(output[1, 1, 0])
    # CHECK: 1
    print(output[1, 0, 1])
    # CHECK: 2
    print(output[1, 1, 1])
    # CHECK: 1
    print(output[1, 0, 2])
    # CHECK: 2
    print(output[1, 1, 2])


fn test_broadcast_multi_axes_nested():
    # parameters
    alias input_shape = DimList(2, 1, 2, 1, 2)
    alias output_shape = DimList(2, 2, 2, 2, 2)

    # Create a 5D tensor of shape (2, 1, 2, 1, 2), of the form
    # [[[[[1, 2]], [[3, 4]]]], [[[[5, 6]], [[7, 8]]]]]
    var input_stack = InlineArray[
        Scalar[DType.index], Int(input_shape.product())
    ](uninitialized=True)
    var input = NDBuffer[DType.index, 5, _, input_shape](input_stack)

    input[IndexList[5](0, 0, 0, 0, 0)] = 1
    input[IndexList[5](0, 0, 0, 0, 1)] = 2
    input[IndexList[5](0, 0, 1, 0, 0)] = 3
    input[IndexList[5](0, 0, 1, 0, 1)] = 4
    input[IndexList[5](1, 0, 0, 0, 0)] = 5
    input[IndexList[5](1, 0, 0, 0, 1)] = 6
    input[IndexList[5](1, 0, 1, 0, 0)] = 7
    input[IndexList[5](1, 0, 1, 0, 1)] = 8

    # Create a 5D tensor of shape (2, 2, 2, 2, 2)
    var output_stack = InlineArray[
        Scalar[DType.index], Int(output_shape.product())
    ](uninitialized=True)
    var output = NDBuffer[DType.index, 5, _, output_shape](output_stack)
    output.fill(0)

    broadcast(output, input)

    # CHECK: 1
    print(output[0, 0, 0, 0, 0])
    # CHECK: 2
    print(output[0, 0, 0, 0, 1])
    # CHECK: 1
    print(output[0, 0, 0, 1, 0])
    # CHECK: 2
    print(output[0, 0, 0, 1, 1])
    # CHECK: 3
    print(output[0, 0, 1, 0, 0])
    # CHECK: 4
    print(output[0, 0, 1, 0, 1])
    # CHECK: 3
    print(output[0, 0, 1, 1, 0])
    # CHECK: 4
    print(output[0, 0, 1, 1, 1])

    # CHECK: 1
    print(output[0, 1, 0, 0, 0])
    # CHECK: 2
    print(output[0, 1, 0, 0, 1])
    # CHECK: 1
    print(output[0, 1, 0, 1, 0])
    # CHECK: 2
    print(output[0, 1, 0, 1, 1])
    # CHECK: 3
    print(output[0, 1, 1, 0, 0])
    # CHECK: 4
    print(output[0, 1, 1, 0, 1])
    # CHECK: 3
    print(output[0, 1, 1, 1, 0])
    # CHECK: 4
    print(output[0, 1, 1, 1, 1])

    # CHECK: 5
    print(output[1, 0, 0, 0, 0])
    # CHECK: 6
    print(output[1, 0, 0, 0, 1])
    # CHECK: 5
    print(output[1, 0, 0, 1, 0])
    # CHECK: 6
    print(output[1, 0, 0, 1, 1])
    # CHECK: 7
    print(output[1, 0, 1, 0, 0])
    # CHECK: 8
    print(output[1, 0, 1, 0, 1])
    # CHECK: 7
    print(output[1, 0, 1, 1, 0])
    # CHECK: 8
    print(output[1, 0, 1, 1, 1])

    # CHECK: 5
    print(output[1, 1, 0, 0, 0])
    # CHECK: 6
    print(output[1, 1, 0, 0, 1])
    # CHECK: 5
    print(output[1, 1, 0, 1, 0])
    # CHECK: 6
    print(output[1, 1, 0, 1, 1])
    # CHECK: 7
    print(output[1, 1, 1, 0, 0])
    # CHECK: 8
    print(output[1, 1, 1, 0, 1])
    # CHECK: 7
    print(output[1, 1, 1, 1, 0])
    # CHECK: 8
    print(output[1, 1, 1, 1, 1])


fn main():
    test_broadcast_empty_shape()
    test_broadcast_same_shape()
    test_broadcast_single_axis()
    test_broadcast_multi_axes()
    test_broadcast_multi_axes_nested()
