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
from memory import UnsafePointer
from nn.pad import pad_constant, pad_reflect, pad_repeat

from utils.index import IndexList


# CHECK-LABEL: test_pad_1d
fn test_pad_1d():
    print("== test_pad_1d")

    alias in_shape = DimList(3)
    alias out_shape = DimList(6)

    # Create an input matrix of the form
    # [1, 2, 3]
    var input_stack = InlineArray[Scalar[DType.index], Int(in_shape.product())](
        1, 2, 3
    )
    var input = NDBuffer[DType.index, 1, _, in_shape](input_stack)

    # Create a padding array of the form
    # [1, 2]
    var paddings_stack = InlineArray[Scalar[DType.index], 2](1, 2)
    var paddings = NDBuffer[DType.index, 1, _, 2](paddings_stack)

    # Create an output matrix of the form
    # [0, 0, 0, 0, 0, 0]
    var output_stack = InlineArray[
        Scalar[DType.index], Int(out_shape.product())
    ](uninitialized=True)
    var output = NDBuffer[DType.index, 1, _, out_shape](output_stack)
    output.fill(0)

    var constant = Scalar[DType.index](5)

    # pad
    pad_constant(output, input, paddings.data, constant)

    # output should have form
    # [5, 1, 2, 3, 5, 5]

    # CHECK: 5
    print(output[0])
    # CHECK: 1
    print(output[1])
    # CHECK: 2
    print(output[2])
    # CHECK: 3
    print(output[3])
    # CHECK: 5
    print(output[4])
    # CHECK: 5
    print(output[5])


# CHECK-LABEL: test_pad_reflect_1d
fn test_pad_reflect_1d():
    print("== test_pad_reflect_1d")

    alias in_shape = DimList(3)
    alias out_shape = DimList(8)

    # Create an input matrix of the form
    # [1, 2, 3]
    var input_stack = InlineArray[Scalar[DType.index], Int(in_shape.product())](
        1, 2, 3
    )
    var input = NDBuffer[DType.index, 1, _, in_shape](input_stack)

    # Create an output matrix of the form
    # [0, 0, 0, 0, 0, 0, 0, 0]
    var output_stack = InlineArray[
        Scalar[DType.index], Int(out_shape.product())
    ](uninitialized=True)
    var output = NDBuffer[DType.index, 1, _, out_shape](output_stack)
    output.fill(0)

    # Create a padding array of the form
    # [3, 2]
    var paddings_stack = InlineArray[Scalar[DType.index], 2](3, 2)
    var paddings = NDBuffer[DType.index, 1, _, 2](paddings_stack)

    # pad
    pad_reflect(output, input, paddings.data)

    # output should have form
    # [2, 3, 2, 1, 2, 3, 2, 1]

    # CHECK: 2
    print(output[0])
    # CHECK: 3
    print(output[1])
    # CHECK: 2
    print(output[2])
    # CHECK: 1
    print(output[3])
    # CHECK: 2
    print(output[4])
    # CHECK: 3
    print(output[5])
    # CHECK: 2
    print(output[6])
    # CHECK: 1
    print(output[7])


# CHECK-LABEL: test_pad_repeat_1d
fn test_pad_repeat_1d():
    print("== test_pad_repeat_1d")

    alias in_shape = DimList(3)
    alias out_shape = DimList(8)

    # Create an input matrix of the form
    # [1, 2, 3]
    var input_stack = InlineArray[Scalar[DType.index], Int(in_shape.product())](
        1, 2, 3
    )
    var input = NDBuffer[DType.index, 1, _, in_shape](input_stack)

    # Create an output matrix of the form
    # [0, 0, 0, 0, 0, 0, 0, 0]
    var output_stack = InlineArray[
        Scalar[DType.index], Int(out_shape.product())
    ](uninitialized=True)
    var output = NDBuffer[DType.index, 1, _, out_shape](output_stack)
    output.fill(0)

    # Create a padding array of the form
    # [3, 2]
    var paddings_stack = InlineArray[Scalar[DType.index], 2](3, 2)
    var paddings = NDBuffer[DType.index, 1, _, 2](paddings_stack)

    # pad
    pad_repeat(output, input, paddings.data)

    # output should have form
    # [1, 1, 1, 1, 2, 3, 3, 3]

    # CHECK: 1
    print(output[0])
    # CHECK: 1
    print(output[1])
    # CHECK: 1
    print(output[2])
    # CHECK: 1
    print(output[3])
    # CHECK: 2
    print(output[4])
    # CHECK: 3
    print(output[5])
    # CHECK: 3
    print(output[6])
    # CHECK: 3
    print(output[7])


# CHECK-LABEL: test_pad_2d
fn test_pad_2d():
    print("== test_pad_2d")

    alias in_shape = DimList(2, 2)
    alias out_shape = DimList(3, 4)

    # Create an input matrix of the form
    # [[1, 2],
    #  [3, 4]]
    var input_stack = InlineArray[Scalar[DType.index], Int(in_shape.product())](
        uninitialized=True
    )
    var input = NDBuffer[DType.index, 2, _, in_shape](input_stack)
    input[IndexList[2](0, 0)] = 1
    input[IndexList[2](0, 1)] = 2
    input[IndexList[2](1, 0)] = 3
    input[IndexList[2](1, 1)] = 4

    # Create a padding array of the form
    # [1, 0, 1, 1]
    var paddings_stack = InlineArray[Scalar[DType.index], 4](1, 0, 1, 1)
    var paddings = NDBuffer[DType.index, 1, _, 4](paddings_stack)

    # Create an output matrix of the form
    # [[0, 0, 0, 0]
    #  [0, 0, 0, 0]
    #  [0, 0, 0, 0]]
    var output_stack = InlineArray[
        Scalar[DType.index], Int(out_shape.product())
    ](uninitialized=True)
    var output = NDBuffer[DType.index, 2, _, out_shape](output_stack)
    output.fill(0)

    var constant = Scalar[DType.index](6)

    # pad
    pad_constant(output, input, paddings.data, constant)

    # output should have form
    # [[6, 6, 6, 6]
    #  [6, 1, 2, 6]
    #  [6, 3, 4, 6]]

    # CHECK: 6
    print(output[0, 0])
    # CHECK: 6
    print(output[0, 1])
    # CHECK: 6
    print(output[0, 2])
    # CHECK: 6
    print(output[0, 3])
    # CHECK: 6
    print(output[1, 0])
    # CHECK: 1
    print(output[1, 1])
    # CHECK: 2
    print(output[1, 2])
    # CHECK: 6
    print(output[1, 3])
    # CHECK: 6
    print(output[2, 0])
    # CHECK: 3
    print(output[2, 1])
    # CHECK: 4
    print(output[2, 2])
    # CHECK: 6
    print(output[2, 3])


# CHECK-LABEL: test_pad_reflect_2d
fn test_pad_reflect_2d():
    print("== test_pad_reflect_2d")

    alias in_shape = DimList(2, 2)
    alias out_shape = DimList(6, 3)

    # Create an input matrix of the form
    # [[1, 2],
    #  [3, 4]]
    var input_stack = InlineArray[Scalar[DType.index], Int(in_shape.product())](
        uninitialized=True
    )
    var input = NDBuffer[DType.index, 2, _, in_shape](input_stack)
    input[IndexList[2](0, 0)] = 1
    input[IndexList[2](0, 1)] = 2
    input[IndexList[2](1, 0)] = 3
    input[IndexList[2](1, 1)] = 4

    # Create a padding array of the form
    # [2, 2, 1, 0]
    var paddings_stack = InlineArray[Scalar[DType.index], 4](2, 2, 1, 0)
    var paddings = NDBuffer[DType.index, 1, _, 4](paddings_stack)

    # Create an output matrix of the form
    # [[0 0 0]
    #  [0 0 0]
    #  [0 0 0]
    #  [0 0 0]
    #  [0 0 0]
    #  [0 0 0]]
    var output_stack = InlineArray[
        Scalar[DType.index], Int(out_shape.product())
    ](uninitialized=True)
    var output = NDBuffer[DType.index, 2, _, out_shape](output_stack)
    output.fill(0)

    # pad
    pad_reflect(output, input, paddings.data)

    # output should have form
    # [[2 1 2]
    #  [4 3 4]
    #  [2 1 2]
    #  [4 3 4]
    #  [2 1 2]
    #  [4 3 4]]

    # CHECK: 2
    print(output[0, 0])
    # CHECK: 1
    print(output[0, 1])
    # CHECK: 2
    print(output[0, 2])
    # CHECK: 4
    print(output[1, 0])
    # CHECK: 3
    print(output[1, 1])
    # CHECK: 4
    print(output[1, 2])
    # CHECK: 2
    print(output[2, 0])
    # CHECK: 1
    print(output[2, 1])
    # CHECK: 2
    print(output[2, 2])
    # CHECK: 4
    print(output[3, 0])
    # CHECK: 3
    print(output[3, 1])
    # CHECK: 4
    print(output[3, 2])
    # CHECK: 2
    print(output[4, 0])
    # CHECK: 1
    print(output[4, 1])
    # CHECK: 2
    print(output[4, 2])
    # CHECK: 4
    print(output[5, 0])
    # CHECK: 3
    print(output[5, 1])
    # CHECK: 4
    print(output[5, 2])


# CHECK-LABEL: test_pad_repeat_2d
fn test_pad_repeat_2d():
    print("== test_pad_repeat_2d")

    alias in_shape = DimList(2, 2)
    alias out_shape = DimList(6, 3)

    # Create an input matrix of the form
    # [[1, 2],
    #  [3, 4]]
    var input_stack = InlineArray[Scalar[DType.index], Int(in_shape.product())](
        uninitialized=True
    )
    var input = NDBuffer[DType.index, 2, _, in_shape](input_stack)
    input[IndexList[2](0, 0)] = 1
    input[IndexList[2](0, 1)] = 2
    input[IndexList[2](1, 0)] = 3
    input[IndexList[2](1, 1)] = 4

    # Create a padding array of the form
    # [2, 2, 1, 0]
    var paddings_stack = InlineArray[Scalar[DType.index], 4](2, 2, 1, 0)
    var paddings = NDBuffer[DType.index, 1, _, 4](paddings_stack)

    # Create an output matrix of the form
    # [[0 0 0]
    #  [0 0 0]
    #  [0 0 0]
    #  [0 0 0]
    #  [0 0 0]
    #  [0 0 0]]
    var output_stack = InlineArray[
        Scalar[DType.index], Int(out_shape.product())
    ](uninitialized=True)
    var output = NDBuffer[DType.index, 2, _, out_shape](output_stack)
    output.fill(0)

    # pad
    pad_repeat(output, input, paddings.data)

    # output should have form
    # [[1, 1, 2],
    #  [1, 1, 2],
    #  [1, 1, 2],
    #  [3, 3, 4],
    #  [3, 3, 4],
    #  [3, 3, 4]]

    # CHECK: 1
    print(output[0, 0])
    # CHECK: 1
    print(output[0, 1])
    # CHECK: 2
    print(output[0, 2])
    # CHECK: 1
    print(output[1, 0])
    # CHECK: 1
    print(output[1, 1])
    # CHECK: 2
    print(output[1, 2])
    # CHECK: 1
    print(output[2, 0])
    # CHECK: 1
    print(output[2, 1])
    # CHECK: 2
    print(output[2, 2])
    # CHECK: 3
    print(output[3, 0])
    # CHECK: 3
    print(output[3, 1])
    # CHECK: 4
    print(output[3, 2])
    # CHECK: 3
    print(output[4, 0])
    # CHECK: 3
    print(output[4, 1])
    # CHECK: 4
    print(output[4, 2])
    # CHECK: 3
    print(output[5, 0])
    # CHECK: 3
    print(output[5, 1])
    # CHECK: 4
    print(output[5, 2])


# CHECK-LABEL: test_pad_3d
fn test_pad_3d():
    print("== test_pad_3d")

    alias in_shape = DimList(1, 2, 2)
    alias out_shape = DimList(2, 3, 3)

    # Create an input matrix of the form
    # [[[1, 2],
    #   [3, 4]]]
    var input_stack = InlineArray[Scalar[DType.index], Int(in_shape.product())](
        uninitialized=True
    )
    var input = NDBuffer[DType.index, 3, _, in_shape](input_stack)
    input[IndexList[3](0, 0, 0)] = 1
    input[IndexList[3](0, 0, 1)] = 2
    input[IndexList[3](0, 1, 0)] = 3
    input[IndexList[3](0, 1, 1)] = 4

    # Create a padding array of the form
    # [1, 0, 0, 1, 1, 0]
    var paddings_stack = InlineArray[Scalar[DType.index], 6](1, 0, 0, 1, 1, 0)
    var paddings = NDBuffer[DType.index, 1, _, 6](paddings_stack)

    # Create an output matrix of the form
    # [[[0, 0, 0]
    #   [0, 0, 0]
    #   [0, 0, 0]]
    #  [[0, 0, 0]
    #   [0, 0, 0]
    #   [0, 0, 0]]]
    var output_stack = InlineArray[
        Scalar[DType.index], Int(out_shape.product())
    ](uninitialized=True)
    var output = NDBuffer[DType.index, 3, _, out_shape](output_stack)
    output.fill(0)

    var constant = Scalar[DType.index](7)

    # pad
    pad_constant(output, input, paddings.data, constant)

    # output should have form
    # [[[7, 7, 7]
    #   [7, 7, 7]
    #   [7, 7, 7]]
    #  [[7, 1, 2]
    #   [7, 3, 4]
    #   [7, 7, 7]]]

    # CHECK: 7
    print(output[0, 0, 0])
    # CHECK: 7
    print(output[0, 0, 1])
    # CHECK: 7
    print(output[0, 0, 2])
    # CHECK: 7
    print(output[0, 1, 0])
    # CHECK: 7
    print(output[0, 1, 1])
    # CHECK: 7
    print(output[0, 1, 2])
    # CHECK: 7
    print(output[0, 2, 0])
    # CHECK: 7
    print(output[0, 2, 1])
    # CHECK: 7
    print(output[0, 2, 2])
    # CHECK: 7
    print(output[1, 0, 0])
    # CHECK: 1
    print(output[1, 0, 1])
    # CHECK: 2
    print(output[1, 0, 2])
    # CHECK: 7
    print(output[1, 1, 0])
    # CHECK: 3
    print(output[1, 1, 1])
    # CHECK: 4
    print(output[1, 1, 2])
    # CHECK: 7
    print(output[1, 2, 0])
    # CHECK: 7
    print(output[1, 2, 1])
    # CHECK: 7
    print(output[1, 2, 2])


# CHECK-LABEL: test_pad_reflect_3d
fn test_pad_reflect_3d():
    print("== test_pad_reflect_3d")
    alias in_shape = DimList(2, 2, 2)
    alias out_shape = DimList(4, 3, 3)

    # Create an input matrix of the form
    # [[[1, 2],
    #   [3, 4]],
    #  [[1, 2],
    #   [3 ,4]]]
    var input_stack = InlineArray[Scalar[DType.index], Int(in_shape.product())](
        uninitialized=True
    )

    var input = NDBuffer[DType.index, 3, _, in_shape](input_stack)
    input[IndexList[3](0, 0, 0)] = 1
    input[IndexList[3](0, 0, 1)] = 2
    input[IndexList[3](0, 1, 0)] = 3
    input[IndexList[3](0, 1, 1)] = 4
    input[IndexList[3](1, 0, 0)] = 1
    input[IndexList[3](1, 0, 1)] = 2
    input[IndexList[3](1, 1, 0)] = 3
    input[IndexList[3](1, 1, 1)] = 4

    # Create a padding array of the form
    # [1, 1, 0, 1, 1, 0]
    var paddings_stack = InlineArray[Scalar[DType.index], 6](1, 1, 0, 1, 1, 0)
    var paddings = NDBuffer[DType.index, 1, _, 6](paddings_stack)

    # Create an output matrix of the form
    # [[[0 0 0]
    #   [0 0 0]
    #   [0 0 0]]
    #  [[0 0 0]
    #   [0 0 0]
    #   [0 0 0]]
    #  [[0 0 0]
    #   [0 0 0]
    #   [0 0 0]]
    #  [[0 0 0]
    #   [0 0 0]
    #   [0 0 0]]]
    var output_stack = InlineArray[
        Scalar[DType.index], Int(out_shape.product())
    ](uninitialized=True)
    var output = NDBuffer[DType.index, 3, _, out_shape](output_stack)
    output.fill(0)

    # pad
    pad_reflect(output, input, paddings.data)

    # output should have form
    # [[[2 1 2]
    #   [4 3 4]
    #   [2 1 2]]
    #  [[2 1 2]
    #   [4 3 4]
    #   [2 1 2]]
    #  [[2 1 2]
    #   [4 3 4]
    #   [2 1 2]]
    #  [[2 1 2]
    #   [4 3 4]
    #   [2 1 2]]]

    # CHECK: 2
    print(output[0, 0, 0])
    # CHECK: 1
    print(output[0, 0, 1])
    # CHECK: 2
    print(output[0, 0, 2])
    # CHECK: 4
    print(output[0, 1, 0])
    # CHECK: 3
    print(output[0, 1, 1])
    # CHECK: 4
    print(output[0, 1, 2])
    # CHECK: 2
    print(output[0, 2, 0])
    # CHECK: 1
    print(output[0, 2, 1])
    # CHECK: 2
    print(output[0, 2, 2])
    # CHECK: 2
    print(output[1, 0, 0])
    # CHECK: 1
    print(output[1, 0, 1])
    # CHECK: 2
    print(output[1, 0, 2])
    # CHECK: 4
    print(output[1, 1, 0])
    # CHECK: 3
    print(output[1, 1, 1])
    # CHECK: 4
    print(output[1, 1, 2])
    # CHECK: 2
    print(output[1, 2, 0])
    # CHECK: 1
    print(output[1, 2, 1])
    # CHECK: 2
    print(output[1, 2, 2])
    # CHECK: 2
    print(output[2, 0, 0])
    # CHECK: 1
    print(output[2, 0, 1])
    # CHECK: 2
    print(output[2, 0, 2])
    # CHECK: 4
    print(output[2, 1, 0])
    # CHECK: 3
    print(output[2, 1, 1])
    # CHECK: 4
    print(output[2, 1, 2])
    # CHECK: 2
    print(output[2, 2, 0])
    # CHECK: 1
    print(output[2, 2, 1])
    # CHECK: 2
    print(output[2, 2, 2])
    # CHECK: 2
    print(output[3, 0, 0])
    # CHECK: 1
    print(output[3, 0, 1])
    # CHECK: 2
    print(output[3, 0, 2])
    # CHECK: 4
    print(output[3, 1, 0])
    # CHECK: 3
    print(output[3, 1, 1])
    # CHECK: 4
    print(output[3, 1, 2])
    # CHECK: 2
    print(output[3, 2, 0])
    # CHECK: 1
    print(output[3, 2, 1])
    # CHECK: 2
    print(output[3, 2, 2])


# CHECK-LABEL: test_pad_reflect_3d_singleton
fn test_pad_reflect_3d_singleton():
    print("== test_pad_reflect_3d_singleton")
    alias in_shape = DimList(1, 1, 1)
    alias out_shape = DimList(2, 2, 5)

    # Create an input matrix of the form
    # [[[1]]]
    var input_stack = InlineArray[Scalar[DType.index], Int(in_shape.product())](
        uninitialized=True
    )
    var input = NDBuffer[DType.index, 3, _, in_shape](input_stack)
    input[IndexList[3](0, 0, 0)] = 1

    # Create a padding array of the form
    # [1, 0, 0, 1, 2, 2]
    var paddings_stack = InlineArray[Scalar[DType.index], 6](1, 0, 0, 1, 2, 2)
    var paddings = NDBuffer[DType.index, 1, _, paddings_stack.size](
        paddings_stack
    )

    # Create an output matrix of the form
    # [[[0 0 0 0 0]
    #   [0 0 0 0 0]]
    #  [[0 0 0 0 0]
    #   [0 0 0 0 0]]]
    var output_stack = InlineArray[
        Scalar[DType.index], Int(out_shape.product())
    ](uninitialized=True)
    var output = NDBuffer[DType.index, 3, _, out_shape](output_stack)
    output.fill(0)

    # pad
    pad_reflect(output, input, paddings.data)

    # output should have the form
    # [[[1 1 1 1 1]
    #   [1 1 1 1 1]]
    #  [[1 1 1 1 1]
    #   [1 1 1 1 1]]]

    # CHECK: 1
    print(output[0, 0, 0])
    # CHECK: 1
    print(output[0, 0, 1])
    # CHECK: 1
    print(output[0, 0, 2])
    # CHECK: 1
    print(output[0, 0, 3])
    # CHECK: 1
    print(output[0, 0, 4])
    # CHECK: 1
    print(output[0, 1, 0])
    # CHECK: 1
    print(output[0, 1, 1])
    # CHECK: 1
    print(output[0, 1, 2])
    # CHECK: 1
    print(output[0, 1, 3])
    # CHECK: 1
    print(output[0, 1, 4])
    # CHECK: 1
    print(output[1, 0, 0])
    # CHECK: 1
    print(output[1, 0, 1])
    # CHECK: 1
    print(output[1, 0, 2])
    # CHECK: 1
    print(output[1, 0, 3])
    # CHECK: 1
    print(output[1, 0, 4])
    # CHECK: 1
    print(output[1, 1, 0])
    # CHECK: 1
    print(output[1, 1, 1])
    # CHECK: 1
    print(output[1, 1, 2])
    # CHECK: 1
    print(output[1, 1, 3])
    # CHECK: 1
    print(output[1, 1, 4])


# CHECK-LABEL: test_pad_reflect_4d_big_input
fn test_pad_reflect_4d_big_input():
    print("== test_pad_reflect_4d_big_input")

    alias in_shape = DimList(1, 1, 512, 512)
    alias in_size = 1 * 1 * 512 * 512
    alias out_shape = DimList(2, 3, 1024, 1024)
    alias out_size = 2 * 3 * 1024 * 1024

    # create a big input matrix and fill it with ones
    var input_ptr = UnsafePointer[Scalar[DType.index]].alloc(in_size)
    var input = NDBuffer[DType.index, 4, _, in_shape](input_ptr, in_shape)
    input.fill(1)

    # create a padding array of the form
    # [1, 0, 1, 1, 256, 256, 256, 256]
    var paddings_stack = InlineArray[Scalar[DType.index], 8](
        1, 0, 1, 1, 256, 256, 256, 256
    )
    var paddings = NDBuffer[DType.index, 1, _, 8](paddings_stack)

    # create an even bigger output matrix and fill it with zeros
    var output_ptr = UnsafePointer[Scalar[DType.index]].alloc(out_size)
    var output = NDBuffer[DType.index, 4, _, out_shape](output_ptr, out_shape)
    output.fill(0)

    # pad
    pad_reflect(output, input, paddings.data)

    # CHECK: 1
    print(output[0, 0, 0, 0])

    input_ptr.free()
    output_ptr.free()


# CHECK-LABEL: test_pad_repeat_3d
fn test_pad_repeat_3d():
    print("== test_pad_repeat_3d")
    alias in_shape = DimList(2, 2, 2)
    alias out_shape = DimList(5, 4, 3)

    # Create an input matrix of the form
    # [[[1, 2],
    #   [3, 4]],
    #  [[1, 2],
    #   [3 ,4]]]
    var input_stack = InlineArray[Scalar[DType.index], Int(in_shape.product())](
        uninitialized=True
    )
    var input = NDBuffer[DType.index, 3, _, in_shape](input_stack)
    input[IndexList[3](0, 0, 0)] = 1
    input[IndexList[3](0, 0, 1)] = 2
    input[IndexList[3](0, 1, 0)] = 3
    input[IndexList[3](0, 1, 1)] = 4
    input[IndexList[3](1, 0, 0)] = 1
    input[IndexList[3](1, 0, 1)] = 2
    input[IndexList[3](1, 1, 0)] = 3
    input[IndexList[3](1, 1, 1)] = 4

    # Create a padding array of the form
    # [1, 1, 0, 1, 1, 0]
    var paddings_stack = InlineArray[Scalar[DType.index], 6](1, 2, 0, 2, 0, 1)
    var paddings = NDBuffer[DType.index, 1, _, 6](paddings_stack)

    # Create an output array equivalent to np.zeros((5, 4, 3))
    var output_stack = InlineArray[
        Scalar[DType.index], Int(out_shape.product())
    ](uninitialized=True)
    var output = NDBuffer[DType.index, 3, _, out_shape](output_stack)
    output.fill(0)

    # pad
    pad_repeat(output, input, paddings.data)

    # output should have form
    # [[[1, 2, 2],
    #   [3, 4, 4],
    #   [3, 4, 4],
    #   [3, 4, 4]],
    #
    #  [[1, 2, 2],
    #   [3, 4, 4],
    #   [3, 4, 4],
    #   [3, 4, 4]],
    #
    #  [[1, 2, 2],
    #   [3, 4, 4],
    #   [3, 4, 4],
    #   [3, 4, 4]],
    #
    #  [[1, 2, 2],
    #   [3, 4, 4],
    #   [3, 4, 4],
    #   [3, 4, 4]],
    #
    #  [[1, 2, 2],
    #   [3, 4, 4],
    #   [3, 4, 4],
    #   [3, 4, 4]]]

    # CHECK: 1
    print(output[0, 0, 0])
    # CHECK: 2
    print(output[0, 0, 1])
    # CHECK: 2
    print(output[0, 0, 2])
    # CHECK: 3
    print(output[0, 1, 0])
    # CHECK: 4
    print(output[0, 1, 1])
    # CHECK: 4
    print(output[0, 1, 2])
    # CHECK: 3
    print(output[0, 2, 0])
    # CHECK: 4
    print(output[0, 2, 1])
    # CHECK: 4
    print(output[0, 2, 2])
    # CHECK: 3
    print(output[0, 3, 0])
    # CHECK: 4
    print(output[0, 3, 1])
    # CHECK: 4
    print(output[0, 3, 2])
    # CHECK: 1
    print(output[1, 0, 0])
    # CHECK: 2
    print(output[1, 0, 1])
    # CHECK: 2
    print(output[1, 0, 2])
    # CHECK: 3
    print(output[1, 1, 0])
    # CHECK: 4
    print(output[1, 1, 1])
    # CHECK: 4
    print(output[1, 1, 2])
    # CHECK: 3
    print(output[1, 2, 0])
    # CHECK: 4
    print(output[1, 2, 1])
    # CHECK: 4
    print(output[1, 2, 2])
    # CHECK: 3
    print(output[1, 3, 0])
    # CHECK: 4
    print(output[1, 3, 1])
    # CHECK: 4
    print(output[1, 3, 2])
    # CHECK: 1
    print(output[2, 0, 0])
    # CHECK: 2
    print(output[2, 0, 1])
    # CHECK: 2
    print(output[2, 0, 2])
    # CHECK: 3
    print(output[2, 1, 0])
    # CHECK: 4
    print(output[2, 1, 1])
    # CHECK: 4
    print(output[2, 1, 2])
    # CHECK: 3
    print(output[2, 2, 0])
    # CHECK: 4
    print(output[2, 2, 1])
    # CHECK: 4
    print(output[2, 2, 2])
    # CHECK: 3
    print(output[2, 3, 0])
    # CHECK: 4
    print(output[2, 3, 1])
    # CHECK: 4
    print(output[2, 3, 2])
    # CHECK: 1
    print(output[3, 0, 0])
    # CHECK: 2
    print(output[3, 0, 1])
    # CHECK: 2
    print(output[3, 0, 2])
    # CHECK: 3
    print(output[3, 1, 0])
    # CHECK: 4
    print(output[3, 1, 1])
    # CHECK: 4
    print(output[3, 1, 2])
    # CHECK: 3
    print(output[3, 2, 0])
    # CHECK: 4
    print(output[3, 2, 1])
    # CHECK: 4
    print(output[3, 2, 2])
    # CHECK: 3
    print(output[3, 3, 0])
    # CHECK: 4
    print(output[3, 3, 1])
    # CHECK: 4
    print(output[3, 3, 2])
    # CHECK: 1
    print(output[4, 0, 0])
    # CHECK: 2
    print(output[4, 0, 1])
    # CHECK: 2
    print(output[4, 0, 2])
    # CHECK: 3
    print(output[4, 1, 0])
    # CHECK: 4
    print(output[4, 1, 1])
    # CHECK: 4
    print(output[4, 1, 2])
    # CHECK: 3
    print(output[4, 2, 0])
    # CHECK: 4
    print(output[4, 2, 1])
    # CHECK: 4
    print(output[4, 2, 2])
    # CHECK: 3
    print(output[4, 3, 0])
    # CHECK: 4
    print(output[4, 3, 1])
    # CHECK: 4
    print(output[4, 3, 2])


fn main():
    test_pad_1d()
    test_pad_reflect_1d()
    test_pad_repeat_1d()
    test_pad_2d()
    test_pad_reflect_2d()
    test_pad_repeat_2d()
    test_pad_3d()
    test_pad_reflect_3d()
    test_pad_reflect_3d_singleton()
    test_pad_reflect_4d_big_input()
    test_pad_repeat_3d()
