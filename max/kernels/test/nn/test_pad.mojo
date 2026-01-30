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

from memory import LegacyUnsafePointer

comptime UnsafePointer = LegacyUnsafePointer[mut=True, ...]

from layout._layout import row_major
from layout._tile_tensor import TileTensor
from nn.pad import pad_constant, pad_reflect, pad_repeat
from testing import assert_equal


# CHECK-LABEL: test_pad_1d
fn test_pad_1d() raises:
    print("== test_pad_1d")

    # Create an input matrix of the form
    # [1, 2, 3]
    var input_stack: InlineArray[Scalar[DType.int], 3] = [
        1,
        2,
        3,
    ]
    var input = TileTensor(input_stack, row_major[3]())

    # Create a padding array of the form
    # [1, 2]
    var paddings_stack: InlineArray[Scalar[DType.int], 2] = [1, 2]
    var paddings = TileTensor(paddings_stack, row_major[2]())

    # Create an output matrix of the form
    # [0, 0, 0, 0, 0, 0]
    var output_stack = InlineArray[Scalar[DType.int], 6](uninitialized=True)
    var output = TileTensor(output_stack, row_major[6]()).fill(0)

    var constant = Scalar[DType.int](5)

    # pad
    pad_constant(output, input, paddings.ptr, constant)

    # output should have form
    # [5, 1, 2, 3, 5, 5]

    assert_equal(output[0], 5)
    assert_equal(output[1], 1)
    assert_equal(output[2], 2)
    assert_equal(output[3], 3)
    assert_equal(output[4], 5)
    assert_equal(output[5], 5)


# CHECK-LABEL: test_pad_reflect_1d
fn test_pad_reflect_1d() raises:
    print("== test_pad_reflect_1d")

    # Create an input matrix of the form
    # [1, 2, 3]
    var input_stack: InlineArray[Scalar[DType.int], 3] = [
        1,
        2,
        3,
    ]
    var input = TileTensor(input_stack, row_major[3]())

    # Create an output matrix of the form
    # [0, 0, 0, 0, 0, 0, 0, 0]
    var output_stack = InlineArray[Scalar[DType.int], 8](uninitialized=True)
    var output = TileTensor(output_stack, row_major[8]()).fill(0)

    # Create a padding array of the form
    # [3, 2]
    var paddings_stack: InlineArray[Scalar[DType.int], 2] = [3, 2]
    var paddings = TileTensor(paddings_stack, row_major[2]())

    # pad
    pad_reflect(output, input, paddings.ptr)

    # output should have form
    # [2, 3, 2, 1, 2, 3, 2, 1]

    assert_equal(output[0], 2)
    assert_equal(output[1], 3)
    assert_equal(output[2], 2)
    assert_equal(output[3], 1)
    assert_equal(output[4], 2)
    assert_equal(output[5], 3)
    assert_equal(output[6], 2)
    assert_equal(output[7], 1)


# CHECK-LABEL: test_pad_repeat_1d
fn test_pad_repeat_1d() raises:
    print("== test_pad_repeat_1d")

    # Create an input matrix of the form
    # [1, 2, 3]
    var input_stack: InlineArray[Scalar[DType.int], 3] = [
        1,
        2,
        3,
    ]
    var input = TileTensor(input_stack, row_major[3]())

    # Create an output matrix of the form
    # [0, 0, 0, 0, 0, 0, 0, 0]
    var output_stack = InlineArray[Scalar[DType.int], 8](uninitialized=True)
    var output = TileTensor(output_stack, row_major[8]()).fill(0)

    # Create a padding array of the form
    # [3, 2]
    var paddings_stack: InlineArray[Scalar[DType.int], 2] = [3, 2]
    var paddings = TileTensor(paddings_stack, row_major[2]())

    # pad
    pad_repeat(output, input, paddings.ptr)

    # output should have form
    # [1, 1, 1, 1, 2, 3, 3, 3]

    assert_equal(output[0], 1)
    assert_equal(output[1], 1)
    assert_equal(output[2], 1)
    assert_equal(output[3], 1)
    assert_equal(output[4], 2)
    assert_equal(output[5], 3)
    assert_equal(output[6], 3)
    assert_equal(output[7], 3)


# CHECK-LABEL: test_pad_2d
fn test_pad_2d() raises:
    print("== test_pad_2d")

    # Create an input matrix of the form
    # [[1, 2],
    #  [3, 4]]
    var input_stack = InlineArray[Scalar[DType.int], 4](uninitialized=True)
    var input = TileTensor(input_stack, row_major[2, 2]())
    input[0, 0] = 1
    input[0, 1] = 2
    input[1, 0] = 3
    input[1, 1] = 4

    # Create a padding array of the form
    # [1, 0, 1, 1]
    var paddings_stack: InlineArray[Scalar[DType.int], 4] = [1, 0, 1, 1]
    var paddings = TileTensor(paddings_stack, row_major[4]())

    # Create an output matrix of the form
    # [[0, 0, 0, 0]
    #  [0, 0, 0, 0]
    #  [0, 0, 0, 0]]
    var output_stack = InlineArray[Scalar[DType.int], 12](uninitialized=True)
    var output = TileTensor(output_stack, row_major[3, 4]()).fill(0)

    var constant = Scalar[DType.int](6)

    # pad
    pad_constant(output, input, paddings.ptr, constant)

    # output should have form
    # [[6, 6, 6, 6]
    #  [6, 1, 2, 6]
    #  [6, 3, 4, 6]]

    assert_equal(output[0, 0], 6)
    assert_equal(output[0, 1], 6)
    assert_equal(output[0, 2], 6)
    assert_equal(output[0, 3], 6)
    assert_equal(output[1, 0], 6)
    assert_equal(output[1, 1], 1)
    assert_equal(output[1, 2], 2)
    assert_equal(output[1, 3], 6)
    assert_equal(output[2, 0], 6)
    assert_equal(output[2, 1], 3)
    assert_equal(output[2, 2], 4)
    assert_equal(output[2, 3], 6)


# CHECK-LABEL: test_pad_reflect_2d
fn test_pad_reflect_2d() raises:
    print("== test_pad_reflect_2d")

    # Create an input matrix of the form
    # [[1, 2],
    #  [3, 4]]
    var input_stack = InlineArray[Scalar[DType.int], 4](uninitialized=True)
    var input = TileTensor(input_stack, row_major[2, 2]())
    input[0, 0] = 1
    input[0, 1] = 2
    input[1, 0] = 3
    input[1, 1] = 4

    # Create a padding array of the form
    # [2, 2, 1, 0]
    var paddings_stack: InlineArray[Scalar[DType.int], 4] = [2, 2, 1, 0]
    var paddings = TileTensor(paddings_stack, row_major[4]())

    # Create an output matrix of the form
    # [[0 0 0]
    #  [0 0 0]
    #  [0 0 0]
    #  [0 0 0]
    #  [0 0 0]
    #  [0 0 0]]
    var output_stack = InlineArray[Scalar[DType.int], 18](uninitialized=True)
    var output = TileTensor(output_stack, row_major[6, 3]()).fill(0)

    # pad
    pad_reflect(output, input, paddings.ptr)

    # output should have form
    # [[2 1 2]
    #  [4 3 4]
    #  [2 1 2]
    #  [4 3 4]
    #  [2 1 2]
    #  [4 3 4]]

    assert_equal(output[0, 0], 2)
    assert_equal(output[0, 1], 1)
    assert_equal(output[0, 2], 2)
    assert_equal(output[1, 0], 4)
    assert_equal(output[1, 1], 3)
    assert_equal(output[1, 2], 4)
    assert_equal(output[2, 0], 2)
    assert_equal(output[2, 1], 1)
    assert_equal(output[2, 2], 2)
    assert_equal(output[3, 0], 4)
    assert_equal(output[3, 1], 3)
    assert_equal(output[3, 2], 4)
    assert_equal(output[4, 0], 2)
    assert_equal(output[4, 1], 1)
    assert_equal(output[4, 2], 2)
    assert_equal(output[5, 0], 4)
    assert_equal(output[5, 1], 3)
    assert_equal(output[5, 2], 4)


# CHECK-LABEL: test_pad_repeat_2d
fn test_pad_repeat_2d() raises:
    print("== test_pad_repeat_2d")

    # Create an input matrix of the form
    # [[1, 2],
    #  [3, 4]]
    var input_stack = InlineArray[Scalar[DType.int], 4](uninitialized=True)
    var input = TileTensor(input_stack, row_major[2, 2]())
    input[0, 0] = 1
    input[0, 1] = 2
    input[1, 0] = 3
    input[1, 1] = 4

    # Create a padding array of the form
    # [2, 2, 1, 0]
    var paddings_stack: InlineArray[Scalar[DType.int], 4] = [2, 2, 1, 0]
    var paddings = TileTensor(paddings_stack, row_major[4]())

    # Create an output matrix of the form
    # [[0 0 0]
    #  [0 0 0]
    #  [0 0 0]
    #  [0 0 0]
    #  [0 0 0]
    #  [0 0 0]]
    var output_stack = InlineArray[Scalar[DType.int], 18](uninitialized=True)
    var output = TileTensor(output_stack, row_major[6, 3]()).fill(0)

    # pad
    pad_repeat(output, input, paddings.ptr)

    # output should have form
    # [[1, 1, 2],
    #  [1, 1, 2],
    #  [1, 1, 2],
    #  [3, 3, 4],
    #  [3, 3, 4],
    #  [3, 3, 4]]

    assert_equal(output[0, 0], 1)
    assert_equal(output[0, 1], 1)
    assert_equal(output[0, 2], 2)
    assert_equal(output[1, 0], 1)
    assert_equal(output[1, 1], 1)
    assert_equal(output[1, 2], 2)
    assert_equal(output[2, 0], 1)
    assert_equal(output[2, 1], 1)
    assert_equal(output[2, 2], 2)
    assert_equal(output[3, 0], 3)
    assert_equal(output[3, 1], 3)
    assert_equal(output[3, 2], 4)
    assert_equal(output[4, 0], 3)
    assert_equal(output[4, 1], 3)
    assert_equal(output[4, 2], 4)
    assert_equal(output[5, 0], 3)
    assert_equal(output[5, 1], 3)
    assert_equal(output[5, 2], 4)


# CHECK-LABEL: test_pad_3d
fn test_pad_3d() raises:
    print("== test_pad_3d")

    # Create an input matrix of the form
    # [[[1, 2],
    #   [3, 4]]]
    var input_stack = InlineArray[Scalar[DType.int], 4](uninitialized=True)
    var input = TileTensor(input_stack, row_major[1, 2, 2]())
    input[0, 0, 0] = 1
    input[0, 0, 1] = 2
    input[0, 1, 0] = 3
    input[0, 1, 1] = 4

    # Create a padding array of the form
    # [1, 0, 0, 1, 1, 0]
    var paddings_stack: InlineArray[Scalar[DType.int], 6] = [1, 0, 0, 1, 1, 0]
    var paddings = TileTensor(paddings_stack, row_major[6]())

    # Create an output matrix of the form
    # [[[0, 0, 0]
    #   [0, 0, 0]
    #   [0, 0, 0]]
    #  [[0, 0, 0]
    #   [0, 0, 0]
    #   [0, 0, 0]]]
    var output_stack = InlineArray[Scalar[DType.int], 18](uninitialized=True)
    var output = TileTensor(output_stack, row_major[2, 3, 3]()).fill(0)

    var constant = Scalar[DType.int](7)

    # pad
    pad_constant(output, input, paddings.ptr, constant)

    # output should have form
    # [[[7, 7, 7]
    #   [7, 7, 7]
    #   [7, 7, 7]]
    #  [[7, 1, 2]
    #   [7, 3, 4]
    #   [7, 7, 7]]]

    assert_equal(output[0, 0, 0], 7)
    assert_equal(output[0, 0, 1], 7)
    assert_equal(output[0, 0, 2], 7)
    assert_equal(output[0, 1, 0], 7)
    assert_equal(output[0, 1, 1], 7)
    assert_equal(output[0, 1, 2], 7)
    assert_equal(output[0, 2, 0], 7)
    assert_equal(output[0, 2, 1], 7)
    assert_equal(output[0, 2, 2], 7)
    assert_equal(output[1, 0, 0], 7)
    assert_equal(output[1, 0, 1], 1)
    assert_equal(output[1, 0, 2], 2)
    assert_equal(output[1, 1, 0], 7)
    assert_equal(output[1, 1, 1], 3)
    assert_equal(output[1, 1, 2], 4)
    assert_equal(output[1, 2, 0], 7)
    assert_equal(output[1, 2, 1], 7)
    assert_equal(output[1, 2, 2], 7)


# CHECK-LABEL: test_pad_reflect_3d
fn test_pad_reflect_3d() raises:
    print("== test_pad_reflect_3d")

    # Create an input matrix of the form
    # [[[1, 2],
    #   [3, 4]],
    #  [[1, 2],
    #   [3 ,4]]]
    var input_stack = InlineArray[Scalar[DType.int], 8](uninitialized=True)
    var input = TileTensor(input_stack, row_major[2, 2, 2]())
    input[0, 0, 0] = 1
    input[0, 0, 1] = 2
    input[0, 1, 0] = 3
    input[0, 1, 1] = 4
    input[1, 0, 0] = 1
    input[1, 0, 1] = 2
    input[1, 1, 0] = 3
    input[1, 1, 1] = 4

    # Create a padding array of the form
    # [1, 1, 0, 1, 1, 0]
    var paddings_stack: InlineArray[Scalar[DType.int], 6] = [1, 1, 0, 1, 1, 0]
    var paddings = TileTensor(paddings_stack, row_major[6]())

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
    var output_stack = InlineArray[Scalar[DType.int], 36](uninitialized=True)
    var output = TileTensor(output_stack, row_major[4, 3, 3]()).fill(0)

    # pad
    pad_reflect(output, input, paddings.ptr)

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

    assert_equal(output[0, 0, 0], 2)
    assert_equal(output[0, 0, 1], 1)
    assert_equal(output[0, 0, 2], 2)
    assert_equal(output[0, 1, 0], 4)
    assert_equal(output[0, 1, 1], 3)
    assert_equal(output[0, 1, 2], 4)
    assert_equal(output[0, 2, 0], 2)
    assert_equal(output[0, 2, 1], 1)
    assert_equal(output[0, 2, 2], 2)
    assert_equal(output[1, 0, 0], 2)
    assert_equal(output[1, 0, 1], 1)
    assert_equal(output[1, 0, 2], 2)
    assert_equal(output[1, 1, 0], 4)
    assert_equal(output[1, 1, 1], 3)
    assert_equal(output[1, 1, 2], 4)
    assert_equal(output[1, 2, 0], 2)
    assert_equal(output[1, 2, 1], 1)
    assert_equal(output[1, 2, 2], 2)
    assert_equal(output[2, 0, 0], 2)
    assert_equal(output[2, 0, 1], 1)
    assert_equal(output[2, 0, 2], 2)
    assert_equal(output[2, 1, 0], 4)
    assert_equal(output[2, 1, 1], 3)
    assert_equal(output[2, 1, 2], 4)
    assert_equal(output[2, 2, 0], 2)
    assert_equal(output[2, 2, 1], 1)
    assert_equal(output[2, 2, 2], 2)
    assert_equal(output[3, 0, 0], 2)
    assert_equal(output[3, 0, 1], 1)
    assert_equal(output[3, 0, 2], 2)
    assert_equal(output[3, 1, 0], 4)
    assert_equal(output[3, 1, 1], 3)
    assert_equal(output[3, 1, 2], 4)
    assert_equal(output[3, 2, 0], 2)
    assert_equal(output[3, 2, 1], 1)
    assert_equal(output[3, 2, 2], 2)


# CHECK-LABEL: test_pad_reflect_3d_singleton
fn test_pad_reflect_3d_singleton() raises:
    print("== test_pad_reflect_3d_singleton")

    # Create an input matrix of the form
    # [[[1]]]
    var input_stack = InlineArray[Scalar[DType.int], 1](uninitialized=True)
    var input = TileTensor(input_stack, row_major[1, 1, 1]())
    input[0, 0, 0] = 1

    # Create a padding array of the form
    # [1, 0, 0, 1, 2, 2]
    var paddings_stack: InlineArray[Scalar[DType.int], 6] = [
        1,
        0,
        0,
        1,
        2,
        2,
    ]
    var paddings = TileTensor(paddings_stack, row_major[6]())

    # Create an output matrix of the form
    # [[[0 0 0 0 0]
    #   [0 0 0 0 0]]
    #  [[0 0 0 0 0]
    #   [0 0 0 0 0]]]
    var output_stack = InlineArray[Scalar[DType.int], 20](uninitialized=True)
    var output = TileTensor(output_stack, row_major[2, 2, 5]()).fill(0)

    # pad
    pad_reflect(output, input, paddings.ptr)

    # output should have the form
    # [[[1 1 1 1 1]
    #   [1 1 1 1 1]]
    #  [[1 1 1 1 1]
    #   [1 1 1 1 1]]]

    assert_equal(output[0, 0, 0], 1)
    assert_equal(output[0, 0, 1], 1)
    assert_equal(output[0, 0, 2], 1)
    assert_equal(output[0, 0, 3], 1)
    assert_equal(output[0, 0, 4], 1)
    assert_equal(output[0, 1, 0], 1)
    assert_equal(output[0, 1, 1], 1)
    assert_equal(output[0, 1, 2], 1)
    assert_equal(output[0, 1, 3], 1)
    assert_equal(output[0, 1, 4], 1)
    assert_equal(output[1, 0, 0], 1)
    assert_equal(output[1, 0, 1], 1)
    assert_equal(output[1, 0, 2], 1)
    assert_equal(output[1, 0, 3], 1)
    assert_equal(output[1, 0, 4], 1)
    assert_equal(output[1, 1, 0], 1)
    assert_equal(output[1, 1, 1], 1)
    assert_equal(output[1, 1, 2], 1)
    assert_equal(output[1, 1, 3], 1)
    assert_equal(output[1, 1, 4], 1)


# CHECK-LABEL: test_pad_reflect_4d_big_input
fn test_pad_reflect_4d_big_input() raises:
    print("== test_pad_reflect_4d_big_input")

    comptime in_size = 1 * 1 * 512 * 512
    comptime out_size = 2 * 3 * 1024 * 1024

    # create a big input matrix and fill it with ones
    var input_ptr = UnsafePointer[Scalar[DType.int]].alloc(in_size)
    var input = TileTensor(input_ptr, row_major[1, 1, 512, 512]()).fill(1)

    # create a padding array of the form
    # [1, 0, 1, 1, 256, 256, 256, 256]
    var paddings_stack: InlineArray[Scalar[DType.int], 8] = [
        1,
        0,
        1,
        1,
        256,
        256,
        256,
        256,
    ]
    var paddings = TileTensor(paddings_stack, row_major[8]())

    # create an even bigger output matrix and fill it with zeros
    var output_ptr = UnsafePointer[Scalar[DType.int]].alloc(out_size)
    var output = TileTensor(output_ptr, row_major[2, 3, 1024, 1024]()).fill(0)

    # pad
    pad_reflect(output, input, paddings.ptr)

    assert_equal(output[0, 0, 0, 0], 1)

    input_ptr.free()
    output_ptr.free()


# CHECK-LABEL: test_pad_repeat_3d
fn test_pad_repeat_3d() raises:
    print("== test_pad_repeat_3d")

    # Create an input matrix of the form
    # [[[1, 2],
    #   [3, 4]],
    #  [[1, 2],
    #   [3 ,4]]]
    var input_stack = InlineArray[Scalar[DType.int], 8](uninitialized=True)
    var input = TileTensor(input_stack, row_major[2, 2, 2]())
    input[0, 0, 0] = 1
    input[0, 0, 1] = 2
    input[0, 1, 0] = 3
    input[0, 1, 1] = 4
    input[1, 0, 0] = 1
    input[1, 0, 1] = 2
    input[1, 1, 0] = 3
    input[1, 1, 1] = 4

    # Create a padding array of the form
    # [1, 1, 0, 1, 1, 0]
    var paddings_stack: InlineArray[Scalar[DType.int], 6] = [1, 2, 0, 2, 0, 1]
    var paddings = TileTensor(paddings_stack, row_major[6]())

    # Create an output array equivalent to np.zeros((5, 4, 3))
    var output_stack = InlineArray[Scalar[DType.int], 60](uninitialized=True)
    var output = TileTensor(output_stack, row_major[5, 4, 3]()).fill(0)

    # pad
    pad_repeat(output, input, paddings.ptr)

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

    assert_equal(output[0, 0, 0], 1)
    assert_equal(output[0, 0, 1], 2)
    assert_equal(output[0, 0, 2], 2)
    assert_equal(output[0, 1, 0], 3)
    assert_equal(output[0, 1, 1], 4)
    assert_equal(output[0, 1, 2], 4)
    assert_equal(output[0, 2, 0], 3)
    assert_equal(output[0, 2, 1], 4)
    assert_equal(output[0, 2, 2], 4)
    assert_equal(output[0, 3, 0], 3)
    assert_equal(output[0, 3, 1], 4)
    assert_equal(output[0, 3, 2], 4)
    assert_equal(output[1, 0, 0], 1)
    assert_equal(output[1, 0, 1], 2)
    assert_equal(output[1, 0, 2], 2)
    assert_equal(output[1, 1, 0], 3)
    assert_equal(output[1, 1, 1], 4)
    assert_equal(output[1, 1, 2], 4)
    assert_equal(output[1, 2, 0], 3)
    assert_equal(output[1, 2, 1], 4)
    assert_equal(output[1, 2, 2], 4)
    assert_equal(output[1, 3, 0], 3)
    assert_equal(output[1, 3, 1], 4)
    assert_equal(output[1, 3, 2], 4)
    assert_equal(output[2, 0, 0], 1)
    assert_equal(output[2, 0, 1], 2)
    assert_equal(output[2, 0, 2], 2)
    assert_equal(output[2, 1, 0], 3)
    assert_equal(output[2, 1, 1], 4)
    assert_equal(output[2, 1, 2], 4)
    assert_equal(output[2, 2, 0], 3)
    assert_equal(output[2, 2, 1], 4)
    assert_equal(output[2, 2, 2], 4)
    assert_equal(output[2, 3, 0], 3)
    assert_equal(output[2, 3, 1], 4)
    assert_equal(output[2, 3, 2], 4)
    assert_equal(output[3, 0, 0], 1)
    assert_equal(output[3, 0, 1], 2)
    assert_equal(output[3, 0, 2], 2)
    assert_equal(output[3, 1, 0], 3)
    assert_equal(output[3, 1, 1], 4)
    assert_equal(output[3, 1, 2], 4)
    assert_equal(output[3, 2, 0], 3)
    assert_equal(output[3, 2, 1], 4)
    assert_equal(output[3, 2, 2], 4)
    assert_equal(output[3, 3, 0], 3)
    assert_equal(output[3, 3, 1], 4)
    assert_equal(output[3, 3, 2], 4)
    assert_equal(output[4, 0, 0], 1)
    assert_equal(output[4, 0, 1], 2)
    assert_equal(output[4, 0, 2], 2)
    assert_equal(output[4, 1, 0], 3)
    assert_equal(output[4, 1, 1], 4)
    assert_equal(output[4, 1, 2], 4)
    assert_equal(output[4, 2, 0], 3)
    assert_equal(output[4, 2, 1], 4)
    assert_equal(output[4, 2, 2], 4)
    assert_equal(output[4, 3, 0], 3)
    assert_equal(output[4, 3, 1], 4)
    assert_equal(output[4, 3, 2], 4)


def main():
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
