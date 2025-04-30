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
# UNSUPPORTED: asan
# COM: causes leak in elaborator
# RUN: %mojo-no-debug %s | FileCheck %s

from buffer import NDBuffer
from buffer.dimlist import DimList
from linalg.transpose import (
    _simplify_transpose_perms,
    transpose,
    transpose_inplace,
)

from utils.index import Index, IndexList


# CHECK-LABEL: test_transpose_4x4
fn test_transpose_4x4():
    print("== test_transpose_4x4")

    # Create a matrix of the form
    # [[0, 1, 2, 3],
    #  [4, 5, 6, 7],
    # ...
    #  [12, 13, 14, 15]]
    var matrix = NDBuffer[
        DType.index,
        2,
        MutableAnyOrigin,
        DimList(4, 4),
    ].stack_allocation()

    matrix[IndexList[2](0, 0)] = 0
    matrix[IndexList[2](0, 1)] = 1
    matrix[IndexList[2](0, 2)] = 2
    matrix[IndexList[2](0, 3)] = 3
    matrix[IndexList[2](1, 0)] = 4
    matrix[IndexList[2](1, 1)] = 5
    matrix[IndexList[2](1, 2)] = 6
    matrix[IndexList[2](1, 3)] = 7
    matrix[IndexList[2](2, 0)] = 8
    matrix[IndexList[2](2, 1)] = 9
    matrix[IndexList[2](2, 2)] = 10
    matrix[IndexList[2](2, 3)] = 11
    matrix[IndexList[2](3, 0)] = 12
    matrix[IndexList[2](3, 1)] = 13
    matrix[IndexList[2](3, 2)] = 14
    matrix[IndexList[2](3, 3)] = 15

    transpose_inplace[4, 4, DType.index](matrix)

    # CHECK: 0
    print(matrix[0, 0])

    # CHECK: 4
    print(matrix[0, 1])

    # CHECK: 8
    print(matrix[0, 2])

    # CHECK: 12
    print(matrix[0, 3])

    # CHECK: 1
    print(matrix[1, 0])

    # CHECK: 5
    print(matrix[1, 1])

    # CHECK: 9
    print(matrix[1, 2])

    # CHECK: 13
    print(matrix[1, 3])

    # CHECK: 2
    print(matrix[2, 0])

    # CHECK: 6
    print(matrix[2, 1])

    # CHECK: 10
    print(matrix[2, 2])

    # CHECK: 14
    print(matrix[2, 3])

    # CHECK: 3
    print(matrix[3, 0])

    # CHECK: 7
    print(matrix[3, 1])

    # CHECK: 11
    print(matrix[3, 2])

    # CHECK: 15
    print(matrix[3, 3])


# CHECK-LABEL: test_transpose_8x8
fn test_transpose_8x8():
    print("== test_transpose_8x8")

    alias num_rows: Int = 8
    alias num_cols: Int = 8

    var matrix = NDBuffer[
        DType.index,
        2,
        MutableAnyOrigin,
        DimList(num_rows, num_cols),
    ].stack_allocation()

    for i in range(num_rows):
        for j in range(num_cols):
            var val = i * num_cols + j
            matrix[IndexList[2](i, j)] = val

    transpose_inplace[num_rows, num_cols, DType.index](matrix)

    for i in range(num_rows):
        for j in range(num_cols):
            var expected: Int = j * num_rows + i
            var actual: Int = Int(matrix[i, j][0])
            # CHECK-NOT: Transpose 8x8 failed
            if expected != actual:
                print("Transpose 8x8 failed")


# CHECK-LABEL: test_transpose_16x16
fn test_transpose_16x16():
    print("== test_transpose_16x16")

    alias num_rows: Int = 16
    alias num_cols: Int = 16

    var matrix = NDBuffer[
        DType.index,
        2,
        MutableAnyOrigin,
        DimList(num_rows, num_cols),
    ].stack_allocation()

    for i in range(num_rows):
        for j in range(num_cols):
            var val = i * num_cols + j
            matrix[IndexList[2](i, j)] = val

    transpose_inplace[num_rows, num_cols, DType.index](matrix)

    for i in range(num_rows):
        for j in range(num_cols):
            var expected: Int = j * num_rows + i
            var actual: Int = Int(matrix[i, j][0])
            # CHECK-NOT: Transpose 16x16 failed
            if expected != actual:
                print("Transpose 16x16 failed")


# CHECK-LABEL: test_transpose_2d_identity
fn test_transpose_2d_identity() raises:
    print("== test_transpose_2d_identity")

    alias in_shape = DimList(3, 3)
    # Create an input matrix of the form
    # [[1, 2, 3],
    #  [4, 5, 6],
    #  [7, 8, 9]]
    var input = NDBuffer[
        DType.index, 2, MutableAnyOrigin, in_shape
    ].stack_allocation()
    input[IndexList[2](0, 0)] = 1
    input[IndexList[2](0, 1)] = 2
    input[IndexList[2](0, 2)] = 3
    input[IndexList[2](1, 0)] = 4
    input[IndexList[2](1, 1)] = 5
    input[IndexList[2](1, 2)] = 6
    input[IndexList[2](2, 0)] = 7
    input[IndexList[2](2, 1)] = 8
    input[IndexList[2](2, 2)] = 9

    # Create an identity permutation array of the form
    # [0, 1]
    var perm = NDBuffer[DType.index, 1, MutableAnyOrigin, 2].stack_allocation()
    perm[0] = 0
    perm[1] = 1

    # Create an output matrix of the form
    # [[-1, -1, -1],
    #  [-1, -1, -1],
    #  [-1, -1, -1]]
    alias out_shape = DimList(3, 3)
    var output = NDBuffer[
        DType.index, 2, MutableAnyOrigin, out_shape
    ].stack_allocation()
    output.fill(0)

    # transpose
    transpose(output, input, perm.data)

    # output should have form
    # [[1, 2, 3],
    #  [4, 5, 6],
    #  [7, 8, 9]]

    # CHECK: 1
    print(output[0, 0])
    # CHECK: 2
    print(output[0, 1])
    # CHECK: 3
    print(output[0, 2])
    # CHECK: 4
    print(output[1, 0])
    # CHECK: 5
    print(output[1, 1])
    # CHECK: 6
    print(output[1, 2])
    # CHECK: 7
    print(output[2, 0])
    # CHECK: 8
    print(output[2, 1])
    # CHECK: 9
    print(output[2, 2])


# CHECK-LABEL: test_transpose_2d
fn test_transpose_2d() raises:
    print("== test_transpose_2d")

    alias in_shape = DimList(3, 3)
    # Create an input matrix of the form
    # [[1, 2, 3],
    #  [4, 5, 6],
    #  [7, 8, 9]]
    var input = NDBuffer[
        DType.index, 2, MutableAnyOrigin, in_shape
    ].stack_allocation()
    input[IndexList[2](0, 0)] = 1
    input[IndexList[2](0, 1)] = 2
    input[IndexList[2](0, 2)] = 3
    input[IndexList[2](1, 0)] = 4
    input[IndexList[2](1, 1)] = 5
    input[IndexList[2](1, 2)] = 6
    input[IndexList[2](2, 0)] = 7
    input[IndexList[2](2, 1)] = 8
    input[IndexList[2](2, 2)] = 9

    # Create a permutation array of the form
    # [1, 0]
    var perm = NDBuffer[DType.index, 1, MutableAnyOrigin, 2].stack_allocation()
    perm[0] = 1
    perm[1] = 0

    # Create an output matrix of the form
    # [[-1, -1, -1],
    #  [-1, -1, -1],
    #  [-1, -1, -1]]
    alias out_shape = DimList(3, 3)
    var output = NDBuffer[
        DType.index, 2, MutableAnyOrigin, out_shape
    ].stack_allocation()
    output.fill(0)

    # transpose
    transpose(output, input, perm.data)

    # output should have form
    # [[1, 4, 7],
    #  [2, 5, 8],
    #  [3, 6, 9]]

    # CHECK: 1
    print(output[0, 0])
    # CHECK: 4
    print(output[0, 1])
    # CHECK: 7
    print(output[0, 2])
    # CHECK: 2
    print(output[1, 0])
    # CHECK: 5
    print(output[1, 1])
    # CHECK: 8
    print(output[1, 2])
    # CHECK: 3
    print(output[2, 0])
    # CHECK: 6
    print(output[2, 1])
    # CHECK: 9
    print(output[2, 2])


# CHECK-LABEL: test_transpose_3d_identity
fn test_transpose_3d_identity() raises:
    print("== test_transpose_3d_identity")

    alias in_shape = DimList(2, 2, 3)
    # Create an input matrix of the form
    # [[[1, 2, 3],
    #   [4, 5, 6]],
    #  [[7, 8, 9],
    #   [10, 11, 12]]]
    var input = NDBuffer[
        DType.index, 3, MutableAnyOrigin, in_shape
    ].stack_allocation()
    input[IndexList[3](0, 0, 0)] = 1
    input[IndexList[3](0, 0, 1)] = 2
    input[IndexList[3](0, 0, 2)] = 3
    input[IndexList[3](0, 1, 0)] = 4
    input[IndexList[3](0, 1, 1)] = 5
    input[IndexList[3](0, 1, 2)] = 6
    input[IndexList[3](1, 0, 0)] = 7
    input[IndexList[3](1, 0, 1)] = 8
    input[IndexList[3](1, 0, 2)] = 9
    input[IndexList[3](1, 1, 0)] = 10
    input[IndexList[3](1, 1, 1)] = 11
    input[IndexList[3](1, 1, 2)] = 12

    # Create an identity permutation array of the form
    # [0, 1, 2]
    var perm = NDBuffer[DType.index, 1, MutableAnyOrigin, 3].stack_allocation()
    perm[0] = 0
    perm[1] = 1
    perm[2] = 2

    # Create an output matrix of the form
    # [[[-1, -1, -1],
    #   [-1, -1, -1]],
    #  [[-1, -1, -1],
    #   [-1, -1, -1]]]
    alias out_shape = DimList(2, 2, 3)
    var output = NDBuffer[
        DType.index, 3, MutableAnyOrigin, out_shape
    ].stack_allocation()
    output.fill(0)

    # transpose
    transpose(output, input, perm.data)

    # output should have form
    # [[[1, 2, 3],
    #   [4, 5, 6]],
    #  [[7, 8, 9],
    #   [10, 11, 12]]]

    # CHECK: 1
    print(output[0, 0, 0])
    # CHECK: 2
    print(output[0, 0, 1])
    # CHECK: 3
    print(output[0, 0, 2])
    # CHECK: 4
    print(output[0, 1, 0])
    # CHECK: 5
    print(output[0, 1, 1])
    # CHECK: 6
    print(output[0, 1, 2])
    # CHECK: 7
    print(output[1, 0, 0])
    # CHECK: 8
    print(output[1, 0, 1])
    # CHECK: 9
    print(output[1, 0, 2])
    # CHECK: 10
    print(output[1, 1, 0])
    # CHECK: 11
    print(output[1, 1, 1])
    # CHECK: 12
    print(output[1, 1, 2])


# CHECK-LABEL: test_transpose_3d
fn test_transpose_3d() raises:
    print("== test_transpose_3d")

    alias in_shape = DimList(2, 2, 3)
    # Create an input matrix of the form
    # [[[1, 2, 3],
    #   [4, 5, 6]],
    #  [[7, 8, 9],
    #   [10, 11, 12]]]
    var input = NDBuffer[
        DType.index, 3, MutableAnyOrigin, in_shape
    ].stack_allocation()
    input[IndexList[3](0, 0, 0)] = 1
    input[IndexList[3](0, 0, 1)] = 2
    input[IndexList[3](0, 0, 2)] = 3
    input[IndexList[3](0, 1, 0)] = 4
    input[IndexList[3](0, 1, 1)] = 5
    input[IndexList[3](0, 1, 2)] = 6
    input[IndexList[3](1, 0, 0)] = 7
    input[IndexList[3](1, 0, 1)] = 8
    input[IndexList[3](1, 0, 2)] = 9
    input[IndexList[3](1, 1, 0)] = 10
    input[IndexList[3](1, 1, 1)] = 11
    input[IndexList[3](1, 1, 2)] = 12

    # Create a identity permutation array of the form
    # [2, 0, 1]
    var perm = NDBuffer[DType.index, 1, MutableAnyOrigin, 3].stack_allocation()
    perm[0] = 2
    perm[1] = 0
    perm[2] = 1

    # Create an output matrix of the form
    # [[[-1, -1, -1],
    #   [-1, -1, -1]],
    #  [[-1, -1, -1],
    #   [-1, -1, -1]]]
    alias out_shape = DimList(3, 2, 2)
    var output = NDBuffer[
        DType.index, 3, MutableAnyOrigin, out_shape
    ].stack_allocation()
    output.fill(0)

    # transpose
    transpose(output, input, perm.data)

    # output should have form (easily verifiable via numpy)
    # [[[1, 4],
    #   [7, 10]],
    #  [[2, 5],
    #   [8, 11]]
    #  [[3, 6],
    #   [9, 12]]]

    # CHECK: 1
    print(output[0, 0, 0])
    # CHECK: 4
    print(output[0, 0, 1])
    # CHECK: 7
    print(output[0, 1, 0])
    # CHECK: 10
    print(output[0, 1, 1])
    # CHECK: 2
    print(output[1, 0, 0])
    # CHECK: 5
    print(output[1, 0, 1])
    # CHECK: 8
    print(output[1, 1, 0])
    # CHECK: 11
    print(output[1, 1, 1])
    # CHECK: 3
    print(output[2, 0, 0])
    # CHECK: 6
    print(output[2, 0, 1])
    # CHECK: 9
    print(output[2, 1, 0])
    # CHECK: 12
    print(output[2, 1, 1])


# CHECK-LABEL: test_transpose_si64
fn test_transpose_si64() raises:
    print("== test_transpose_si64")

    alias in_shape = DimList(2, 2, 3)
    # Create an input matrix of the form
    # [[[1, 2, 3],
    #   [4, 5, 6]],
    #  [[7, 8, 9],
    #   [10, 11, 12]]]
    var input = NDBuffer[
        DType.int64, 3, MutableAnyOrigin, in_shape
    ].stack_allocation()
    input[IndexList[3](0, 0, 0)] = 1
    input[IndexList[3](0, 0, 1)] = 2
    input[IndexList[3](0, 0, 2)] = 3
    input[IndexList[3](0, 1, 0)] = 4
    input[IndexList[3](0, 1, 1)] = 5
    input[IndexList[3](0, 1, 2)] = 6
    input[IndexList[3](1, 0, 0)] = 7
    input[IndexList[3](1, 0, 1)] = 8
    input[IndexList[3](1, 0, 2)] = 9
    input[IndexList[3](1, 1, 0)] = 10
    input[IndexList[3](1, 1, 1)] = 11
    input[IndexList[3](1, 1, 2)] = 12

    # Create a identity permutation array of the form
    # [2, 1, 0]
    var perm = NDBuffer[DType.index, 1, MutableAnyOrigin, 3].stack_allocation()
    perm[0] = 2
    perm[1] = 1
    perm[2] = 0

    # Create an output matrix of the form
    # [[[-1, -1, -1],
    #   [-1, -1, -1]],
    #  [[-1, -1, -1],
    #   [-1, -1, -1]]]
    alias out_shape = DimList(3, 2, 2)
    var output = NDBuffer[
        DType.int64, 3, MutableAnyOrigin, out_shape
    ].stack_allocation()
    output.fill(0)

    # transpose
    transpose(output, input, perm.data)

    # output should have form (easily verifiable via numpy)
    # [[[ 1,  7],
    #   [ 4, 10]],
    #  [[ 2,  8],
    #   [ 5, 11]],
    #  [[ 3,  9],
    #   [ 6, 12]]]

    # CHECK: 1
    print(output[0, 0, 0])
    # CHECK: 7
    print(output[0, 0, 1])
    # CHECK: 4
    print(output[0, 1, 0])
    # CHECK: 10
    print(output[0, 1, 1])
    # CHECK: 2
    print(output[1, 0, 0])
    # CHECK: 8
    print(output[1, 0, 1])
    # CHECK: 5
    print(output[1, 1, 0])
    # CHECK: 11
    print(output[1, 1, 1])
    # CHECK: 3
    print(output[2, 0, 0])
    # CHECK: 9
    print(output[2, 0, 1])
    # CHECK: 6
    print(output[2, 1, 0])
    # CHECK: 12
    print(output[2, 1, 1])


# CHECK-LABEL: test_simplify_perm
fn test_simplify_perm():
    print("== test_simplify_perm")
    var perm = IndexList[4](0, 2, 3, 1)
    var shape = IndexList[4](8, 3, 200, 200)
    var rank = 4
    _simplify_transpose_perms[4](rank, shape, perm)
    # CHECK: 0, 2, 1
    print(perm)
    # CHECK: 8, 3, 40000
    print(shape)
    # CHECK: 3
    print(rank)

    rank = 4
    perm = IndexList[4](0, 2, 3, 1)
    shape = IndexList[4](1, 3, 200, 200)
    _simplify_transpose_perms[4](rank, shape, perm)
    # CHECK: 1, 0
    print(perm)
    # CHECK: 3, 40000
    print(shape)
    # CHECK: 2
    print(rank)

    rank = 4
    perm = IndexList[4](0, 2, 1, 3)
    shape = IndexList[4](8, 3, 200, 200)
    _simplify_transpose_perms[4](rank, shape, perm)
    # CHECK: 0, 2, 1, 3
    print(perm)
    # CHECK: 8, 3, 200, 200
    print(shape)
    # CHECK: 4
    print(rank)

    rank = 4
    perm = IndexList[4](0, 2, 1, 3)
    shape = IndexList[4](1, 3, 200, 200)
    _simplify_transpose_perms[4](rank, shape, perm)
    # CHECK: 1, 0, 2
    print(perm)
    # CHECK: 3, 200, 200
    print(shape)
    # CHECK: 3
    print(rank)

    rank = 4
    perm = IndexList[4](2, 1, 0, 3)
    shape = IndexList[4](1, 3, 200, 200)
    _simplify_transpose_perms[4](rank, shape, perm)
    # CHECK: 1, 0, 2
    print(perm)
    # CHECK: 3, 200, 200
    print(shape)
    # CHECK: 3
    print(rank)

    rank = 4
    perm = IndexList[4](3, 2, 1, 0)
    shape = IndexList[4](1, 3, 200, 200)
    _simplify_transpose_perms[4](rank, shape, perm)
    # CHECK: 2, 1, 0
    print(perm)
    # CHECK: 3, 200, 200
    print(shape)
    # CHECK: 3
    print(rank)

    rank = 4
    perm = IndexList[4](0, 2, 1, 3)
    shape = IndexList[4](1, 3, 1, 200)
    _simplify_transpose_perms[4](rank, shape, perm)
    # CHECK: 0
    print(perm)
    # CHECK: 600
    print(shape)
    # CHECK: 1
    print(rank)

    rank = 4
    perm = IndexList[4](0, 3, 1, 2)
    shape = IndexList[4](9, 1, 2, 3)
    _simplify_transpose_perms[4](rank, shape, perm)
    # CHECK: 0, 2, 1
    print(perm)
    # CHECK: 9, 2, 3
    print(shape)
    # CHECK: 3
    print(rank)

    rank = 2
    var perm2 = IndexList[2](0, 1)
    var shape2 = IndexList[2](20, 30)
    _simplify_transpose_perms[2](rank, shape2, perm2)
    # CHECK: 0
    print(perm2)
    # CHECK: 600
    print(shape2)
    # CHECK: 1
    print(rank)

    rank = 2
    perm2 = IndexList[2](1, 0)
    shape2 = IndexList[2](20, 30)
    _simplify_transpose_perms[2](rank, shape2, perm2)
    # CHECK: 1, 0
    print(perm2)
    # CHECK: 20, 30
    print(shape2)
    # CHECK: 2
    print(rank)

    rank = 2
    perm2 = IndexList[2](1, 0)
    shape2 = IndexList[2](20, 1)
    _simplify_transpose_perms[2](rank, shape2, perm2)
    # CHECK: 0
    print(perm2)
    # CHECK: 20
    print(shape2)
    # CHECK: 1
    print(rank)


fn main() raises:
    test_transpose_4x4()
    test_transpose_8x8()
    test_transpose_16x16()
    test_transpose_2d_identity()
    test_transpose_2d()
    test_transpose_3d_identity()
    test_transpose_3d()
    test_transpose_si64()
    test_simplify_perm()
