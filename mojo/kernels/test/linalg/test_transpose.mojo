# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# RUN: mojo %s | FileCheck %s

from Buffer import Buffer, NDBuffer
from DType import DType
from Index import Index, StaticIntTuple
from IO import print
from List import create_dim_list
from Transpose import transpose, transpose_inplace
from Range import range


# CHECK-LABEL: test_transpose_4x4
fn test_transpose_4x4():
    print("== test_transpose_4x4")

    # Create a matrix of the form
    # [[0, 1, 2, 3],
    #  [4, 5, 6, 7],
    # ...
    #  [12, 13, 14, 15]]
    var matrix = NDBuffer[
        2,
        create_dim_list(4, 4),
        DType.index,
    ].stack_allocation()

    matrix[StaticIntTuple[2](0, 0)] = 0
    matrix[StaticIntTuple[2](0, 1)] = 1
    matrix[StaticIntTuple[2](0, 2)] = 2
    matrix[StaticIntTuple[2](0, 3)] = 3
    matrix[StaticIntTuple[2](1, 0)] = 4
    matrix[StaticIntTuple[2](1, 1)] = 5
    matrix[StaticIntTuple[2](1, 2)] = 6
    matrix[StaticIntTuple[2](1, 3)] = 7
    matrix[StaticIntTuple[2](2, 0)] = 8
    matrix[StaticIntTuple[2](2, 1)] = 9
    matrix[StaticIntTuple[2](2, 2)] = 10
    matrix[StaticIntTuple[2](2, 3)] = 11
    matrix[StaticIntTuple[2](3, 0)] = 12
    matrix[StaticIntTuple[2](3, 1)] = 13
    matrix[StaticIntTuple[2](3, 2)] = 14
    matrix[StaticIntTuple[2](3, 3)] = 15

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
        2,
        create_dim_list(num_rows, num_cols),
        DType.index,
    ].stack_allocation()

    for i in range(num_rows):
        for j in range(num_cols):
            let val = i * num_cols + j
            matrix[StaticIntTuple[2](i, j)] = val

    transpose_inplace[num_rows, num_cols, DType.index](matrix)

    # TODO(#8365) use `i` and `j`
    for ii in range(num_rows):
        for jj in range(num_cols):
            let expected: Int = jj * num_rows + ii
            let actual: Int = matrix[ii, jj][0].value
            # CHECK-NOT: Transpose 8x8 failed
            if expected != actual:
                print("Transpose 8x8 failed")


# CHECK-LABEL: test_transpose_16x16
fn test_transpose_16x16():
    print("== test_transpose_16x16")

    alias num_rows: Int = 16
    alias num_cols: Int = 16

    var matrix = NDBuffer[
        2,
        create_dim_list(num_rows, num_cols),
        DType.index,
    ].stack_allocation()

    for i in range(num_rows):
        for j in range(num_cols):
            let val = i * num_cols + j
            matrix[StaticIntTuple[2](i, j)] = val

    transpose_inplace[num_rows, num_cols, DType.index](matrix)

    # TODO(#8365) use `i` and `j`
    for ii in range(num_rows):
        for jj in range(num_cols):
            let expected: Int = jj * num_rows + ii
            let actual: Int = matrix[ii, jj][0].value
            # CHECK-NOT: Transpose 16x16 failed
            if expected != actual:
                print("Transpose 16x16 failed")


# CHECK-LABEL: test_transpose_2d_identity
fn test_transpose_2d_identity():
    print("== test_transpose_2d_identity")

    alias in_shape = create_dim_list(3, 3)
    # Create an input matrix of the form
    # [[1, 2, 3],
    #  [4, 5, 6],
    #  [7, 8, 9]]
    var input = NDBuffer[2, in_shape, DType.index].stack_allocation()
    input[StaticIntTuple[2](0, 0)] = 1
    input[StaticIntTuple[2](0, 1)] = 2
    input[StaticIntTuple[2](0, 2)] = 3
    input[StaticIntTuple[2](1, 0)] = 4
    input[StaticIntTuple[2](1, 1)] = 5
    input[StaticIntTuple[2](1, 2)] = 6
    input[StaticIntTuple[2](2, 0)] = 7
    input[StaticIntTuple[2](2, 1)] = 8
    input[StaticIntTuple[2](2, 2)] = 9

    # Create an identity permutation array of the form
    # [0, 1]
    var perm = Buffer[2, DType.index].stack_allocation()
    perm[0] = 0
    perm[1] = 1

    # Create an output matrix of the form
    # [[-1, -1, -1],
    #  [-1, -1, -1],
    #  [-1, -1, -1]]
    alias out_shape = create_dim_list(3, 3)
    var output = NDBuffer[2, out_shape, DType.index].stack_allocation()
    output.fill(0)

    # transpose
    transpose(output, input, perm.data)

    # output should have form
    # [[1, 2, 3],
    #  [4, 5, 6],
    #  [7, 8, 9]]

    # check: 1
    print(output[0, 0])
    # check: 2
    print(output[0, 1])
    # check: 3
    print(output[0, 2])
    # check: 4
    print(output[1, 0])
    # check: 5
    print(output[1, 1])
    # check: 6
    print(output[1, 2])
    # check: 7
    print(output[2, 0])
    # check: 8
    print(output[2, 1])
    # check: 9
    print(output[2, 2])


# CHECK-LABEL: test_transpose_2d
fn test_transpose_2d():
    print("== test_transpose_2d")

    alias in_shape = create_dim_list(3, 3)
    # Create an input matrix of the form
    # [[1, 2, 3],
    #  [4, 5, 6],
    #  [7, 8, 9]]
    var input = NDBuffer[2, in_shape, DType.index].stack_allocation()
    input[StaticIntTuple[2](0, 0)] = 1
    input[StaticIntTuple[2](0, 1)] = 2
    input[StaticIntTuple[2](0, 2)] = 3
    input[StaticIntTuple[2](1, 0)] = 4
    input[StaticIntTuple[2](1, 1)] = 5
    input[StaticIntTuple[2](1, 2)] = 6
    input[StaticIntTuple[2](2, 0)] = 7
    input[StaticIntTuple[2](2, 1)] = 8
    input[StaticIntTuple[2](2, 2)] = 9

    # Create a permutation array of the form
    # [1, 0]
    var perm = Buffer[2, DType.index].stack_allocation()
    perm[0] = 1
    perm[1] = 0

    # Create an output matrix of the form
    # [[-1, -1, -1],
    #  [-1, -1, -1],
    #  [-1, -1, -1]]
    alias out_shape = create_dim_list(3, 3)
    var output = NDBuffer[2, out_shape, DType.index].stack_allocation()
    output.fill(0)

    # transpose
    transpose(output, input, perm.data)

    # output should have form
    # [[1, 4, 7],
    #  [2, 5, 8],
    #  [3, 6, 9]]

    # check: 1
    print(output[0, 0])
    # check: 4
    print(output[0, 1])
    # check: 7
    print(output[0, 2])
    # check: 2
    print(output[1, 0])
    # check: 5
    print(output[1, 1])
    # check: 8
    print(output[1, 2])
    # check: 3
    print(output[2, 0])
    # check: 6
    print(output[2, 1])
    # check: 9
    print(output[2, 2])


# CHECK-LABEL: test_transpose_3d_identity
fn test_transpose_3d_identity():
    print("== test_transpose_3d_identity")

    alias in_shape = create_dim_list(2, 2, 3)
    # Create an input matrix of the form
    # [[[1, 2, 3],
    #   [4, 5, 6]],
    #  [[7, 8, 9],
    #   [10, 11, 12]]]
    var input = NDBuffer[3, in_shape, DType.index].stack_allocation()
    input[StaticIntTuple[3](0, 0, 0)] = 1
    input[StaticIntTuple[3](0, 0, 1)] = 2
    input[StaticIntTuple[3](0, 0, 2)] = 3
    input[StaticIntTuple[3](0, 1, 0)] = 4
    input[StaticIntTuple[3](0, 1, 1)] = 5
    input[StaticIntTuple[3](0, 1, 2)] = 6
    input[StaticIntTuple[3](1, 0, 0)] = 7
    input[StaticIntTuple[3](1, 0, 1)] = 8
    input[StaticIntTuple[3](1, 0, 2)] = 9
    input[StaticIntTuple[3](1, 1, 0)] = 10
    input[StaticIntTuple[3](1, 1, 1)] = 11
    input[StaticIntTuple[3](1, 1, 2)] = 12

    # Create an identity permutation array of the form
    # [0, 1, 2]
    var perm = Buffer[3, DType.index].stack_allocation()
    perm[0] = 0
    perm[1] = 1
    perm[2] = 2

    # Create an output matrix of the form
    # [[[-1, -1, -1],
    #   [-1, -1, -1]],
    #  [[-1, -1, -1],
    #   [-1, -1, -1]]]
    alias out_shape = create_dim_list(2, 2, 3)
    var output = NDBuffer[3, out_shape, DType.index].stack_allocation()
    output.fill(0)

    # transpose
    transpose(output, input, perm.data)

    # output should have form
    # [[[1, 2, 3],
    #   [4, 5, 6]],
    #  [[7, 8, 9],
    #   [10, 11, 12]]]

    # check: 1
    print(output[0, 0, 0])
    # check: 2
    print(output[0, 0, 1])
    # check: 3
    print(output[0, 0, 2])
    # check: 4
    print(output[0, 1, 0])
    # check: 5
    print(output[0, 1, 1])
    # check: 6
    print(output[0, 1, 2])
    # check: 7
    print(output[1, 0, 0])
    # check: 8
    print(output[1, 0, 1])
    # check: 9
    print(output[1, 0, 2])
    # check: 10
    print(output[1, 1, 0])
    # check: 11
    print(output[1, 1, 1])
    # check: 12
    print(output[1, 1, 2])


# CHECK-LABEL: test_transpose_3d
fn test_transpose_3d():
    print("== test_transpose_3d")

    alias in_shape = create_dim_list(2, 2, 3)
    # Create an input matrix of the form
    # [[[1, 2, 3],
    #   [4, 5, 6]],
    #  [[7, 8, 9],
    #   [10, 11, 12]]]
    var input = NDBuffer[3, in_shape, DType.index].stack_allocation()
    input[StaticIntTuple[3](0, 0, 0)] = 1
    input[StaticIntTuple[3](0, 0, 1)] = 2
    input[StaticIntTuple[3](0, 0, 2)] = 3
    input[StaticIntTuple[3](0, 1, 0)] = 4
    input[StaticIntTuple[3](0, 1, 1)] = 5
    input[StaticIntTuple[3](0, 1, 2)] = 6
    input[StaticIntTuple[3](1, 0, 0)] = 7
    input[StaticIntTuple[3](1, 0, 1)] = 8
    input[StaticIntTuple[3](1, 0, 2)] = 9
    input[StaticIntTuple[3](1, 1, 0)] = 10
    input[StaticIntTuple[3](1, 1, 1)] = 11
    input[StaticIntTuple[3](1, 1, 2)] = 12

    # Create a identity permutation array of the form
    # [2, 0, 1]
    var perm = Buffer[3, DType.index].stack_allocation()
    perm[0] = 2
    perm[1] = 0
    perm[2] = 1

    # Create an output matrix of the form
    # [[[-1, -1, -1],
    #   [-1, -1, -1]],
    #  [[-1, -1, -1],
    #   [-1, -1, -1]]]
    alias out_shape = create_dim_list(3, 2, 2)
    var output = NDBuffer[3, out_shape, DType.index].stack_allocation()
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

    # check: 1
    print(output[0, 0, 0])
    # check: 4
    print(output[0, 0, 1])
    # check: 7
    print(output[0, 1, 0])
    # check: 10
    print(output[0, 1, 1])
    # check: 2
    print(output[1, 0, 0])
    # check: 5
    print(output[1, 0, 1])
    # check: 8
    print(output[1, 1, 0])
    # check: 11
    print(output[1, 1, 1])
    # check: 3
    print(output[2, 0, 0])
    # check: 6
    print(output[2, 0, 1])
    # check: 9
    print(output[2, 1, 0])
    # check: 12
    print(output[2, 1, 1])


# CHECK-LABEL: test_transpose_si64
fn test_transpose_si64():
    print("== test_transpose_si64")

    alias in_shape = create_dim_list(2, 2, 3)
    # Create an input matrix of the form
    # [[[1, 2, 3],
    #   [4, 5, 6]],
    #  [[7, 8, 9],
    #   [10, 11, 12]]]
    var input = NDBuffer[3, in_shape, DType.si64].stack_allocation()
    input[StaticIntTuple[3](0, 0, 0)] = 1
    input[StaticIntTuple[3](0, 0, 1)] = 2
    input[StaticIntTuple[3](0, 0, 2)] = 3
    input[StaticIntTuple[3](0, 1, 0)] = 4
    input[StaticIntTuple[3](0, 1, 1)] = 5
    input[StaticIntTuple[3](0, 1, 2)] = 6
    input[StaticIntTuple[3](1, 0, 0)] = 7
    input[StaticIntTuple[3](1, 0, 1)] = 8
    input[StaticIntTuple[3](1, 0, 2)] = 9
    input[StaticIntTuple[3](1, 1, 0)] = 10
    input[StaticIntTuple[3](1, 1, 1)] = 11
    input[StaticIntTuple[3](1, 1, 2)] = 12

    # Create a identity permutation array of the form
    # [2, 1, 0]
    var perm = Buffer[3, DType.index].stack_allocation()
    perm[0] = 2
    perm[1] = 1
    perm[2] = 0

    # Create an output matrix of the form
    # [[[-1, -1, -1],
    #   [-1, -1, -1]],
    #  [[-1, -1, -1],
    #   [-1, -1, -1]]]
    alias out_shape = create_dim_list(3, 2, 2)
    var output = NDBuffer[3, out_shape, DType.si64].stack_allocation()
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

    # check: 1
    print(output[0, 0, 0])
    # check: 7
    print(output[0, 0, 1])
    # check: 4
    print(output[0, 1, 0])
    # check: 10
    print(output[0, 1, 1])
    # check: 2
    print(output[1, 0, 0])
    # check: 8
    print(output[1, 0, 1])
    # check: 5
    print(output[1, 1, 0])
    # check: 11
    print(output[1, 1, 1])
    # check: 3
    print(output[2, 0, 0])
    # check: 6
    print(output[2, 0, 1])
    # check: 9
    print(output[2, 1, 0])
    # check: 12
    print(output[2, 1, 1])


fn main():
    test_transpose_4x4()
    test_transpose_8x8()
    test_transpose_16x16()
    test_transpose_2d_identity()
    test_transpose_2d()
    test_transpose_3d_identity()
    test_transpose_3d()
    test_transpose_si64()
