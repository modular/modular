# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# RUN: kgen %s -execute -func='$test_transpose::main():index()' | FileCheck %s

from Buffer import Buffer, NDBuffer
from DType import DType
from Index import Index
from Int import Int
from IO import print
from List import create_kgen_list
from Memory import memset_zero
from Transpose import transpose, transpose_inplace, transpose_nd, _index2D
from Tuple import StaticTuple

# TODO Refactor with `test_broadcast.lit`, e.g., put them into a common file or
# allow NDBuffer to accept StaticIntTuple
fn _index3D(x: Int, y: Int, z: Int) -> StaticTuple[3, __mlir_type.index]:
    return Index(x, y, z).as_tuple()


# CHECK-LABEL: test_transpose_4x4
fn test_transpose_4x4():
    print("== test_transpose_4x4\n")

    # Create a matrix of the form
    # [[0, 1, 2, 3],
    #  [4, 5, 6, 7],
    # ...
    #  [12, 13, 14, 15]]
    var matrix = NDBuffer[
        2,
        create_kgen_list[__mlir_type.index](4, 4),
        DType.index.value,
    ].stack_allocation()

    matrix.__setitem__(_index2D(0, 0), 0)
    matrix.__setitem__(_index2D(0, 1), 1)
    matrix.__setitem__(_index2D(0, 2), 2)
    matrix.__setitem__(_index2D(0, 3), 3)
    matrix.__setitem__(_index2D(1, 0), 4)
    matrix.__setitem__(_index2D(1, 1), 5)
    matrix.__setitem__(_index2D(1, 2), 6)
    matrix.__setitem__(_index2D(1, 3), 7)
    matrix.__setitem__(_index2D(2, 0), 8)
    matrix.__setitem__(_index2D(2, 1), 9)
    matrix.__setitem__(_index2D(2, 2), 10)
    matrix.__setitem__(_index2D(2, 3), 11)
    matrix.__setitem__(_index2D(3, 0), 12)
    matrix.__setitem__(_index2D(3, 1), 13)
    matrix.__setitem__(_index2D(3, 2), 14)
    matrix.__setitem__(_index2D(3, 3), 15)

    transpose_inplace[2, 4, 4, DType.index.value](matrix)

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
    print("== test_transpose_8x8\n")

    var matrix = NDBuffer[
        2,
        create_kgen_list[__mlir_type.index](8, 8),
        DType.index.value,
    ].stack_allocation()

    alias num_rows = 8
    alias num_cols = 8

    var i: Int = 0
    var j: Int = 0
    while i < num_rows:
        j = 0
        while j < num_cols:
            let val = i * num_cols + j
            matrix.__setitem__(_index2D(i, j), val.__as_mlir_index())
            j += 1
        i += 1

    transpose_inplace[2, 8, 8, DType.index.value](matrix)

    i = 0
    while i < num_rows:
        j = 0
        while j < num_cols:
            let expected: Int = j * num_rows + i
            let actual = matrix[i, j][0]
            # CHECK-NOT: Transpose 16x16 failed
            if expected != actual:
                print("Transpose 16x16 failed\n")
            j += 1
        i += 1


# CHECK-LABEL: test_transpose_3x2_out_of_place
fn test_transpose_3x2_out_of_place():
    print("== test_transpose_3x2_out_of_place\n")
    let p1 = Buffer[9, DType.index.value].stack_allocation()
    let p2 = Buffer[9, DType.index.value].stack_allocation()
    var i: Int = 0
    while i < 9:
        p1.__setitem__(i, 0)
        p2.__setitem__(i, 0)
        i += 1

    let src = NDBuffer[
        2, create_kgen_list[__mlir_type.index](3, 2), DType.index.value
    ](p1.data.address)

    let dst = NDBuffer[
        2, create_kgen_list[__mlir_type.index](2, 3), DType.index.value
    ](p2.data.address)

    src.__setitem__(_index2D(0, 0), 0)
    src.__setitem__(_index2D(1, 0), 1)
    src.__setitem__(_index2D(2, 0), 2)
    src.__setitem__(_index2D(0, 1), 3)
    src.__setitem__(_index2D(1, 1), 4)
    src.__setitem__(_index2D(2, 1), 5)
    # transpose a from 4x2 to 2x4
    transpose[2, 2, 3, DType.index.value](dst, src)
    # CHECK: 0
    print(dst[0, 0])
    # CHECK: 1
    print(dst[0, 1])
    # CHECK: 2
    print(dst[0, 2])
    # CHECK: 3
    print(dst[1, 0])
    # CHECK: 4
    print(dst[1, 1])
    # CHECK: 5
    print(dst[1, 2])
    # Overflow should be zero
    # CHECK: 0
    print(dst[2, 0])


# CHECK-LABEL: test_transpose_2d_identity
fn test_transpose_2d_identity():
    print("== test_transpose_2d_identity\n")

    alias in_shape = create_kgen_list[__mlir_type.index](3, 3)
    # Create an input matrix of the form
    # [[1, 2, 3],
    #  [4, 5, 6],
    #  [7, 8, 9]]
    var input = NDBuffer[2, in_shape, DType.index.value].stack_allocation()
    input.__setitem__(_index2D(0, 0), 1)
    input.__setitem__(_index2D(0, 1), 2)
    input.__setitem__(_index2D(0, 2), 3)
    input.__setitem__(_index2D(1, 0), 4)
    input.__setitem__(_index2D(1, 1), 5)
    input.__setitem__(_index2D(1, 2), 6)
    input.__setitem__(_index2D(2, 0), 7)
    input.__setitem__(_index2D(2, 1), 8)
    input.__setitem__(_index2D(2, 2), 9)

    # Create an identity permutation array of the form
    # [0, 1]
    var perm = Buffer[2, DType.index.value].stack_allocation()
    perm.__setitem__(0, 0)
    perm.__setitem__(1, 1)

    # Create an output matrix of the form
    # [[-1, -1, -1],
    #  [-1, -1, -1],
    #  [-1, -1, -1]]
    alias out_shape = create_kgen_list[__mlir_type.index](3, 3)
    var output = NDBuffer[2, out_shape, DType.index.value].stack_allocation()
    memset_zero[DType.index.value](output.data, output.size())

    # transpose
    transpose_nd[2, out_shape, in_shape, DType.index.value](
        output, input, perm.data
    )

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
    print("== test_transpose_2d\n")

    alias in_shape = create_kgen_list[__mlir_type.index](3, 3)
    # Create an input matrix of the form
    # [[1, 2, 3],
    #  [4, 5, 6],
    #  [7, 8, 9]]
    var input = NDBuffer[2, in_shape, DType.index.value].stack_allocation()
    input.__setitem__(_index2D(0, 0), 1)
    input.__setitem__(_index2D(0, 1), 2)
    input.__setitem__(_index2D(0, 2), 3)
    input.__setitem__(_index2D(1, 0), 4)
    input.__setitem__(_index2D(1, 1), 5)
    input.__setitem__(_index2D(1, 2), 6)
    input.__setitem__(_index2D(2, 0), 7)
    input.__setitem__(_index2D(2, 1), 8)
    input.__setitem__(_index2D(2, 2), 9)

    # Create a permutation array of the form
    # [1, 0]
    var perm = Buffer[2, DType.index.value].stack_allocation()
    perm.__setitem__(0, 1)
    perm.__setitem__(1, 0)

    # Create an output matrix of the form
    # [[-1, -1, -1],
    #  [-1, -1, -1],
    #  [-1, -1, -1]]
    alias out_shape = create_kgen_list[__mlir_type.index](3, 3)
    var output = NDBuffer[2, out_shape, DType.index.value].stack_allocation()
    memset_zero[DType.index.value](output.data, output.size())

    # transpose
    transpose_nd[2, out_shape, in_shape, DType.index.value](
        output, input, perm.data
    )

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
    print("== test_transpose_3d_identity\n")

    alias in_shape = create_kgen_list[__mlir_type.index](2, 2, 3)
    # Create an input matrix of the form
    # [[[1, 2, 3],
    #   [4, 5, 6]],
    #  [[7, 8, 9],
    #   [10, 11, 12]]]
    var input = NDBuffer[3, in_shape, DType.index.value].stack_allocation()
    input.__setitem__(_index3D(0, 0, 0), 1)
    input.__setitem__(_index3D(0, 0, 1), 2)
    input.__setitem__(_index3D(0, 0, 2), 3)
    input.__setitem__(_index3D(0, 1, 0), 4)
    input.__setitem__(_index3D(0, 1, 1), 5)
    input.__setitem__(_index3D(0, 1, 2), 6)
    input.__setitem__(_index3D(1, 0, 0), 7)
    input.__setitem__(_index3D(1, 0, 1), 8)
    input.__setitem__(_index3D(1, 0, 2), 9)
    input.__setitem__(_index3D(1, 1, 0), 10)
    input.__setitem__(_index3D(1, 1, 1), 11)
    input.__setitem__(_index3D(1, 1, 2), 12)

    # Create an identity permutation array of the form
    # [0, 1, 2]
    var perm = Buffer[3, DType.index.value].stack_allocation()
    perm.__setitem__(0, 0)
    perm.__setitem__(1, 1)
    perm.__setitem__(2, 2)

    # Create an output matrix of the form
    # [[[-1, -1, -1],
    #   [-1, -1, -1]],
    #  [[-1, -1, -1],
    #   [-1, -1, -1]]]
    alias out_shape = create_kgen_list[__mlir_type.index](2, 2, 3)
    var output = NDBuffer[3, out_shape, DType.index.value].stack_allocation()
    memset_zero[DType.index.value](output.data, output.size())

    # transpose
    transpose_nd[3, out_shape, in_shape, DType.index.value](
        output, input, perm.data
    )

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
    print("== test_transpose_3d\n")

    alias in_shape = create_kgen_list[__mlir_type.index](2, 2, 3)
    # Create an input matrix of the form
    # [[[1, 2, 3],
    #   [4, 5, 6]],
    #  [[7, 8, 9],
    #   [10, 11, 12]]]
    var input = NDBuffer[3, in_shape, DType.index.value].stack_allocation()
    input.__setitem__(_index3D(0, 0, 0), 1)
    input.__setitem__(_index3D(0, 0, 1), 2)
    input.__setitem__(_index3D(0, 0, 2), 3)
    input.__setitem__(_index3D(0, 1, 0), 4)
    input.__setitem__(_index3D(0, 1, 1), 5)
    input.__setitem__(_index3D(0, 1, 2), 6)
    input.__setitem__(_index3D(1, 0, 0), 7)
    input.__setitem__(_index3D(1, 0, 1), 8)
    input.__setitem__(_index3D(1, 0, 2), 9)
    input.__setitem__(_index3D(1, 1, 0), 10)
    input.__setitem__(_index3D(1, 1, 1), 11)
    input.__setitem__(_index3D(1, 1, 2), 12)

    # Create a identity permutation array of the form
    # [2, 0, 1]
    var perm = Buffer[3, DType.index.value].stack_allocation()
    perm.__setitem__(0, 2)
    perm.__setitem__(1, 0)
    perm.__setitem__(2, 1)

    # Create an output matrix of the form
    # [[[-1, -1, -1],
    #   [-1, -1, -1]],
    #  [[-1, -1, -1],
    #   [-1, -1, -1]]]
    alias out_shape = create_kgen_list[__mlir_type.index](3, 2, 2)
    var output = NDBuffer[3, out_shape, DType.index.value].stack_allocation()
    memset_zero[DType.index.value](output.data, output.size())

    # transpose
    transpose_nd[3, out_shape, in_shape, DType.index.value](
        output, input, perm.data
    )

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


@export
fn main() -> __mlir_type.index:
    test_transpose_4x4()
    test_transpose_8x8()
    test_transpose_3x2_out_of_place()
    test_transpose_2d_identity()
    test_transpose_2d()
    test_transpose_3d_identity()
    test_transpose_3d()
    return 0
