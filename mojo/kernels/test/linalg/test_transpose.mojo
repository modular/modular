# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# RUN: kgen %s -execute -func='$test_transpose::main():index()' -I %stdlibdir | FileCheck %s

from Buffer import NDBuffer
from DType import DType
from Int import Int
from IO import print
from List import create_kgen_list
from Transpose import transpose_inplace, _index2D

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
    print(matrix.__getitem__(_index2D(0, 0)))

    # CHECK: 4
    print(matrix.__getitem__(_index2D(0, 1)))

    # CHECK: 8
    print(matrix.__getitem__(_index2D(0, 2)))

    # CHECK: 12
    print(matrix.__getitem__(_index2D(0, 3)))

    # CHECK: 1
    print(matrix.__getitem__(_index2D(1, 0)))

    # CHECK: 5
    print(matrix.__getitem__(_index2D(1, 1)))

    # CHECK: 9
    print(matrix.__getitem__(_index2D(1, 2)))

    # CHECK: 13
    print(matrix.__getitem__(_index2D(1, 3)))

    # CHECK: 2
    print(matrix.__getitem__(_index2D(2, 0)))

    # CHECK: 6
    print(matrix.__getitem__(_index2D(2, 1)))

    # CHECK: 10
    print(matrix.__getitem__(_index2D(2, 2)))

    # CHECK: 14
    print(matrix.__getitem__(_index2D(2, 3)))

    # CHECK: 3
    print(matrix.__getitem__(_index2D(3, 0)))

    # CHECK: 7
    print(matrix.__getitem__(_index2D(3, 1)))

    # CHECK: 11
    print(matrix.__getitem__(_index2D(3, 2)))

    # CHECK: 15
    print(matrix.__getitem__(_index2D(3, 3)))


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
            let actual = matrix.__getitem__(_index2D(i, j)).__getitem__(0)
            # CHECK-NOT: Transpose 16x16 failed
            if expected != actual:
                print("Transpose 16x16 failed\n")
            j += 1
        i += 1


@export
fn main() -> __mlir_type.index:
    test_transpose_4x4()
    test_transpose_8x8()
    return 0
