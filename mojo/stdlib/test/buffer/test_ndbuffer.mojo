# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# RUN: kgen %s -execute -func='$lit_stdlib_ndbuffer::main():index()' -I %stdlibdir | FileCheck %s

from List import create_kgen_list
from Buffer import NDBuffer, _compute_ndbuffer_offset
from Transpose import _index2D
from IO import print


# CHECK-LABEL: test_ndbuffer
fn test_ndbuffer():
    print("== test_ndbuffer\n")

    # Create a buffer of size 16
    var buffer = __mlir_op.`pop.stack_allocation`[
        count:16, _type : __mlir_type.`!pop.pointer<scalar<index>>`
    ]()

    # Create a matrix of the form
    # [[0, 1, 2, 3],
    #  [4, 5, 6, 7],
    # ...
    #  [12, 13, 14, 15]]
    var matrix = NDBuffer[
        2,
        create_kgen_list[__mlir_type.index](4, 4),
        __mlir_attr.`#kgen.dtype.constant<index> : !kgen.dtype`,
    ](buffer)

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

    # CHECK: 11
    print(
        _compute_ndbuffer_offset[
            2,
            create_kgen_list[__mlir_type.index](4, 4),
            __mlir_attr.`#kgen.dtype.constant<index> : !kgen.dtype`,
        ](matrix, _index2D(2, 3))
    )

    # CHECK: 14
    print(
        _compute_ndbuffer_offset[
            2,
            create_kgen_list[__mlir_type.index](4, 4),
            __mlir_attr.`#kgen.dtype.constant<index> : !kgen.dtype`,
        ](matrix, _index2D(3, 2))
    )

    # CHECK: 15
    print(
        _compute_ndbuffer_offset[
            2,
            create_kgen_list[__mlir_type.index](4, 4),
            __mlir_attr.`#kgen.dtype.constant<index> : !kgen.dtype`,
        ](matrix, _index2D(3, 3))
    )

    # CHECK: 2
    print(matrix.get_rank())

    # CHECK: 16
    print(matrix.size())

    # CHECK: 0
    print(matrix.__getitem__(_index2D(0, 0)))

    # CHECK: 1
    print(matrix.__getitem__(_index2D(0, 1)))

    # CHECK: 2
    print(matrix.__getitem__(_index2D(0, 2)))

    # CHECK: 3
    print(matrix.__getitem__(_index2D(0, 3)))

    # CHECK: 4
    print(matrix.__getitem__(_index2D(1, 0)))

    # CHECK: 5
    print(matrix.__getitem__(_index2D(1, 1)))

    # CHECK: 6
    print(matrix.__getitem__(_index2D(1, 2)))

    # CHECK: 7
    print(matrix.__getitem__(_index2D(1, 3)))

    # CHECK: 8
    print(matrix.__getitem__(_index2D(2, 0)))

    # CHECK: 9
    print(matrix.__getitem__(_index2D(2, 1)))

    # CHECK: 10
    print(matrix.__getitem__(_index2D(2, 2)))

    # CHECK: 11
    print(matrix.__getitem__(_index2D(2, 3)))

    # CHECK: 12
    print(matrix.__getitem__(_index2D(3, 0)))

    # CHECK: 13
    print(matrix.__getitem__(_index2D(3, 1)))

    # CHECK: 14
    print(matrix.__getitem__(_index2D(3, 2)))

    # CHECK: 15
    print(matrix.__getitem__(_index2D(3, 3)))


@export
fn main() -> __mlir_type.index:
    test_ndbuffer()
    return 0
