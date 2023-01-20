# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# RUN: kgen %s -execute -func='$lit_stdlib_ndbuffer_dynamic_shape::main():index()' -I %stdlibdir | FileCheck %s

from List import create_kgen_list
from Buffer import NDBuffer, _compute_ndbuffer_offset
from Transpose import _index2D
from IO import print


# CHECK-LABEL: test_ndbuffer_dynamic_shape
fn test_ndbuffer_dynamic_shape():
    print("== test_ndbuffer_dynamic_shape\n")

    # Create a buffer of size 16
    var buffer = __mlir_op.`pop.stack_allocation`[
        count:16, _type : __mlir_type.`!pop.pointer<scalar<index>>`
    ]()

    var matrix = NDBuffer[
        2,
        create_kgen_list[__mlir_type.index](
            __mlir_attr.`#kgen.unknown : index`,
            __mlir_attr.`#kgen.unknown : index`,
        ),
        __mlir_attr.`#kgen.dtype.constant<index> : !kgen.dtype`,
    ](buffer)

    matrix.dynamic_shape.__setitem__[0](42)
    matrix.dynamic_shape.__setitem__[1](43)

    # CHECK: 42
    print(matrix.dim[0]())
    # CHECK: 43
    print(matrix.dim[1]())

    # Mix static and dynamic shape.
    var matrix2 = NDBuffer[
        2,
        create_kgen_list[__mlir_type.index](
            42, __mlir_attr.`#kgen.unknown : index`
        ),
        __mlir_attr.`#kgen.dtype.constant<index> : !kgen.dtype`,
    ](buffer)

    matrix2.dynamic_shape.__setitem__[1](43)

    # CHECK: 42
    print(matrix2.dim[0]())
    # CHECK: 43
    print(matrix2.dim[1]())


@export
fn main() -> __mlir_type.index:
    test_ndbuffer_dynamic_shape()
    return 0
