# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# RUN: kgen %s -execute -func='$lit_stdlib_ndbuffer_indexing::main():index()' -I %stdlibdir | FileCheck %s

from Buffer import Buffer, NDBuffer
from Int import Int
from List import create_kgen_list
from IO import print
from Tuple import StaticTuple


# CHECK-LABEL: test_ndbuffer_indexing
fn test_ndbuffer_indexing():
    print("== test_ndbuffer_indexing\n")

    # The total amount of data to allocate
    alias total_buffer_size: __mlir_type.index = 2 * 3 * 4 * 5 * 6

    # Create a buffer for indexing test:
    var _data = __mlir_op.`pop.stack_allocation`[
        count:total_buffer_size,
        _type : __mlir_type.`!pop.pointer<scalar<index>>`,
    ]()

    # Fill data with increasing order, so that the value of each element in
    #  the test buffer is equal to it's linear index.:
    var fillBufferView = Buffer[
        total_buffer_size,
        __mlir_attr.`#kgen.dtype.constant<index> : !kgen.dtype`,
    ](_data)

    var fillIdx: Int = 0
    while fillIdx < total_buffer_size:
        fillBufferView.__setitem__(fillIdx, fillIdx.__as_mlir_index())
        fillIdx += 1

    # ===------------------------------------------------------------------=== #
    # Test 1DBuffer:
    # ===------------------------------------------------------------------=== #

    var bufferView1D = NDBuffer[
        1,
        create_kgen_list[
            __mlir_type.index,
        ](6),
        __mlir_attr.`#kgen.dtype.constant<index> : !kgen.dtype`,
    ](_data)

    # Try to access element[5]
    let index1d = StaticTuple[1, __mlir_type.index].constant(
        create_kgen_list[__mlir_type.index](5)
    )

    # CHECK: [5]
    print[1, __mlir_attr.`#kgen.dtype.constant<index> : !kgen.dtype`](
        bufferView1D.__getitem__(index1d)
    )

    # ===------------------------------------------------------------------=== #
    # Test 2DBuffer:
    # ===------------------------------------------------------------------=== #

    var bufferView2D = NDBuffer[
        2,
        create_kgen_list[__mlir_type.index](5, 6),
        __mlir_attr.`#kgen.dtype.constant<index> : !kgen.dtype`,
    ](_data)

    # Try to access element[4,5]
    let index2d = StaticTuple[2, __mlir_type.index].constant(
        create_kgen_list[__mlir_type.index](4, 5)
    )

    # Result should be 4*6+5 = 29
    # CHECK: [29]
    print[1, __mlir_attr.`#kgen.dtype.constant<index> : !kgen.dtype`](
        bufferView2D.__getitem__(index2d)
    )

    # ===------------------------------------------------------------------=== #
    # Test 3DBuffer:
    # ===------------------------------------------------------------------=== #

    var bufferView3D = NDBuffer[
        3,
        create_kgen_list[
            __mlir_type.index,
        ](4, 5, 6),
        __mlir_attr.`#kgen.dtype.constant<index> : !kgen.dtype`,
    ](_data)

    # Try to access element[3,4,5]
    let index3d = StaticTuple[3, __mlir_type.index].constant(
        create_kgen_list[__mlir_type.index](3, 4, 5)
    )

    # Result should be 3*(5*6)+4*6+5 = 119
    # CHECK: [119]
    print[1, __mlir_attr.`#kgen.dtype.constant<index> : !kgen.dtype`](
        bufferView3D.__getitem__(index3d)
    )

    # ===------------------------------------------------------------------=== #
    # Test 4DBuffer:
    # ===------------------------------------------------------------------=== #

    var bufferView4D = NDBuffer[
        4,
        create_kgen_list[__mlir_type.index](3, 4, 5, 6),
        __mlir_attr.`#kgen.dtype.constant<index> : !kgen.dtype`,
    ](_data)

    # Try to access element[2,3,4,5]
    let index4d = StaticTuple[4, __mlir_type.index].constant(
        create_kgen_list[__mlir_type.index](2, 3, 4, 5)
    )

    # Result should be 2*4*5*6+3*5*6+4*6+5 = 359
    # CHECK: [359]
    print[1, __mlir_attr.`#kgen.dtype.constant<index> : !kgen.dtype`](
        bufferView4D.__getitem__(index4d)
    )

    # ===------------------------------------------------------------------=== #
    # Test 5DBuffer:
    # ===------------------------------------------------------------------=== #

    var bufferView5D = NDBuffer[
        5,
        create_kgen_list[__mlir_type.index](2, 3, 4, 5, 6),
        __mlir_attr.`#kgen.dtype.constant<index> : !kgen.dtype`,
    ](_data)

    # Try to access element[1,2,3,4,5]
    let index5d = StaticTuple[5, __mlir_type.index].constant(
        create_kgen_list[__mlir_type.index](1, 2, 3, 4, 5)
    )

    # Result should be 1*3*4*5*6+2*4*5*6+3*5*6+4*6+5 = 719
    # CHECK: [719]
    print[1, __mlir_attr.`#kgen.dtype.constant<index> : !kgen.dtype`](
        bufferView5D.__getitem__(index5d)
    )


@export
fn main() -> __mlir_type.index:
    test_ndbuffer_indexing()
    return 0
