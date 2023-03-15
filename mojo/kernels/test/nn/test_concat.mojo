# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# RUN: mojo %s | FileCheck %s

from Buffer import Buffer, NDBuffer, DynamicRankBuffer
from IO import print
from Concat import concat
from DType import DType
from Pointer import DTypePointer
from Int import Int
from Range import range
from List import Dim, VariadicList


fn test_concat():
    print("== test_concat\n")
    alias x1_sz = 2 * 2 * 1 * 2
    alias x2_sz = 2 * 2 * 2 * 2
    alias x3_sz = 2 * 2 * 3 * 2

    alias type = DType.f32.value
    alias rank = 4
    alias concat_axis = 2

    let x1 = Buffer[x1_sz, type].stack_allocation().fill(0)
    let x2 = Buffer[x2_sz, type].stack_allocation().fill(1)
    let x3 = Buffer[x3_sz, type].stack_allocation().fill(2)
    let s1 = Buffer[rank, DType.index.value].stack_allocation()
    let s2 = Buffer[rank, DType.index.value].stack_allocation()
    let s3 = Buffer[rank, DType.index.value].stack_allocation()
    for i in range(rank):
        s1[i] = 2
        s2[i] = 2
        s3[i] = 2
    s1[concat_axis] = 1
    s2[concat_axis] = 2
    s3[concat_axis] = 3

    let x1_dyn = DynamicRankBuffer(
        x1.data.bitcast[DType.invalid.value](), rank, s1.data, type
    )
    let x2_dyn = DynamicRankBuffer(
        x2.data.bitcast[DType.invalid.value](), rank, s2.data, type
    )
    let x3_dyn = DynamicRankBuffer(
        x3.data.bitcast[DType.invalid.value](), rank, s3.data, type
    )

    alias out_sz = x1_sz + x2_sz + x3_sz
    let _output = Buffer[out_sz, type].stack_allocation().fill(-1)
    let output = Buffer[Dim(), type](_output.data, _output.__len__())

    let input_list = VariadicList[DynamicRankBuffer](x1_dyn, x2_dyn, x3_dyn)

    concat[type](output, concat_axis, input_list)

    # CHECK-COUNT-2: [0.000000]
    # CHECK-COUNT-4: [1.000000]
    # CHECK-COUNT-6: [2.000000]
    # CHECK-COUNT-2: [0.000000]
    # CHECK-COUNT-4: [1.000000]
    # CHECK-COUNT-6: [2.000000]
    # CHECK-COUNT-2: [0.000000]
    # CHECK-COUNT-4: [1.000000]
    # CHECK-COUNT-6: [2.000000]
    for ii in range(out_sz):
        print(output[ii])


fn main():
    test_concat()
