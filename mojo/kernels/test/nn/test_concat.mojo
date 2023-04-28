# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# RUN: mojo %s | FileCheck %s

from Buffer import Buffer, NDBuffer, DynamicRankBuffer, max_rank
from IO import print
from Concat import concat, _concat_parallel
from DType import DType
from Pointer import DTypePointer
from Index import StaticIntTuple
from Range import range
from List import Dim, VariadicList
from LLCL import Runtime, OwningOutputChainPtr


fn test_concat():
    print("== test_concat")
    alias x1_sz = 2 * 2 * 1 * 2
    alias x2_sz = 2 * 2 * 2 * 2
    alias x3_sz = 2 * 2 * 3 * 2

    alias type = DType.f32.value
    alias rank = 4
    alias concat_axis = 2

    let x1 = Buffer[x1_sz, type].stack_allocation()
    x1.fill(0)
    let x2 = Buffer[x2_sz, type].stack_allocation()
    x2.fill(1)
    let x3 = Buffer[x3_sz, type].stack_allocation()
    x3.fill(2)
    let s1 = StaticIntTuple[max_rank](2, 2, 1, 2, 0)
    let s2 = StaticIntTuple[max_rank](2, 2, 2, 2, 0)
    let s3 = StaticIntTuple[max_rank](2, 2, 3, 2, 0)

    let x1_dyn = DynamicRankBuffer(
        x1.data.bitcast[DType.invalid.value](), rank, s1, type
    )
    let x2_dyn = DynamicRankBuffer(
        x2.data.bitcast[DType.invalid.value](), rank, s2, type
    )
    let x3_dyn = DynamicRankBuffer(
        x3.data.bitcast[DType.invalid.value](), rank, s3, type
    )

    alias out_sz = x1_sz + x2_sz + x3_sz
    let _output = Buffer[out_sz, type].stack_allocation()
    _output.fill(-1)
    let output = Buffer[Dim(), type](_output.data, _output.__len__())

    let output_dyn = DynamicRankBuffer(
        output.data.bitcast[DType.invalid.value](), rank, out_sz, type
    )

    let input_list = VariadicList[DynamicRankBuffer](x1_dyn, x2_dyn, x3_dyn)

    let rt = Runtime(4)
    let out_chain = OwningOutputChainPtr(rt)
    concat(output_dyn, concat_axis, input_list, out_chain.borrow())
    out_chain.wait()

    # CHECK: == test_concat
    # CHECK-COUNT-2: 0.000000
    # CHECK-COUNT-4: 1.000000
    # CHECK-COUNT-6: 2.000000
    # CHECK-COUNT-2: 0.000000
    # CHECK-COUNT-4: 1.000000
    # CHECK-COUNT-6: 2.000000
    # CHECK-COUNT-2: 0.000000
    # CHECK-COUNT-4: 1.000000
    # CHECK-COUNT-6: 2.000000
    for i in range(out_sz):
        print(output[i])


fn test_concat_parallel():
    print("== test_concat_parallel")
    alias x1_sz = 2 * 2 * 1 * 2
    alias x2_sz = 2 * 2 * 2 * 2
    alias x3_sz = 2 * 2 * 3 * 2

    alias type = DType.f32.value
    alias rank = 4
    alias concat_axis = 2

    let x1 = Buffer[x1_sz, type].stack_allocation()
    x1.fill(0)
    let x2 = Buffer[x2_sz, type].stack_allocation()
    x2.fill(1)
    let x3 = Buffer[x3_sz, type].stack_allocation()
    x3.fill(2)
    let s1 = StaticIntTuple[max_rank](2, 2, 1, 2, 0)
    let s2 = StaticIntTuple[max_rank](2, 2, 2, 2, 0)
    let s3 = StaticIntTuple[max_rank](2, 2, 3, 2, 0)

    let x1_dyn = DynamicRankBuffer(
        x1.data.bitcast[DType.invalid.value](), rank, s1, type
    )
    let x2_dyn = DynamicRankBuffer(
        x2.data.bitcast[DType.invalid.value](), rank, s2, type
    )
    let x3_dyn = DynamicRankBuffer(
        x3.data.bitcast[DType.invalid.value](), rank, s3, type
    )

    alias out_sz = x1_sz + x2_sz + x3_sz
    let _output = Buffer[out_sz, type].stack_allocation()
    _output.fill(-1)
    let output = Buffer[Dim(), type](_output.data, _output.__len__())

    let output_dyn = DynamicRankBuffer(
        output.data.bitcast[DType.invalid.value](), rank, out_sz, type
    )

    let input_list = VariadicList[DynamicRankBuffer](x1_dyn, x2_dyn, x3_dyn)

    let rt = Runtime(4)
    let out_chain = OwningOutputChainPtr(rt)
    _concat_parallel(output_dyn, concat_axis, input_list, out_chain.borrow())
    out_chain.wait()

    # CHECK: == test_concat_parallel
    # CHECK-COUNT-2: 0.000000
    # CHECK-COUNT-4: 1.000000
    # CHECK-COUNT-6: 2.000000
    # CHECK-COUNT-2: 0.000000
    # CHECK-COUNT-4: 1.000000
    # CHECK-COUNT-6: 2.000000
    # CHECK-COUNT-2: 0.000000
    # CHECK-COUNT-4: 1.000000
    # CHECK-COUNT-6: 2.000000
    for i in range(out_sz):
        print(output[i])


fn main():
    test_concat()
    test_concat_parallel()
