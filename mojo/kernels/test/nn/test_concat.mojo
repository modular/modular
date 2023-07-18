# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# RUN: %mojo %s | FileCheck %s

from Buffer import Buffer, NDBuffer, DynamicRankBuffer
from IO import print
from Concat import concat, _concat_parallel
from DType import DType
from Pointer import DTypePointer
from Index import StaticIntTuple
from Range import range
from List import Dim, VariadicList, DimList
from LLCL import Runtime, OwningOutputChainPtr


fn test_concat():
    print("== test_concat")

    alias type = DType.float32.value
    alias rank = 4
    alias concat_axis = 2

    alias s1 = DimList(2, 2, 1, 2, 0)
    alias s2 = DimList(2, 2, 2, 2, 0)
    alias s3 = DimList(2, 2, 3, 2, 0)

    let x1 = NDBuffer[rank, s1, type].stack_allocation()
    let x2 = NDBuffer[rank, s2, type].stack_allocation()
    let x3 = NDBuffer[rank, s3, type].stack_allocation()
    x1.fill(0)
    x2.fill(1)
    x3.fill(2)
    let x1_dyn = NDBuffer[rank, DimList.create_unknown[rank](), type](
        x1.data, s1, type
    )
    let x2_dyn = NDBuffer[rank, DimList.create_unknown[rank](), type](
        x2.data, s2, type
    )
    let x3_dyn = NDBuffer[rank, DimList.create_unknown[rank](), type](
        x3.data, s3, type
    )

    alias out_shape = DimList(2, 2, 6, 2, 0)
    let output = NDBuffer[rank, out_shape, type].stack_allocation()
    output.fill(-1)
    let output_dyn = NDBuffer[rank, DimList.create_unknown[rank](), type](
        output.data, out_shape, type
    )

    let input_list = VariadicList[
        NDBuffer[rank, DimList.create_unknown[rank](), type]
    ](x1_dyn, x2_dyn, x3_dyn)

    with Runtime(4) as rt:
        let out_chain = OwningOutputChainPtr(rt)
        concat[rank, type](
            output_dyn, concat_axis, input_list, out_chain.borrow()
        )
        out_chain.wait()

    # CHECK: == test_concat
    # CHECK-COUNT-2: 0.0
    # CHECK-COUNT-4: 1.0
    # CHECK-COUNT-6: 2.0
    # CHECK-COUNT-2: 0.0
    # CHECK-COUNT-4: 1.0
    # CHECK-COUNT-6: 2.0
    # CHECK-COUNT-2: 0.0
    # CHECK-COUNT-4: 1.0
    # CHECK-COUNT-6: 2.0
    for i in range(out_shape.product[rank]().get()):
        print(output.flatten()[i])


fn test_concat_parallel():
    print("== test_concat_parallel")

    alias type = DType.float32.value
    alias rank = 4
    alias concat_axis = 2

    alias s1 = DimList(2, 2, 1, 2, 0)
    alias s2 = DimList(2, 2, 2, 2, 0)
    alias s3 = DimList(2, 2, 3, 2, 0)

    let x1 = NDBuffer[rank, s1, type].stack_allocation()
    let x2 = NDBuffer[rank, s2, type].stack_allocation()
    let x3 = NDBuffer[rank, s3, type].stack_allocation()
    x1.fill(0)
    x2.fill(1)
    x3.fill(2)
    let x1_dyn = NDBuffer[rank, DimList.create_unknown[rank](), type](
        x1.data, s1, type
    )
    let x2_dyn = NDBuffer[rank, DimList.create_unknown[rank](), type](
        x2.data, s2, type
    )
    let x3_dyn = NDBuffer[rank, DimList.create_unknown[rank](), type](
        x3.data, s3, type
    )

    alias out_shape = DimList(2, 2, 6, 2, 0)
    let output = NDBuffer[rank, out_shape, type]().stack_allocation()
    output.fill(-1)
    let output_dyn = NDBuffer[rank, DimList.create_unknown[rank](), type](
        output.data, out_shape, type
    )

    let input_list = VariadicList[
        NDBuffer[rank, DimList.create_unknown[rank](), type]
    ](x1_dyn, x2_dyn, x3_dyn)

    with Runtime(4) as rt:
        let out_chain = OwningOutputChainPtr(rt)
        _concat_parallel[rank, type](
            output_dyn, concat_axis, input_list, out_chain.borrow()
        )
        out_chain.wait()

    # CHECK: == test_concat_parallel
    # CHECK-COUNT-2: 0.0
    # CHECK-COUNT-4: 1.0
    # CHECK-COUNT-6: 2.0
    # CHECK-COUNT-2: 0.0
    # CHECK-COUNT-4: 1.0
    # CHECK-COUNT-6: 2.0
    # CHECK-COUNT-2: 0.0
    # CHECK-COUNT-4: 1.0
    # CHECK-COUNT-6: 2.0
    for i in range(out_shape.product[rank]().get()):
        print(output.flatten()[i])


fn main():
    test_concat()
    test_concat_parallel()
