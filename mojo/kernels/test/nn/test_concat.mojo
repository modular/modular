# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# TODO(#19566): Reenable compilation with `-debug-level full`
# RUN: %mojo %s | FileCheck %s

from Concat import _concat_parallel, concat
from memory.buffer import Buffer, DynamicRankBuffer, NDBuffer
from memory.unsafe import DTypePointer
from runtime.llcl import OwningOutputChainPtr, Runtime

from utils.index import StaticIntTuple
from utils.list import Dim, DimList, VariadicList

# FIXME(#18257): Flaky LSAN crashes.
# UNSUPPORTED: asan


fn test_concat():
    print("== test_concat")

    alias type = DType.float32
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
        x1.data, s1
    )
    let x2_dyn = NDBuffer[rank, DimList.create_unknown[rank](), type](
        x2.data, s2
    )
    let x3_dyn = NDBuffer[rank, DimList.create_unknown[rank](), type](
        x3.data, s3
    )

    alias out_shape = DimList(2, 2, 6, 2, 0)
    let output = NDBuffer[rank, out_shape, type].stack_allocation()
    output.fill(-1)
    let output_dyn = NDBuffer[rank, DimList.create_unknown[rank](), type](
        output.data, out_shape
    )

    let input_list = VariadicList[
        NDBuffer[rank, DimList.create_unknown[rank](), type]
    ](x1_dyn, x2_dyn, x3_dyn)

    with Runtime(4) as rt:
        let out_chain = OwningOutputChainPtr(rt)
        concat[rank, type, False](
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

    alias type = DType.float32
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
        x1.data, s1
    )
    let x2_dyn = NDBuffer[rank, DimList.create_unknown[rank](), type](
        x2.data, s2
    )
    let x3_dyn = NDBuffer[rank, DimList.create_unknown[rank](), type](
        x3.data, s3
    )

    alias out_shape = DimList(2, 2, 6, 2, 0)
    let output = NDBuffer[rank, out_shape, type]().stack_allocation()
    output.fill(-1)
    let output_dyn = NDBuffer[rank, DimList.create_unknown[rank](), type](
        output.data, out_shape
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
