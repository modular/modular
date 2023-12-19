# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# RUN: %mojo -debug-level full %s | FileCheck %s

from Concat import (
    _concat_parallel,
    concat,
    variadic_list_to_vector,
    _concat_serial,
)
from memory.buffer import Buffer, DynamicRankBuffer, NDBuffer
from memory.unsafe import DTypePointer
from runtime.llcl import OwningOutputChainPtr, Runtime

from utils.index import StaticIntTuple
from utils.list import Dim, DimList


fn test_concat() raises:
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
        let input_vec = variadic_list_to_vector(input_list)
        let out_chain = OwningOutputChainPtr(rt)
        concat[rank, type, False](
            output_dyn, concat_axis, input_vec, out_chain.borrow()
        )
        out_chain.wait()
        input_vec._del_old()

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
        let input_vec = variadic_list_to_vector(input_list)
        _concat_parallel[rank, type](
            output_dyn, concat_axis, input_vec, out_chain.borrow()
        )
        out_chain.wait()
        input_vec._del_old()

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


# CHECK-LABEL: test_concat_inner
fn test_concat_inner():
    print("== test_concat_inner")

    alias type = DType.float32
    alias rank = 5
    alias concat_axis = 2

    alias s1 = DimList(1, 1, 1, 2, 2)
    alias s2 = DimList(1, 1, 2, 2, 2)
    alias s3 = DimList(1, 1, 3, 2, 2)

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

    alias out_shape = DimList(1, 1, 6, 2, 2)
    let output = NDBuffer[rank, out_shape, type]().stack_allocation()
    output.fill(-1)
    let output_dyn = NDBuffer[rank, DimList.create_unknown[rank](), type](
        output.data, out_shape
    )

    let input_list = VariadicList[
        NDBuffer[rank, DimList.create_unknown[rank](), type]
    ](x1_dyn, x2_dyn, x3_dyn)

    let input_vec = variadic_list_to_vector(input_list)
    _concat_serial[rank, type](output_dyn, concat_axis, input_vec)
    input_vec._del_old()

    # CHECK-COUNT-4: 0.0
    # CHECK-COUNT-8: 1.0
    # CHECK-COUNT-12: 2.0
    for i in range(out_shape.product[rank]().get()):
        print(output.flatten()[i])


fn main() raises:
    test_concat()
    test_concat_parallel()
    test_concat_inner()
