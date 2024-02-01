# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# RUN: %mojo -debug-level full %s | FileCheck %s

from math import mul

from algorithm import elementwise
from NN.Arange import arange, arange_shape
from memory import stack_allocation
from memory.buffer import Buffer, NDBuffer
from runtime.llcl import Runtime
from NN.Slice import slice_as_copy, slice_as_view

from utils.index import Index, StaticIntTuple
from utils.list import Dim, DimList


fn print_elements[
    type: DType, in_rank: Int
](tensor: NDBuffer[in_rank, DimList.create_unknown[in_rank](), type]):
    print("New shape:", tensor.dynamic_shape)
    print("New strides:", tensor.dynamic_stride)

    @always_inline
    @parameter
    fn print_elements_lambda[
        simd_width: Int, rank: Int
    ](idx: StaticIntTuple[rank]):
        let index = rebind[StaticIntTuple[in_rank]](idx)
        print(tensor[index])

    with Runtime(1) as runtime:
        elementwise[in_rank, 1, print_elements_lambda](
            rebind[StaticIntTuple[in_rank]](tensor.dynamic_shape),
        )


# slice_dim
fn test_arange[
    dtype: DType,
](start: Int, stop: Int, step: Int):
    let memory1 = stack_allocation[1, dtype, 1]()
    let start_tensor = NDBuffer[1, DimList.create_unknown[1](), dtype](
        memory1, StaticIntTuple[1](1)
    )
    start_tensor[0] = start

    let memory2 = stack_allocation[1, dtype, 1]()
    let stop_tensor = NDBuffer[1, DimList.create_unknown[1](), dtype](
        memory2, StaticIntTuple[1](1)
    )
    stop_tensor[0] = stop

    let memory3 = stack_allocation[1, dtype, 1]()
    let step_tensor = NDBuffer[1, DimList.create_unknown[1](), dtype](
        memory3, StaticIntTuple[1](1)
    )
    step_tensor[0] = step

    var outshape = StaticIntTuple[1]()
    try:
        outshape = arange_shape[dtype, True](
            start_tensor, stop_tensor, step_tensor
        )
    except e:
        print(e)
    print("Expected output shape: ")
    print(outshape)

    alias max_output_size = 64

    if max_output_size < outshape[0]:
        print("Memory is larger than static limit, test failed")
        return

    let memory4 = stack_allocation[max_output_size, dtype, 1]()
    let out_tensor = NDBuffer[1, DimList.create_unknown[1](), dtype](
        memory4, outshape
    )

    with Runtime(1) as runtime:

        @always_inline
        @parameter
        fn arange_lambda[simd_width: Int, rank: Int](idx: StaticIntTuple[rank]):
            let index = rebind[StaticIntTuple[1]](idx)
            let range_val = arange[dtype, simd_width](
                start_tensor, stop_tensor, step_tensor, index
            )
            out_tensor.simd_store[simd_width](index, range_val)

        elementwise[1, 1, arange_lambda](
            rebind[StaticIntTuple[1]](out_tensor.dynamic_shape),
        )

        print_elements[dtype, 1](out_tensor)


# CHECK-LABEL: == test_arrange_basic
fn test_arrange_basic():
    print("== test_arrange_basic")

    # CHECK-NEXT: Expected output shape:
    # CHECK-NEXT: (6,)
    # CHECK-NEXT: New shape: (6,)
    # CHECK-NEXT: New strides: (1,)
    # CHECK-NEXT: 0
    # CHECK-NEXT: 1
    # CHECK-NEXT: 2
    # CHECK-NEXT: 3
    # CHECK-NEXT: 4
    # CHECK-NEXT: 5

    # print(np.arange(0, 6, 1))
    test_arange[DType.int32](0, 6, 1)

    # CHECK-NEXT: Expected output shape:
    # CHECK-NEXT: (3,)
    # CHECK-NEXT: New shape: (3,)
    # CHECK-NEXT: New strides: (1,)
    # CHECK-NEXT: 38
    # CHECK-NEXT: 15
    # CHECK-NEXT: -8

    # print(np.arange(38, -13, -23))
    test_arange[DType.int32](38, -13, -23)


fn main():
    test_arrange_basic()
