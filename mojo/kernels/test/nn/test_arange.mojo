# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# RUN: %mojo-no-debug %s | FileCheck %s

from algorithm import elementwise
from buffer import NDBuffer
from buffer.dimlist import Dim, DimList
from memory import stack_allocation
from nn.arange import arange, arange_shape
from nn.slice import slice_as_copy, slice_as_view

from utils.index import Index, IndexList


def print_elements[type: DType, in_rank: Int](tensor: NDBuffer[type, in_rank]):
    print("New shape:", tensor.get_shape())
    print("New strides:", tensor.get_strides())

    @always_inline
    @parameter
    fn print_elements_lambda[simd_width: Int, rank: Int](idx: IndexList[rank]):
        var index = rebind[IndexList[in_rank]](idx)
        print(tensor[index])

    elementwise[print_elements_lambda, 1](tensor.get_shape())


# slice_dim
def test_arange[
    dtype: DType,
](start: Int, stop: Int, step: Int):
    var memory1 = stack_allocation[1, dtype, 1]()
    var start_tensor = NDBuffer[dtype, 1](memory1, IndexList[1](1))
    start_tensor[0] = start

    var memory2 = stack_allocation[1, dtype, 1]()
    var stop_tensor = NDBuffer[dtype, 1](memory2, IndexList[1](1))
    stop_tensor[0] = stop

    var memory3 = stack_allocation[1, dtype, 1]()
    var step_tensor = NDBuffer[dtype, 1](memory3, IndexList[1](1))
    step_tensor[0] = step

    var outshape = IndexList[1]()
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

    var memory4 = stack_allocation[max_output_size, dtype, 1]()
    var out_tensor = NDBuffer[dtype, 1](memory4, outshape)

    @always_inline
    @__copy_capture(out_tensor, step_tensor, start_tensor, stop_tensor)
    @parameter
    fn arange_lambda[simd_width: Int, rank: Int](idx: IndexList[rank]):
        var index = rebind[IndexList[1]](idx)
        var range_val = arange[dtype, simd_width](
            start_tensor, stop_tensor, step_tensor, index
        )
        out_tensor.store[width=simd_width](index, range_val)

    elementwise[arange_lambda, 1](
        rebind[IndexList[1]](out_tensor.get_shape()),
    )

    print_elements[dtype, 1](out_tensor)


# CHECK-LABEL: == test_arrange_basic
def test_arrange_basic():
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


def main():
    test_arrange_basic()
