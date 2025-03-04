# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# RUN: %mojo-no-debug %s | FileCheck %s

from algorithm.functional import (
    _get_start_indices_of_nth_subvolume,
    elementwise,
)
from buffer import NDBuffer
from buffer.dimlist import Dim, DimList
from collections import InlineArray

from utils.index import Index, IndexList


def test_elementwise[
    numelems: Int, outer_rank: Int, is_blocking: Bool
](dims: DimList):
    var memory1 = InlineArray[Float32, numelems](unsafe_uninitialized=True)
    var buffer1 = NDBuffer[DType.float32, outer_rank](
        memory1.unsafe_ptr(), dims
    )

    var memory2 = InlineArray[Float32, numelems](unsafe_uninitialized=True)
    var buffer2 = NDBuffer[DType.float32, outer_rank](
        memory2.unsafe_ptr(), dims
    )

    var memory3 = InlineArray[Float32, numelems](unsafe_uninitialized=True)
    var out_buffer = NDBuffer[DType.float32, outer_rank](
        memory3.unsafe_ptr(), dims
    )

    var x: Float32 = 1.0
    for i in range(numelems):
        buffer1.data[i] = 2.0
        buffer2.data[i] = Float32(x.value)
        out_buffer.data[i] = 0.0
        x += 1.0

    @always_inline
    @parameter
    fn func[simd_width: Int, rank: Int](idx: IndexList[rank]):
        var index = rebind[IndexList[outer_rank]](idx)
        var in1 = buffer1.load[width=simd_width](index)
        var in2 = buffer2.load[width=simd_width](index)
        out_buffer.store[width=simd_width](index, in1 * in2)

    elementwise[func, simd_width=1, use_blocking_impl=is_blocking](
        rebind[IndexList[outer_rank]](out_buffer.get_shape()),
    )

    for i2 in range(min(numelems, 64)):
        if out_buffer.data.offset(i2).load() != 2 * (i2 + 1):
            print("ERROR")


def test_elementwise_implicit_runtime():
    print("== test_elementwise_implicit_runtime")
    var vector = NDBuffer[DType.index, 1, 20].stack_allocation()

    for i in range(len(vector)):
        vector[i] = i

    @always_inline
    @parameter
    fn func[simd_width: Int, rank: Int](idx: IndexList[rank]):
        vector[idx[0]] = 42

    elementwise[func, simd_width=1](20)

    for i in range(len(vector)):
        if Int(vector[i]) != 42:
            print("ERROR: Expecting the result to be 42")
            return

    print("OK")


fn test_indices_conversion():
    print("== Testing indices conversion:")
    var shape = IndexList[4](3, 4, 5, 6)
    print(_get_start_indices_of_nth_subvolume[0](10, shape))
    print(_get_start_indices_of_nth_subvolume[1](10, shape))
    print(_get_start_indices_of_nth_subvolume[2](10, shape))
    print(_get_start_indices_of_nth_subvolume[3](2, shape))
    print(_get_start_indices_of_nth_subvolume[4](0, shape))


def main():
    # CHECK-LABEL: == Testing 1D:
    # CHECK-NOT: ERROR
    print("== Testing 1D:")
    test_elementwise[16, 1, False](DimList(16))

    # CHECK-LABEL: == Testing 1D blocking:
    # CHECK-NOT: ERROR
    print("== Testing 1D blocking:")
    test_elementwise[16, 1, True](DimList(16))

    # CHECK-LABEL: == Testing 2D:
    # CHECK-NOT: ERROR
    print("== Testing 2D:")
    test_elementwise[16, 2, False](DimList(4, 4))

    # CHECK-LABEL: == Testing 2D blocking:
    # CHECK-NOT: ERROR
    print("== Testing 2D blocking:")
    test_elementwise[16, 2, True](DimList(4, 4))

    # CHECK-LABEL: == Testing 3D:
    # CHECK-NOT: ERROR
    print("== Testing 3D:")
    test_elementwise[16, 3, False](DimList(4, 2, 2))

    # CHECK-LABEL: == Testing 3D blocking:
    # CHECK-NOT: ERROR
    print("== Testing 3D blocking:")
    test_elementwise[16, 3, True](DimList(4, 2, 2))

    # CHECK-LABEL: == Testing 4D:
    # CHECK-NOT: ERROR
    print("== Testing 4D:")
    test_elementwise[32, 4, False](DimList(4, 2, 2, 2))

    # CHECK-LABEL: == Testing 4D blocking:
    # CHECK-NOT: ERROR
    print("== Testing 4D blocking:")
    test_elementwise[32, 4, False](DimList(4, 2, 2, 2))

    # CHECK-LABEL: == Testing 5D:
    # CHECK-NOT: ERROR
    print("== Testing 5D:")
    test_elementwise[32, 5, False](DimList(4, 2, 1, 2, 2))

    # CHECK-LABEL: == Testing 5D blocking:
    # CHECK-NOT: ERROR
    print("== Testing 5D blocking:")
    test_elementwise[32, 5, True](DimList(4, 2, 1, 2, 2))

    # CHECK-LABEL: == Testing large:
    # CHECK-NOT: ERROR
    print("== Testing large:")
    test_elementwise[131072, 2, False](DimList(1024, 128))

    # CHECK-LABEL: == Testing large blocking:
    # CHECK-NOT: ERROR
    print("== Testing large blocking:")
    test_elementwise[131072, 2, True](DimList(1024, 128))

    # CHECK-LABEL: == test_elementwise_implicit_runtime
    # CHECK-NOT: ERROR
    # CHECK: OK
    test_elementwise_implicit_runtime()

    # CHECK-LABEL: == Testing indices conversion:
    # CHECK: (0, 0, 1, 4)
    # CHECK: (0, 2, 0, 0)
    # CHECK: (2, 2, 0, 0)
    # CHECK: (2, 0, 0, 0)
    # CHECK: (0, 0, 0, 0)
    test_indices_conversion()
