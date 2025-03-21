# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# RUN: %mojo-no-debug %s | FileCheck %s

from sys.info import simdwidthof

from buffer import NDBuffer
from buffer.dimlist import Dim, DimList
from nn.softmax import logsoftmax, softmax_2_pass


# CHECK-LABEL: test_logsoftmax
fn test_logsoftmax() raises:
    print("== test_logsoftmax")
    alias type = DType.float32
    alias simd_width = simdwidthof[type]()

    fn logsoftmax_test_nd[rank: Int, shape: DimList]() raises:
        var in_stack = InlineArray[Scalar[type], Int(shape.product())](
            uninitialized=True
        )
        var in_buf = NDBuffer[type, rank, _, shape](in_stack)
        var out_stack = InlineArray[Scalar[type], Int(shape.product())](
            uninitialized=True
        )
        var out_buf = NDBuffer[type, rank, _, shape](out_stack)
        var in_buf_flat = in_buf.flatten()
        var out_buf_flat = out_buf.flatten()
        out_buf.zero()
        for i in range(len(in_buf_flat)):
            in_buf_flat[i] = i

        logsoftmax[type, simd_width, rank, shape](in_buf, out_buf, rank - 1)

        for i in range(len(out_buf_flat)):
            print(out_buf_flat[i])

    logsoftmax_test_nd[1, DimList(5)]()

    # CHECK: -4.45191{{[0-9]+}}
    # CHECK-NEXT: -3.451914{{[0-9]+}}
    # CHECK-NEXT: -2.451914{{[0-9]+}}
    # CHECK-NEXT: -1.451914{{[0-9]+}}
    # CHECK-NEXT: -0.451914{{[0-9]+}}

    logsoftmax_test_nd[2, DimList(3, 4)]()

    # CHECK: -3.440189{{[0-9]+}}
    # CHECK-NEXT: -2.440189{{[0-9]+}}
    # CHECK-NEXT: -1.440189{{[0-9]+}}
    # CHECK-NEXT: -0.440189{{[0-9]+}}
    # CHECK-NEXT: -3.440189{{[0-9]+}}
    # CHECK-NEXT: -2.440189{{[0-9]+}}
    # CHECK-NEXT: -1.440189{{[0-9]+}}
    # CHECK-NEXT: -0.440189{{[0-9]+}}
    # CHECK-NEXT: -3.440189{{[0-9]+}}
    # CHECK-NEXT: -2.440189{{[0-9]+}}
    # CHECK-NEXT: -1.440189{{[0-9]+}}
    # CHECK-NEXT: -0.440189{{[0-9]+}}


# CHECK-LABEL: test_softmax_2pass
fn test_softmax_2pass():
    print("== test_softmax_2pass")
    alias type = DType.float32
    alias simd_width = simdwidthof[type]()
    alias sz = 5

    var in_stack = InlineArray[Scalar[type], sz](uninitialized=True)
    var in_buf = NDBuffer[type, 1, _, sz](in_stack)
    for i in range(sz):
        in_buf[i] = i
    var out_stack = InlineArray[Scalar[type], sz](uninitialized=True)
    var out_buf = NDBuffer[type, 1, _, sz](out_stack)
    out_buf.zero()

    softmax_2_pass[simd_width, sz, type](out_buf, in_buf)

    for i in range(sz):
        print(out_buf[i])

    # CHECK: 0.01165{{[0-9]+}}
    # CHECK-NEXT: 0.03168{{[0-9]+}}
    # CHECK-NEXT: 0.08612{{[0-9]+}}
    # CHECK-NEXT: 0.23412{{[0-9]+}}
    # CHECK-NEXT: 0.63640{{[0-9]+}}


fn main() raises:
    test_logsoftmax()
    test_softmax_2pass()
