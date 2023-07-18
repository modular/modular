# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# RUN: %mojo %s | FileCheck %s

from Softmax import softmax_2_pass
from Buffer import Buffer
from List import Dim
from IO import print
from TargetInfo import simdwidthof
from Range import range
from DType import DType


# CHECK-LABEL: test_softmax_2pass
fn test_softmax_2pass():
    print("== test_softmax_2pass")
    alias type = DType.float32
    alias simd_width = simdwidthof[type]()
    alias sz = 5

    let in_buf = Buffer[sz, type].stack_allocation()
    for i in range(sz):
        in_buf[i] = i
    let out_buf = Buffer[sz, type].stack_allocation()
    out_buf.zero()

    softmax_2_pass[simd_width, sz, type](out_buf, in_buf)

    for i in range(sz):
        print(out_buf[i])

    # CHECK: 0.01165{{[0-9]+}}
    # CHECK-NEXT: 0.03168{{[0-9]+}}
    # CHECK-NEXT: 0.08612{{[0-9]+}}
    # CHECK-NEXT: 0.23412{{[0-9]+}}
    # CHECK-NEXT: 0.63640{{[0-9]+}}


fn main():
    test_softmax_2pass()
