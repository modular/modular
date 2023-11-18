# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# RUN: %mojo -debug-level full %s | FileCheck %s

from sys.info import simdwidthof

from Gemv import gemv
from memory.buffer import Buffer, NDBuffer
from runtime.llcl import OwningOutputChainPtr, Runtime

from utils.list import Dim, DimList


# CHECK-LABEL: test_gemv
fn test_gemv():
    print("== test_gemv")
    alias type = DType.float32
    alias simd_width = simdwidthof[type]()
    alias m = 102
    alias k = 26
    alias n = 1

    let lhs = NDBuffer[2, DimList(m, k), type].stack_allocation()
    let rhs = Buffer[Dim(k), type].stack_allocation()
    let out = Buffer[Dim(m), type].stack_allocation()
    lhs.fill(1)
    rhs.fill(1)
    out.zero()
    gemv[simd_width](out, lhs, rhs)

    for i in range(len(out)):
        if out[i] != k:
            print(out[i])
            print("Error")
    # CHECK-NOT: Error


fn main():
    test_gemv()
