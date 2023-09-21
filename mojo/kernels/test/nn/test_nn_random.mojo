# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# RUN: %mojo -debug-level full %s | FileCheck %s

from random import seed
from memory.buffer import NDBuffer
from RandomNormal import random_normal


fn test_random_normal():
    seed(0)

    alias out_shape = DimList(2, 2)
    var output = NDBuffer[2, out_shape, DType.float32].stack_allocation()
    output.fill(0)

    random_normal[2, DType.float32, out_shape, 0.0, 1.0](output)
    # CHECK-LABEL: == test_random_normal
    print("== test_random_normal")


fn main():
    test_random_normal()
