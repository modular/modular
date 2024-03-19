# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# RUN: %mojo -debug-level full %s | FileCheck %s

from random import seed

from buffer import NDBuffer
from nn.randn import random_normal

from buffer.list import DimList


fn test_random_normal():
    seed(0)

    alias out_shape = DimList(2, 2)
    var output = NDBuffer[DType.float32, 2, out_shape].stack_allocation()
    output.fill(0)

    random_normal[2, DType.float32, out_shape, 0.0, 1.0](output)
    # CHECK-LABEL: == test_random_normal
    print("== test_random_normal")


fn main():
    test_random_normal()
