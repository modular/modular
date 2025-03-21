# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# RUN: %mojo-no-debug %s | FileCheck %s

from random import seed

from buffer import NDBuffer
from buffer.dimlist import DimList
from nn.randn import random_normal


fn test_random_normal():
    seed(0)

    alias out_shape = DimList(2, 2)
    var output_stack = InlineArray[Float32, 4](uninitialized=True)
    var output = NDBuffer[DType.float32, 2, _, out_shape](output_stack)
    output.fill(0)

    random_normal[2, DType.float32, out_shape, 0.0, 1.0](output)
    # CHECK-LABEL: == test_random_normal
    print("== test_random_normal")


fn main():
    test_random_normal()
