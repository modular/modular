# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# RUN: %mojo -debug-level full %s | FileCheck %s

from math.numerics import FPUtils


# CHECK-LABEL: test_numerics
fn test_numerics():
    print("== test_numerics")

    # CHECK: 23
    print(FPUtils[DType.float32].mantissa_width())

    # CHECK: 52
    print(FPUtils[DType.float64].mantissa_width())

    # CHECK: 127
    print(FPUtils[DType.float32].exponent_bias())

    # CHECK: 1023
    print(FPUtils[DType.float64].exponent_bias())


fn main():
    test_numerics()
