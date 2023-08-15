# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# RUN: %mojo -debug-level full %s | FileCheck %s

from IO import print
from math import ldexp


# CHECK-LABEL: test_ldexp
fn test_ldexp():
    print("== test_ldexp")

    # CHECK: 24.0
    print(ldexp(Float32(1.5), 4))

    # CHECK: 24.0
    print(ldexp(Float64(1.5), SIMD[DType.int32, 1](4)))


fn main():
    test_ldexp()
