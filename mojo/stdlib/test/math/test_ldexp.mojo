# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# RUN: mojo %s | FileCheck %s

from Assert import assert_param
from DType import DType
from IO import print
from Math import ldexp, _bits_to_float
from Numerics import FPUtils
from SIMD import SIMD


# CHECK-LABEL: test_ldexp
fn test_ldexp():
    print("== test_ldexp")

    # CHECK: 24.0
    print(ldexp(SIMD[DType.float32, 1](1.5), 4))

    # CHECK: 24.0
    print(ldexp(SIMD[DType.float64, 1](1.5), SIMD[DType.int32, 1](4)))


fn main():
    test_ldexp()
