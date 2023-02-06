# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# RUN: lit %s | FileCheck %s

from Assert import assert_param
from DType import DType
from Int import Int
from IO import print
from Math import ldexp, _bits_to_float
from Numerics import FPUtils
from SIMD import SIMD


# CHECK-LABEL: test_ldexp
fn test_ldexp():
    print("== test_ldexp\n")

    # CHECK: 24.0
    print(ldexp(SIMD[1, DType.f32.value](1.5), 4))

    # CHECK: 24.0
    print(ldexp(SIMD[1, DType.f64.value](1.5), SIMD[1, DType.si32.value](4)))


fn main() -> Int:
    test_ldexp()
    return 0
