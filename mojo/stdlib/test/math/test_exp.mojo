# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# RUN: mojo %s | FileCheck %s

from DType import DType
from IO import print
from Math import exp
from SIMD import SIMD


# CHECK-LABEL: test_exp_float32
fn test_exp_float32():
    print("== test_exp_float32")

    # CHECK: 0.90483{{[0-9]+}}
    print(exp[1, DType.float32](SIMD[DType.float32, 1](-0.1)))

    # CHECK: 1.10517{{[0-9]+}}
    print(exp[1, DType.float32](SIMD[DType.float32, 1](0.1)))

    # CHECK: 7.38905{{[0-9]+}}
    print(exp(SIMD[DType.float32, 1](2)))


# CHECK-LABEL: test_exp_float64
fn test_exp_float64():
    print("== test_exp_float64")

    # CHECK: 0.90483{{[0-9]+}}
    print(exp[1, DType.float64](SIMD[DType.float64, 1](-0.1)))

    # CHECK: 1.10517{{[0-9]+}}
    print(exp[1, DType.float64](SIMD[DType.float64, 1](0.1)))

    # CHECK: 7.38905{{[0-9]+}}
    print(exp(SIMD[DType.float64, 1](2)))


fn main():
    test_exp_float32()
    test_exp_float64()
