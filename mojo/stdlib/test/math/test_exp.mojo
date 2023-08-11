# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# RUN: %mojo %s | FileCheck %s

from DType import DType
from IO import print
from math import exp
from SIMD import Float32, Float64


# CHECK-LABEL: test_exp_float32
fn test_exp_float32():
    print("== test_exp_float32")

    # CHECK: 0.90483{{[0-9]+}}
    print(exp(Float32(-0.1)))

    # CHECK: 1.10517{{[0-9]+}}
    print(exp(Float32(0.1)))

    # CHECK: 7.38905{{[0-9]+}}
    print(exp(Float32(2)))

    # CHECK: inf
    print(exp(Float32(89)))

    # CHECK: inf
    print(exp(Float32(108.5230)))


# CHECK-LABEL: test_exp_float64
fn test_exp_float64():
    print("== test_exp_float64")

    # CHECK: 0.90483{{[0-9]+}}
    print(exp(Float64(-0.1)))

    # CHECK: 1.10517{{[0-9]+}}
    print(exp(Float64(0.1)))

    # CHECK: 7.38905{{[0-9]+}}
    print(exp(Float64(2)))

    # CHECK: 1.6516{{[0-9]+}}e+38
    print(exp(Float64(88)))

    # CHECK: 1.3518{{[0-9]+}}e+47
    print(exp(Float64(108.5230)))


fn main():
    test_exp_float32()
    test_exp_float64()
