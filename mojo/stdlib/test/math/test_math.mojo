# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# RUN: mojo %s | FileCheck %s

from Math import factorial, sin, cos
from IO import print
from SIMD import Float16, Float32

# CHECK-LABEL: test_sin
fn test_sin():
    print("== test_sin")

    # CHECK: 0.84130859375
    print(sin(Float16(1.0)))

    # CHECK: 0.841470956802{{[0-9]+}}
    print(sin(Float32(1.0)))


# CHECK-LABEL: test_cos
fn test_cos():
    print("== test_cos")

    # CHECK: 0.54052734375
    print(cos(Float16(1.0)))

    # CHECK: 0.540302276611{{[0-9]+}}
    print(cos(Float32(1.0)))


# CHECK-LABEL: test_factorial
fn test_factorial():
    print("== test_factorial")

    # CHECK: 1
    print(factorial(0))

    # CHECK: 1
    print(factorial(1))

    # CHECK: 1307674368000
    print(factorial(15))

    # CHECK: 2432902008176640000
    print(factorial(20))


fn main():
    test_sin()
    test_cos()
    test_factorial()
