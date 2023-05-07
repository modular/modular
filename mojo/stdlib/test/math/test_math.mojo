# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# RUN: mojo %s | FileCheck %s

from Math import factorial
from IO import print

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
    test_factorial()
