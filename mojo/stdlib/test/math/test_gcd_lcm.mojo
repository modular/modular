# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# RUN: %mojo %s | FileCheck %s

from math import gcd, lcm


# CHECK-LABEL: test_gcd
fn test_gcd():
    print("== test_gcd")

    # CHECK: 0
    print(gcd(0, 0))

    # CHECK: 4
    print(gcd(0, 4))

    # CHECK: 5
    print(gcd(5, 0))

    # CHECK: 1
    print(gcd(7, 13))

    # CHECK: 4
    print(gcd(8, 12))

    # CHECK: 12
    print(gcd(48, 36))


# CHECK-LABEL: test_lcm
fn test_lcm():
    print("== test_lcm")

    # CHECK: 0
    print(lcm(0, 0))

    # CHECK: 0
    print(lcm(0, 4))

    # CHECK: 0
    print(lcm(5, 0))

    # CHECK: 6
    print(lcm(2, 3))

    # CHECK: 91
    print(lcm(7, 13))

    # CHECK: 24
    print(lcm(8, 12))

    # CHECK: 144
    print(lcm(48, 36))


fn main():
    test_gcd()
    test_lcm()
