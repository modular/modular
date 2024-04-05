# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# UNSUPPORTED: asan
# RUN: %mojo %s | FileCheck %s

from layout.dynamic_tuple import *
from layout.int_tuple import *


def test_product2():
    # CHECK: (1, 3)
    # CHECK: (1, 4)
    # CHECK: (2, 3)
    # CHECK: (2, 4)
    for i in product(IntTuple(1, 2), IntTuple(3, 4)):
        print(i)

    alias prd = product(IntTuple(1, 2), IntTuple(3, 4))

    # CHECK: (1, 3)
    print(prd[0])
    # CHECK: (1, 4)
    print(prd[1])
    # CHECK: (2, 3)
    print(prd[2])
    # CHECK: (2, 4)
    print(prd[3])


def test_product3():
    # CHECK: (1, 3, 5)
    # CHECK: (1, 3, 6)
    # CHECK: (1, 4, 5)
    # CHECK: (1, 4, 6)
    # CHECK: (2, 3, 5)
    # CHECK: (2, 3, 6)
    # CHECK: (2, 4, 5)
    # CHECK: (2, 4, 6)
    for i in product(IntTuple(1, 2), IntTuple(3, 4), IntTuple(5, 6)):
        print(i)

    alias prd = product(IntTuple(1, 2), IntTuple(3, 4), IntTuple(5, 6))

    # CHECK: (1, 3, 5)
    print(prd[0])
    # CHECK: (1, 3, 6)
    print(prd[1])
    # CHECK: (2, 4, 5)
    print(prd[6])
    # CHECK: (2, 4, 6)
    print(prd[7])


def main():
    test_product2()
    test_product3()
