# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# RUN: %mojo-no-debug %s | FileCheck %s

from layout.dynamic_tuple import *
from layout.int_tuple import *


# CHECK-LABEL: test_product2
def test_product2():
    print("== test_product2")
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


# CHECK-LABEL: test_product3
def test_product3():
    print("== test_product3")
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


# CHECK-LABEL: test_product5
def test_product5():
    print("== test_product5")
    # CHECK: (1, 3, 5, 7, 9)
    # CHECK: (1, 3, 5, 7, 10)
    # CHECK: (1, 3, 5, 8, 9)
    # CHECK: (1, 3, 5, 8, 10)
    # CHECK: (1, 3, 6, 7, 9)
    # CHECK: (1, 3, 6, 7, 10)
    # CHECK: (1, 3, 6, 8, 9)
    # CHECK: (1, 3, 6, 8, 10)
    # CHECK: (1, 4, 5, 7, 9)
    # CHECK: (1, 4, 5, 7, 10)
    # CHECK: (1, 4, 5, 8, 9)
    # CHECK: (1, 4, 5, 8, 10)
    # CHECK: (1, 4, 6, 7, 9)
    # CHECK: (1, 4, 6, 7, 10)
    # CHECK: (1, 4, 6, 8, 9)
    # CHECK: (1, 4, 6, 8, 10)
    # CHECK: (2, 3, 5, 7, 9)
    # CHECK: (2, 3, 5, 7, 10)
    # CHECK: (2, 3, 5, 8, 9)
    # CHECK: (2, 3, 5, 8, 10)
    # CHECK: (2, 3, 6, 7, 9)
    # CHECK: (2, 3, 6, 7, 10)
    # CHECK: (2, 3, 6, 8, 9)
    # CHECK: (2, 3, 6, 8, 10)
    # CHECK: (2, 4, 5, 7, 9)
    # CHECK: (2, 4, 5, 7, 10)
    # CHECK: (2, 4, 5, 8, 9)
    # CHECK: (2, 4, 5, 8, 10)
    # CHECK: (2, 4, 6, 7, 9)
    # CHECK: (2, 4, 6, 7, 10)
    # CHECK: (2, 4, 6, 8, 9)
    # CHECK: (2, 4, 6, 8, 10)
    for i in product(
        IntTuple(1, 2),
        IntTuple(3, 4),
        IntTuple(5, 6),
        IntTuple(7, 8),
        IntTuple(9, 10),
    ):
        print(i)


# CHECK-LABEL: test_product_unequal
def test_product_unequal():
    print("== test_product_unequal")
    # CHECK: (1, 3, 5, 1, 9)
    # CHECK: (1, 3, 5, 1, 10)
    # CHECK: (1, 3, 5, 7, 9)
    # CHECK: (1, 3, 5, 7, 10)
    # CHECK: (1, 3, 5, 8, 9)
    # CHECK: (1, 3, 5, 8, 10)
    # CHECK: (1, 4, 5, 1, 9)
    # CHECK: (1, 4, 5, 1, 10)
    # CHECK: (1, 4, 5, 7, 9)
    # CHECK: (1, 4, 5, 7, 10)
    # CHECK: (1, 4, 5, 8, 9)
    # CHECK: (1, 4, 5, 8, 10)
    # CHECK: (1, 5, 5, 1, 9)
    # CHECK: (1, 5, 5, 1, 10)
    # CHECK: (1, 5, 5, 7, 9)
    # CHECK: (1, 5, 5, 7, 10)
    # CHECK: (1, 5, 5, 8, 9)
    # CHECK: (1, 5, 5, 8, 10)
    for i in product(
        IntTuple(1),
        IntTuple(3, 4, 5),
        IntTuple(5),
        IntTuple(1, 7, 8),
        IntTuple(9, 10),
    ):
        print(i)


def main():
    test_product2()
    test_product3()
    test_product5()
    test_product_unequal()
