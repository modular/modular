# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# UNSUPPORTED: asan
# RUN: %mojo %s | FileCheck %s

from layout.dynamic_tuple import *
from layout.int_tuple import *


def main():
    # CHECK: (1, 3)
    # CHECK: (1, 4)
    # CHECK: (2, 3)
    # CHECK: (2, 4)
    for i in product(IntTuple(1, 2), IntTuple(3, 4)):
        print(i)
