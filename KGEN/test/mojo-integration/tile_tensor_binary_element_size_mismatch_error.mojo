# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# TileTensor's in-place binary ops accept an rhs with a different storage
# policy, but the two policies must agree on logical element size. A
# vectorized rhs (element_width=4) against a scalar destination is rejected
# at compile time.

# RUN: not %mojo %s 2>&1 | FileCheck %s

from layout import TileTensor, row_major


def main():
    var a_data = Array[Float32, 8](fill=0)
    var b_data = Array[Float32, 8](fill=0)
    var a = TileTensor(a_data, row_major[2, 4]())
    var b = TileTensor(b_data, row_major[2, 4]())

    # CHECK: in-place binary ops require operands with the same element size
    a += b.vectorize[1, 4]()
