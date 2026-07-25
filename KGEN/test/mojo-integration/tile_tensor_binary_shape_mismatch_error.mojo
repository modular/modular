# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# TileTensor's in-place binary ops require same-rank operands to have
# identical shapes; there is no broadcasting between same-rank tensors.

# RUN: not %mojo %s 2>&1 | FileCheck %s

from layout import TileTensor, row_major


def main():
    var a_data = Array[Float32, 4](fill=0)
    var b_data = Array[Float32, 6](fill=0)
    var a = TileTensor(a_data, row_major[2, 2]())
    var b = TileTensor(b_data, row_major[2, 3]())

    # CHECK: requires shape to be the same for tensors of the same rank
    a += b
