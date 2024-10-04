# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# RUN: mojo %s | FileCheck %s

from tensor_internal.tensor_shape import TensorShape


def main():
    var weights: TensorShape = (1, 2)
    # CHECK: 1x2
    print(weights)

    # CHECK: 3x4
    weights = (UInt(3), UInt(4))
    print(weights)
